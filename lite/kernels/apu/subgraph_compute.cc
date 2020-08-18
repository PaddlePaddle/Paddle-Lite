// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/apu/subgraph_compute.h"
#include <dlfcn.h>
#include <sys/time.h>
#include <time.h>
#include <utility>
#include "lite/backends/apu/device.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/apu/bridges/graph.h"
#include "lite/kernels/apu/bridges/paddle_use_bridges.h"
#include "lite/kernels/apu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace apu {

bool SubgraphEngine::BuildDeviceProgram() {
  if (!origin_program_) {
    BuildOriginProgram();
  }

  unsigned int version;
  Neuron_getVersion(&version);
  VLOG(3) << "Neuron Adapter version: " << version;

  int status = 0;
  subgraph::apu::Graph graph;
  int neuron_errCode = NeuronModel_create(&model_);
  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "[APU] Failed to create the neuron model!";
    return false;
  }
  graph.set_model(model_);
  graph.set_input_names(input_names_);
  graph.set_output_names(output_names_);

  // Convert all of ops and their input vars and weights and added into the APU
  // NIR graph
  const auto& bridges = subgraph::Registry::Instance();
  const auto& insts = origin_program_->instructions(kRootBlockIdx);
  for (auto& inst : insts) {
    auto op = const_cast<OpLite*>(inst.op());
    CHECK(op);
    op->CheckShape();
    op->InferShape();
    std::string op_type = op->op_info()->Type();
    if (!bridges.Exists(op_type, TARGET(kAPU))) {
      return false;
    }

    auto kernel = inst.kernel();
    status |=
        bridges.Select(op_type, TARGET(kAPU))(reinterpret_cast<void*>(&graph),
                                              const_cast<OpLite*>(op),
                                              const_cast<KernelBase*>(kernel));
    if (subgraph::CHECK_FAILED(status)) {
      return false;
    }
  }

  // Get the index of input tensors
  std::vector<uint32_t> input_indices;
  for (int i = 0; i < input_names_.size(); i++) {
    CHECK(graph.Has(input_names_[i])) << "[APU] Failed to find input node "
                                      << input_names_[i];
    auto index = graph.Get(input_names_[i])->index();
    input_indices.push_back(index);
    VLOG(3) << "[APU] Input[" << i << "] name " << input_names_[i] << " dims "
            << origin_itensors_[i]->dims() << " index " << index;
  }

  // Get the index of output tensors
  std::vector<uint32_t> output_indices;
  for (int i = 0; i < output_names_.size(); i++) {
    CHECK(graph.Has(output_names_[i])) << "[APU] Failed to find output node "
                                       << output_names_[i];
    origin_otensors_[i]->mutable_data<int8_t>();
    auto index = graph.Get(output_names_[i])->index();
    output_indices.push_back(index);
    VLOG(3) << "[APU] Output[" << i << "] name " << output_names_[i] << " dims "
            << origin_otensors_[i]->dims() << " index " << index;
  }

  // Indentify the input and output tensors of the neuron model
  NeuronModel_identifyInputsAndOutputs(model_,
                                       input_indices.size(),
                                       &input_indices[0],
                                       output_indices.size(),
                                       &output_indices[0]);
  neuron_errCode = NeuronModel_finish(model_);
  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "[APU] Fail to create NIR model:" << neuron_errCode;
    return false;
  }
  VLOG(3) << "[APU] APU NIR model created!";

  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };
  auto start_time = GetCurrentUS();
  compilation_ = lite::apu::Device::Global().Build(model_);
  if (compilation_ == nullptr) {
    LOG(WARNING) << "[APU] Build APU DLA model failed!";
    return false;
  }
  VLOG(3) << "[APU] APU DLA model created, Build cost "
          << GetCurrentUS() - start_time << " us";
  return true;
}

bool SubgraphEngine::LaunchDeviceProgram() {
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };

  auto start_time = GetCurrentUS();
  NeuronExecution* run = NULL;
  int neuron_errCode = NeuronExecution_create(compilation_, &run);
  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "[APU] Build APU runtime failed!";
    return false;
  }

  // Set input buffer
  for (size_t i = 0; i < origin_itensors_.size(); i++) {
    auto origin_data = origin_itensors_[i]->mutable_data<int8_t>();
    auto converted_data = reinterpret_cast<uint8_t*>(origin_data);
    for (int j = 0; j < origin_itensors_[i]->data_size(); j++) {
      converted_data[j] =
          static_cast<uint8_t>(static_cast<int16_t>(origin_data[j]) + 128);
    }
    NeuronExecution_setInput(
        run, i, NULL, converted_data, origin_itensors_[i]->memory_size());
  }

  // Set output buffer
  for (size_t i = 0; i < origin_otensors_.size(); i++) {
    NeuronExecution_setOutput(
        run,
        i,
        NULL,
        reinterpret_cast<void*>(origin_otensors_[i]->raw_data()),
        origin_otensors_[i]->memory_size());
  }

  neuron_errCode = NeuronExecution_compute(run);
  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "Fail to run execution!" << neuron_errCode;
    return false;
  }

  for (size_t i = 0; i < origin_otensors_.size(); i++) {
    auto converted_data = origin_otensors_[i]->mutable_data<int8_t>();
    auto origin_data = reinterpret_cast<uint8_t*>(converted_data);
    for (int j = 0; j < origin_otensors_[i]->data_size(); j++) {
      converted_data[j] =
          static_cast<int8_t>(static_cast<int16_t>(origin_data[j]) - 128);
    }
  }
  NeuronExecution_free(run);
  VLOG(3) << "[APU] Process cost " << GetCurrentUS() - start_time << " us";
  return true;
}

SubgraphEngine::~SubgraphEngine() {
  if (compilation_) {
    NeuronCompilation_free(compilation_);
  }
  if (model_) {
    NeuronModel_free(model_);
  }
}

void SubgraphCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  engine_.reset(new SubgraphEngine(ctx_.get(),
                                   param.block_idx,
                                   param.program_desc,
                                   param.exec_scope,
                                   param.input_data_names,
                                   param.output_data_names));
  CHECK(engine_);
}

void SubgraphCompute::Run() {
  CHECK(engine_);
  engine_->Run();
}

}  // namespace apu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(subgraph,
                     kAPU,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::apu::SubgraphCompute,
                     def)
    .BindInput("Inputs",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Outputs",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt8),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
