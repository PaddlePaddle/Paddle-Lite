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

inline void* LoadFunc(void* libHandle, const char* name) {
  CHECK(libHandle != nullptr);
  CHECK(name != nullptr);
  void* fn = dlsym(libHandle, name);
  if (fn == nullptr) {
    LOG(WARNING) << "Unable to open Neuron Runtime function [" << name
                 << "] Because " << dlerror();
  }
  return fn;
}

#define LOAD_FUNCTIONS(libHandle, FUNC_NAME, VARIABLE_NAME) \
  FUNC_NAME VARIABLE_NAME =                                 \
      reinterpret_cast<FUNC_NAME>(LoadFunc(libHandle, #FUNC_NAME));

int SubgraphEngine::BuildDeviceProgram() {
  typedef int (*Neuron_getVersion)(uint32_t * version);
  typedef int (*NeuronModel_create)(NeuronModel * *model);
  typedef void (*NeuronModel_free)(NeuronModel * model);
  typedef int (*NeuronModel_finish)(NeuronModel * model);
  typedef int (*NeuronModel_identifyInputsAndOutputs)(NeuronModel * model,
                                                      uint32_t inputCount,
                                                      const uint32_t* inputs,
                                                      uint32_t outputCount,
                                                      const uint32_t* outputs);

  // Open the share library
  libHandle_ = dlopen("libneuron_adapter.so", RTLD_LAZY);
  if (libHandle_ == nullptr) {
    LOG(WARNING) << "Failed to open libneuron_adapter.so. " << dlerror();
    return subgraph::FAILED;
  }

  LOAD_FUNCTIONS(libHandle_, Neuron_getVersion, neuron_getVersion)
  LOAD_FUNCTIONS(libHandle_, NeuronModel_create, neuron_model_create)
  LOAD_FUNCTIONS(libHandle_, NeuronModel_finish, neuron_model_finish)
  LOAD_FUNCTIONS(libHandle_,
                 NeuronModel_identifyInputsAndOutputs,
                 neuron_model_identifyInputsAndOutputs)

  unsigned int version;
  (*neuron_getVersion)(&version);
  VLOG(3) << "Neuron Adapter version: " << version;

  int status = 0;
  subgraph::apu::Graph graph;
  int neuron_errCode = (*neuron_model_create)(&model_);
  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "Fail to create model";
    return subgraph::FAILED;
  }
  graph.set_libHandle(libHandle_);
  graph.set_model(model_);
  graph.set_input_names(input_names_);
  graph.set_output_names(output_names_);

  // Convert all of ops and their input vars and weights and added into the APU
  // NIR graph
  const auto& bridges = subgraph::Registry::Instance();
  for (auto& inst : origin_program_) {
    auto op = const_cast<OpLite*>(inst.op());
    CHECK(op);
    op->CheckShape();
    op->InferShape();
    std::string op_type = op->op_info()->Type();
    if (!bridges.Exists(op_type, TARGET(kAPU))) {
      return subgraph::FAILED;
    }

    auto kernel = inst.kernel();
    status |=
        bridges.Select(op_type, TARGET(kAPU))(reinterpret_cast<void*>(&graph),
                                              const_cast<OpLite*>(op),
                                              const_cast<KernelBase*>(kernel));
    if (subgraph::CHECK_FAILED(status)) {
      return subgraph::FAILED;
    }
  }

  // Get input tensor
  std::vector<uint32_t> ins;
  origin_itensors_.resize(input_names_.size());
  origin_idims_.resize(input_names_.size());
  for (int i = 0; i < input_names_.size(); i++) {
    origin_itensors_[i] = scope_->FindMutableTensor(input_names_[i]);
    CHECK(origin_itensors_[i]);
    origin_idims_[i] = origin_itensors_[i]->dims();
    VLOG(3) << "subgraph input name: " << i << ", " << input_names_[i] << ":"
            << origin_idims_[i].production();
    // Get input index
    int idx;
    if (graph.Has(input_names_[i])) {
      ins.push_back(graph.Get(input_names_[i])->index());
      VLOG(3) << "input idx: " << graph.Get(input_names_[i])->index();
    } else {
      LOG(WARNING) << "Fail to find input: " << input_names_[i];
      return subgraph::FAILED;
    }
  }

  // Get output tensor
  std::vector<uint32_t> outs;
  origin_otensors_.resize(output_names_.size());
  origin_odims_.resize(output_names_.size());
  for (int i = 0; i < output_names_.size(); i++) {
    origin_otensors_[i] = scope_->FindMutableTensor(output_names_[i]);
    CHECK(origin_otensors_[i]);
    origin_odims_[i] = origin_otensors_[i]->dims();
    VLOG(3) << "subgraph output name: " << i << ", " << output_names_[i] << ":"
            << origin_odims_[i].production();
    origin_otensors_[i]->mutable_data<int8_t>();
    // Get input index
    if (graph.Has(output_names_[i])) {
      outs.push_back(graph.Get(output_names_[i])->index());
      VLOG(3) << "output idx: " << graph.Get(output_names_[i])->index();
    } else {
      LOG(WARNING) << "Fail to find output: " << output_names_[i];
      return subgraph::FAILED;
    }
  }

  VLOG(3) << "ins size: " << ins.size() << " outs size:" << outs.size();
  // Set subgraph input/output
  (*neuron_model_identifyInputsAndOutputs)(
      model_, ins.size(), &ins[0], outs.size(), &outs[0]);
  neuron_errCode = (*neuron_model_finish)(model_);
  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "Fail to create NIR model:" << neuron_errCode;
    return subgraph::FAILED;
  }
  VLOG(3) << "[APU] APU NIR model created!";

  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };
  auto start_time = GetCurrentUS();
  compilation_ = lite::apu::Device::Global().Build(libHandle_, model_);
  if (compilation_ == nullptr) {
    LOG(WARNING) << "[APU] Build APU DLA model failed!";
    return subgraph::FAILED;
  }
  VLOG(3) << "[APU] APU DLA model created, Build cost "
          << GetCurrentUS() - start_time << " us";

  return status;
}

int SubgraphEngine::LaunchDeviceProgram() {
  typedef int (*NeuronExecution_create)(NeuronCompilation * compilation,
                                        NeuronExecution * *execution);
  typedef void (*NeuronExecution_free)(NeuronExecution * execution);
  typedef int (*NeuronExecution_setInput)(NeuronExecution * execution,
                                          int32_t index,
                                          const NeuronOperandType* type,
                                          const void* buffer,
                                          size_t length);
  typedef int (*NeuronExecution_setOutput)(NeuronExecution * execution,
                                           int32_t index,
                                           const NeuronOperandType* type,
                                           void* buffer,
                                           size_t length);
  typedef int (*NeuronExecution_compute)(NeuronExecution * execution);

  LOAD_FUNCTIONS(libHandle_, NeuronExecution_create, neuron_execution_create)
  LOAD_FUNCTIONS(libHandle_, NeuronExecution_free, neuron_execution_free)
  LOAD_FUNCTIONS(
      libHandle_, NeuronExecution_setInput, neuron_execution_setInput)
  LOAD_FUNCTIONS(
      libHandle_, NeuronExecution_setOutput, neuron_execution_setOutput)
  LOAD_FUNCTIONS(libHandle_, NeuronExecution_compute, neuron_execution_compute)

  NeuronExecution* run1 = NULL;
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };

  auto start_time = GetCurrentUS();
  int neuron_errCode = (*neuron_execution_create)(compilation_, &run1);
  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "[APU] Build APU runtime failed!";
    return subgraph::FAILED;
  }

  // Set input buffer
  Tensor input_temp;
  for (size_t i = 0; i < origin_itensors_.size(); i++) {
    input_temp.Resize({origin_idims_[i]});
    uint8_t* input_data = input_temp.mutable_data<uint8_t>();
    memcpy(input_data,
           origin_itensors_[i]->raw_data(),
           origin_itensors_[i]->memory_size());
    for (int j = 0; j < origin_itensors_[i]->data_size(); j++) {
      input_data[j] += (uint8_t)128;
    }
    (*neuron_execution_setInput)(
        run1, i, NULL, input_data, origin_itensors_[i]->memory_size());
  }

  // Set output buffer
  for (size_t i = 0; i < origin_otensors_.size(); i++) {
    (*neuron_execution_setOutput)(
        run1,
        i,
        NULL,
        reinterpret_cast<void*>(origin_otensors_[i]->raw_data()),
        origin_otensors_[i]->memory_size());
  }

  neuron_errCode = (*neuron_execution_compute)(run1);
  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "Fail to run execution!" << neuron_errCode;
    return subgraph::FAILED;
  }

  for (size_t i = 0; i < origin_otensors_.size(); i++) {
    int8_t* output_data = origin_otensors_[i]->mutable_data<int8_t>();
    VLOG(3) << "output size:" << origin_otensors_[i]->memory_size();
    for (int j = 0; j < origin_otensors_[i]->data_size(); j++) {
      output_data[j] -= (int8_t)128;
    }
  }
  (*neuron_execution_free)(run1);
  VLOG(3) << "[APU] Process cost " << GetCurrentUS() - start_time << " us";
  return 0;
}

void SubgraphCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  engine_.reset(new SubgraphEngine(ctx_.get(),
                                   param.sub_block_idx,
                                   param.sub_block_desc,
                                   param.input_data_names,
                                   param.output_data_names,
                                   param.scope));
  CHECK(engine_);
  engine_->Build();
}

void SubgraphCompute::Run() {
  CHECK(engine_);
  engine_->Launch();
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
