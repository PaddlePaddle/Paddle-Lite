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
#include "lite/utils/io.h"
#include "lite/utils/md5.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace apu {

// Generate the model name by using md5 hashes based on:
// 1. the sorted variable input names
// 2. the shapes of the origin input tensors
// 3. the sorted variable output names
std::string DeviceProgram::GenerateModelName(
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<std::vector<int64_t>>& origin_idims) {
  std::ostringstream os;
  CHECK_EQ(input_names.size(), origin_idims.size());
  for (int i = 0; i < input_names.size(); i++) {
    os << input_names[i];
    for (auto dim : origin_idims[i]) {
      os << dim;
    }
  }
  for (auto output_name : output_names) {
    os << output_name;
  }
  return MD5(os.str());
}

// Deserialize the generated model
bool DeviceProgram::LoadFromCacheFile(
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<std::vector<int64_t>>& origin_idims,
    const std::string& model_cache_dir) {
  int status;

  // Generate the model name if not initialized
  if (model_name_.empty()) {
    model_name_ = GenerateModelName(input_names, output_names, origin_idims);
  }
  // Load from the cached model file
  auto model_path = model_cache_dir + "/" + model_name_ + ".dla";
  VLOG(3) << "[APU] Load model from " << model_path;

  std::vector<char> compilationBuffer;
  if (!ReadFile(model_path, &compilationBuffer)) {
    LOG(WARNING) << "[NPU] Open " << model_path << " for reading failed!";
    return false;
  }
  model_ = nullptr;
  compilation_ = nullptr;
  status = NeuronModel_restoreFromCompiledNetwork(
      &model_, &compilation_, &compilationBuffer[0], compilationBuffer.size());
  if (status != NEURON_NO_ERROR) {
    LOG(WARNING) << "[APU] Load model failed!" << compilationBuffer.size();
    return false;
  }

  VLOG(3) << "[APU] Complete Load model!";

  // Deserialize the preicisions and shapes of the origin output tensors from
  // the cached configuration file
  auto config_path = model_cache_dir + "/" + model_name_ + ".cfg";
  VLOG(3) << "[APU] Load configuration from " << config_path;
  std::vector<char> config_buffer;
  if (!ReadFile(config_path, &config_buffer)) {
    LOG(WARNING) << "[APU] read from " << config_path << " failed!";
    return false;
  }

  std::string str(config_buffer.begin(), config_buffer.end());
  // Parse the precision and shapes of the output tensors
  auto output_options = Split<std::string>(str, ";");
  CHECK_EQ(output_options.size(), output_names.size());
  origin_otypes_.resize(output_names.size());
  origin_odims_.resize(output_names.size());
  for (int i = 0; i < output_names.size(); i++) {
    auto items = Split<std::string>(output_options[i], ":");
    CHECK_EQ(items.size(), 2);  // precision and shapes
    origin_otypes_[i] = static_cast<PrecisionType>(std::stoi(items[0]));
    origin_odims_[i] = Split<int64_t>(items[1], ",");
  }
  return true;
}

bool DeviceProgram::BuildGraphAndCacheToFile(
    RuntimeProgram* origin_program,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<std::vector<int64_t>>& origin_idims,
    const std::vector<Tensor*>& origin_itensors,
    const std::vector<Tensor*>& origin_otensors,
    const std::string& model_cache_dir) {
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };

  auto start_time = GetCurrentUS();

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
  graph.set_input_names(input_names);
  graph.set_output_names(output_names);

  // Convert all of ops and their input vars and weights and added into the APU
  // NIR graph
  const auto& bridges = subgraph::SubgraphBridgeRegistry::Instance();
  const auto& insts = origin_program->instructions(kRootBlockIdx);

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
  for (int i = 0; i < input_names.size(); i++) {
    CHECK(graph.Has(input_names[i])) << "[APU] Failed to find input node "
                                     << input_names[i];
    auto index = graph.Get(input_names[i])->index();
    input_indices.push_back(index);
    VLOG(3) << "[APU] Input[" << i << "] name " << input_names[i] << " dims "
            << origin_itensors[i]->dims() << " index " << index;
  }

  // Get the index of output tensors
  std::vector<uint32_t> output_indices;
  for (int i = 0; i < output_names.size(); i++) {
    CHECK(graph.Has(output_names[i])) << "[APU] Failed to find output node "
                                      << output_names[i];
    origin_otensors[i]->mutable_data<int8_t>();
    auto index = graph.Get(output_names[i])->index();
    output_indices.push_back(index);
    VLOG(3) << "[APU] Output[" << i << "] name " << output_names[i] << " dims "
            << origin_otensors[i]->dims() << " index " << index;
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

  VLOG(1) << "[APU] APU NIR model created, Create cost "
          << GetCurrentUS() - start_time << " us";

  compilation_ = lite::apu::Device::Global().Build(model_);
  if (compilation_ == nullptr) {
    LOG(WARNING) << "[APU] Build APU DLA model failed!";
    return false;
  }
  VLOG(1) << "[APU] APU DLA model created, Build cost "
          << GetCurrentUS() - start_time << " us";

  start_time = GetCurrentUS();
  CHECK_EQ(origin_otensors.size(), output_names.size());
  origin_otypes_.resize(output_names.size());
  origin_odims_.resize(output_names.size());
  for (size_t i = 0; i < output_names.size(); i++) {
    origin_otypes_[i] = origin_otensors[i]->precision();
    origin_odims_[i] = origin_otensors[i]->dims().Vectorize();
  }
  if (!model_cache_dir.empty()) {
    // Save the generated model to file
    auto model_path = model_cache_dir + "/" + model_name_ + ".dla";
    VLOG(3) << "[APU] Save model to " << model_path;

    size_t compilationSize;
    status = NeuronCompilation_getCompiledNetworkSize(compilation_,
                                                      &compilationSize);
    if (status == NEURON_NO_ERROR) {
      // Serialization DLA
      std::vector<char> model_buffer;
      model_buffer.resize(compilationSize);
      status = NeuronCompilation_storeCompiledNetwork(
          compilation_, &model_buffer[0], compilationSize);
      if (status != NEURON_NO_ERROR) {
        LOG(WARNING) << "[APU] Serialization DLA failed!";
      }

      VLOG(3) << "[APU] Export the model to " << model_path;
      if (!WriteFile(model_path, model_buffer)) {
        LOG(WARNING) << "[APU] Open " << model_path << " for writting failed!";
      }
    }

    // Serialize the precisions and shapes of the origin output tensors into the
    // configuration file
    std::ostringstream os;
    for (int i = 0; i < output_names.size(); i++) {
      os << static_cast<int32_t>(origin_otypes_[i]) << ":";
      for (auto dim : origin_odims_[i]) {
        os << dim << ",";
      }
      os << ";";
    }
    auto str = os.str();
    std::vector<char> config_buffer(str.begin(), str.end());
    auto config_path = model_cache_dir + "/" + model_name_ + ".cfg";
    VLOG(3) << "[APU] Save configuration to " << config_path;
    if (!WriteFile(config_path, config_buffer)) {
      LOG(WARNING) << "[APU] Open " << config_path << " for writting failed!";
    }
  }

  return true;
}

bool SubgraphEngine::BuildDeviceProgram() {
  // Check if the cache device program exists
  if (!device_programs_.count(origin_idims_)) {
    auto device_program = std::make_shared<DeviceProgram>();
    // Obtain the model cache dir from the NPU Context of the subgraph op
    auto model_cache_dir =
        ctx_->As<APUContext>().SubgraphModelCacheDir(exec_scope_);
    VLOG(3) << "[APU] Getting subgraph_model_cache_dir: " << model_cache_dir;
    // Check and load if the cached model and configuration file exists
    if (model_cache_dir.empty() ||
        !device_program->LoadFromCacheFile(
            input_names_, output_names_, origin_idims_, model_cache_dir)) {
      // Build the model online, including converting the paddle ops to the NIR
      // nodes, building the MTK NIR graph, and compile MTK NIR graph to dla
      if (!origin_program_) {
        BuildOriginProgram();
      }
      CHECK(origin_program_) << "[APU] The origin program is not initialized!";
      CHECK_GT(origin_program_->instructions().size(), 0)
          << "[APU] No instructions found in the origin program!";
      if (!device_program->BuildGraphAndCacheToFile(origin_program_.get(),
                                                    input_names_,
                                                    output_names_,
                                                    origin_idims_,
                                                    origin_itensors_,
                                                    origin_otensors_,
                                                    model_cache_dir)) {
        return false;
      }
    }
    if (device_program->model_ == nullptr) {
      LOG(WARNING) << "dla create fail!";
      return false;
    }
    device_programs_[origin_idims_] = device_program;
  }

  // Get the index of output tensors
  auto device_program = device_programs_[origin_idims_];
  CHECK(device_program && device_program->model_);
  for (int i = 0; i < output_names_.size(); i++) {
    origin_otensors_[i]->Resize(device_program->origin_odims_[i]);
    origin_otensors_[i]->mutable_data<int8_t>();
    VLOG(3) << "[APU] Output[" << i << "] name " << output_names_[i] << " dims "
            << origin_otensors_[i]->dims() << " memory_size "
            << origin_otensors_[i]->memory_size();
  }
  return true;
}

bool SubgraphEngine::LaunchDeviceProgram() {
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };

  if (device_programs_.count(origin_idims_) == 0 ||
      device_programs_[origin_idims_]->model_ == nullptr) {
    return LaunchOriginProgram();
  }

  auto device_program = device_programs_[origin_idims_];

  auto start_time = GetCurrentUS();
  NeuronExecution* run = NULL;
  int neuron_errCode =
      NeuronExecution_create(device_program->compilation_, &run);
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
  VLOG(1) << "[APU] Process cost " << GetCurrentUS() - start_time << " us";
  return true;
}

SubgraphEngine::~SubgraphEngine() {
  for (auto& device_program : device_programs_) {
    if (device_program.second->compilation_) {
      NeuronCompilation_free(device_program.second->compilation_);
    }
    if (device_program.second->model_) {
      NeuronModel_free(device_program.second->model_);
    }
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
