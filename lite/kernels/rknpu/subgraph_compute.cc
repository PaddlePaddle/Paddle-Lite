// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/rknpu/subgraph_compute.h"
#include <sys/time.h>
#include <time.h>
#include <utility>
#include "lite/backends/rknpu/device.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/rknpu/bridges/graph.h"
#include "lite/kernels/rknpu/bridges/paddle_use_bridges.h"
#include "lite/kernels/rknpu/bridges/utility.h"
#include "lite/utils/io.h"
#include "lite/utils/md5.h"
#include "rknpu/rknpu_pub.h"  // NOLINT

namespace paddle {
namespace lite {
namespace kernels {
namespace rknpu {

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

// Deserialize the generated model, the precisions and dimensions of the origin
// output tensors of the subgraph op from the cached configuration file and
// binary RK IR graph file
bool DeviceProgram::LoadCacheFromBufferAndFile(
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<std::vector<int64_t>>& origin_idims,
    const std::vector<Tensor*>& origin_itensors,
    const std::vector<Tensor*>& origin_otensors,
    std::vector<char>* model_cache_cfg_buffer,
    std::vector<char>* model_cache_bin_buffer,
    const std::string& model_cache_dir) {
  CHECK(!model_name_.empty());
  // Deserialize the preicisions, shapes and scales of the origin input/output
  // tensors from the cached configuration file
  if (!model_cache_cfg_buffer->empty()) {
    VLOG(3) << "[Rockchip NPU] Load configuration from buffer";
  } else if (!model_cache_dir.empty()) {
    auto config_path = model_cache_dir + "/" + model_name_ + ".cfg";
    VLOG(3) << "[Rockchip NPU] Load configuration from " << config_path;
    if (!ReadFile(config_path, model_cache_cfg_buffer)) {
      LOG(WARNING) << "[Rockchip NPU] read from " << config_path << " failed!";
      return false;
    }
  } else {
    return false;
  }
  std::string str(model_cache_cfg_buffer->begin(),
                  model_cache_cfg_buffer->end());
  // Parse the precision and shapes of the output tensors
  std::vector<std::shared_ptr<rk::nn::Tensor>> device_inodes;
  auto inputs_outputs = Split<std::string>(str, "\n");
  CHECK_EQ(inputs_outputs.size(), 2);  // inputs and outputs
  // Create a new RK IR graph and restore from the cached binary file
  graph_ = std::make_shared<rk::nn::Graph>();
  if (!model_cache_bin_buffer->empty()) {
    VLOG(3) << "[Rockchip NPU] Load model from buffer";
    if (graph_->LoadCache(model_cache_bin_buffer->data(),
                          model_cache_bin_buffer->size()) !=
        rk::nn::RK_SUCCESS) {
      LOG(WARNING)
          << "[Rockchip NPU] Load cached binary graph from buffer failed!";
      return false;
    }
  } else if (!model_cache_dir.empty()) {
    auto model_path = model_cache_dir + "/" + model_name_ + ".bin";
    VLOG(3) << "[Rockchip NPU] Load model from " << model_path;
    if (graph_->LoadCache(model_path) != rk::nn::RK_SUCCESS) {
      LOG(WARNING) << "[Rockchip NPU] Load cached binary graph from "
                   << model_path << " failed!";
      return false;
    }
  } else {
    return false;
  }
  // Restore the input RK IR nodes
  auto input_options = Split<std::string>(inputs_outputs[0], ";");
  CHECK_EQ(input_options.size(), input_names.size());
  for (int i = 0; i < input_names.size(); i++) {
    auto items = Split<std::string>(input_options[i], ":");
    CHECK_EQ(items.size(), 1);  // only scales
    const auto& scales = Split<float>(items[0], ",");
    device_inodes.push_back(
        subgraph::rknpu::CvtTensor(graph_.get(),
                                   input_names[i],
                                   origin_itensors[i]->dims().Vectorize(),
                                   scales,
                                   nullptr,
                                   origin_itensors[i]->precision()));
  }
  // Restore the output RK IR nodes
  std::vector<std::shared_ptr<rk::nn::Tensor>> device_onodes;
  auto output_options = Split<std::string>(inputs_outputs[1], ";");
  CHECK_EQ(output_options.size(), output_names.size());
  origin_otypes_.resize(output_names.size());
  origin_odims_.resize(output_names.size());
  for (int i = 0; i < output_names.size(); i++) {
    auto items = Split<std::string>(output_options[i], ":");
    CHECK_EQ(items.size(), 3);  // precision, shapes and scales
    origin_otypes_[i] = static_cast<PrecisionType>(std::stoi(items[0]));
    origin_odims_[i] = Split<int64_t>(items[1], ",");
    const auto& scales = Split<float>(items[2], ",");
    device_onodes.push_back(subgraph::rknpu::CvtTensor(graph_.get(),
                                                       output_names[i],
                                                       origin_odims_[i],
                                                       scales,
                                                       nullptr,
                                                       origin_otypes_[i]));
  }
  // Create the RK execution for inference, and set the input and output nodes
  execution_ = lite::rknpu::Device::Global().Build(
      model_name_, graph_.get(), device_inodes, device_onodes);
  if (execution_ == nullptr) {
    LOG(WARNING) << "[Rockchip NPU] Build model failed!";
    return false;
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
  CHECK(!model_name_.empty());
  // Create a new RK IR graph
  graph_ = std::make_shared<rk::nn::Graph>();
  if (!model_cache_dir.empty()) {
    // Enable caching the compiled RK IR graph to a binary file when the first
    // run
    auto model_path = model_cache_dir + "/" + model_name_ + ".bin";
    if (graph_->EnableCreateCache(model_path) == rk::nn::RK_SUCCESS) {
      VLOG(3) << "[Rockchip NPU] The compiled RK IR graph will be saved to "
              << model_path << " when the first run";
    } else {
      LOG(WARNING)
          << "[Rockchip NPU] Failed to cache the compiled RK IR graph to "
          << model_path;
    }
  }
  // Convert all of Paddle operators and variables to the RK IR nodes, and add
  // them into the RK IR graph
  int status = 0;
  subgraph::rknpu::Graph graph(graph_.get());
  const auto& bridges = subgraph::SubgraphBridgeRegistry::Instance();
  CHECK(origin_program)
      << "[Rockchip NPU] The origin program is not initialized!";
  CHECK_GT(origin_program->instructions(kRootBlockIdx).size(), 0)
      << "[Rockchip NPU] No instructions found in the origin program!";
  const auto& insts = origin_program->instructions(kRootBlockIdx);
  for (auto& inst : insts) {
    auto op = const_cast<OpLite*>(inst.op());
    CHECK(op);
    op->CheckShape();
    op->InferShape();
    std::string op_type = op->op_info()->Type();
    if (!bridges.Exists(op_type, TARGET(kRKNPU))) {
      return false;
    }
    auto kernel = inst.kernel();
    status |= bridges.Select(op_type, TARGET(kRKNPU))(
        reinterpret_cast<void*>(&graph), op, const_cast<KernelBase*>(kernel));
    if (subgraph::CHECK_FAILED(status)) {
      return false;
    }
  }
  // Collect the input and output nodes from the RK IR graph
  std::vector<std::shared_ptr<rk::nn::Tensor>> device_inodes;
  for (size_t i = 0; i < input_names.size(); i++) {
    CHECK(graph.Has(input_names[i]));
    CHECK(graph.Get(input_names[i])->is_data());
    device_inodes.push_back(graph.Get(input_names[i])->data());
  }
  std::vector<std::shared_ptr<rk::nn::Tensor>> device_onodes;
  for (size_t i = 0; i < output_names.size(); i++) {
    CHECK(graph.Has(output_names[i]));
    device_onodes.push_back(graph.Get(output_names[i])->data());
  }
  // Create the RK execution for inference, and set the input and output nodes
  execution_ = lite::rknpu::Device::Global().Build(
      model_name_, graph_.get(), device_inodes, device_onodes);
  if (execution_ == nullptr) {
    LOG(WARNING) << "[Rockchip NPU] Build model failed!";
    return false;
  }
  // Update the precison and dimensions of the origin output tensors
  CHECK_EQ(origin_otensors.size(), output_names.size());
  origin_otypes_.resize(output_names.size());
  origin_odims_.resize(output_names.size());
  for (size_t i = 0; i < output_names.size(); i++) {
    origin_otypes_[i] = graph.Get(output_names[i])->precision();
    origin_odims_[i] = origin_otensors[i]->dims().Vectorize();
  }
  if (!model_cache_dir.empty()) {
    // Serialize the precisions, shapes and scales of the origin input/output
    // tensors
    // into the configuration file
    std::ostringstream os;
    for (int i = 0; i < input_names.size(); i++) {
      const auto& scales =
          device_inodes[i]->GetAttrs()->qntParamSymmetric.scale;
      for (const auto& scale : scales) {
        os << scale << ",";
      }
      os << ";";
    }
    os << "\n";
    for (int i = 0; i < output_names.size(); i++) {
      os << static_cast<int32_t>(origin_otypes_[i]) << ":";
      for (auto dim : origin_odims_[i]) {
        os << dim << ",";
      }
      os << ":";
      const auto& scales =
          device_onodes[i]->GetAttrs()->qntParamSymmetric.scale;
      for (const auto& scale : scales) {
        os << scale << ",";
      }
      os << ";";
    }
    auto str = os.str();
    std::vector<char> config_buffer(str.begin(), str.end());
    auto config_path = model_cache_dir + "/" + model_name_ + ".cfg";
    VLOG(3) << "[Rockchip NPU] Save configuration to " << config_path;
    if (!WriteFile(config_path, config_buffer)) {
      LOG(WARNING) << "[Rockchip NPU] Open " << config_path
                   << " for writting failed!";
    }
  }
  return true;
}

bool DeviceProgram::PrepareInputsOutputs(
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    std::vector<Tensor*>* origin_itensors,
    std::vector<Tensor*>* origin_otensors) {
  CHECK(!model_name_.empty() && graph_ && execution_);
  // Check the dimensions of the device tensors and the origin tensors
  CHECK_EQ(origin_itensors->size(), input_names.size());
  CHECK_EQ(origin_otensors->size(), output_names.size());
  CHECK_EQ(origin_otypes_.size(), output_names.size());
  CHECK_EQ(origin_odims_.size(), output_names.size());
  device_itensors_.resize(input_names.size());
  device_otensors_.resize(output_names.size());
  for (size_t i = 0; i < input_names.size(); i++) {
    VLOG(3) << "[Rockchip NPU] Inputs[" << i << "] name: " << input_names[i]
            << " dims:" << origin_itensors->at(i)->dims().repr() << " ";
    device_itensors_[i].index = i;
    device_itensors_[i].buf =
        reinterpret_cast<void*>(origin_itensors->at(i)->raw_data());
    device_itensors_[i].size = origin_itensors->at(i)->memory_size();
    device_itensors_[i].pass_through = false;
    device_itensors_[i].type =
        subgraph::rknpu::CvtPrecisionType(origin_itensors->at(i)->precision());
    device_itensors_[i].layout = rk::nn::DataLayoutType::NCHW;
  }
  for (size_t i = 0; i < output_names.size(); i++) {
    origin_otensors->at(i)->Resize(origin_odims_[i]);
    VLOG(3) << "[Rockchip NPU] Outputs[" << i << "] name: " << output_names[i]
            << " dims:" << origin_otensors->at(i)->dims().repr();
    switch (origin_otypes_[i]) {
      case PRECISION(kInt8):
        origin_otensors->at(i)->mutable_data<int8_t>();
        break;
      case PRECISION(kInt32):
        origin_otensors->at(i)->mutable_data<int32_t>();
        break;
      default:
        LOG(FATAL)
            << "[Rockchip NPU] Unable to mutable data with precision type "
            << PrecisionToStr(origin_otypes_[i]);
        break;
    }
    device_otensors_[i].index = i;
    device_otensors_[i].buf =
        reinterpret_cast<void*>(origin_otensors->at(i)->raw_data());
    device_otensors_[i].size = origin_otensors->at(i)->memory_size();
    device_otensors_[i].want_float = false;
    device_otensors_[i].type =
        subgraph::rknpu::CvtPrecisionType(origin_otensors->at(i)->precision());
    device_otensors_[i].layout = rk::nn::DataLayoutType::NCHW;
  }
  return true;
}

bool DeviceProgram::StartExecution() {
  CHECK(!model_name_.empty() && graph_ && execution_);
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };
  int istamp;
  auto start_time = GetCurrentUS();
  CHECK(execution_->SetInputs(device_itensors_) == rk::nn::RK_SUCCESS);
  CHECK(execution_->Run() == rk::nn::RK_SUCCESS);
  CHECK(execution_->GetOutputs(device_otensors_) == rk::nn::RK_SUCCESS);
  VLOG(3) << "[Rockchip NPU] Process cost " << GetCurrentUS() - start_time
          << " us";
  return true;
}

bool SubgraphEngine::BuildDeviceProgram() {
  // Check if the cache device program exists
  if (!device_programs_.count(origin_idims_)) {
    auto device_program = std::make_shared<DeviceProgram>();
    // Generate the model name by the names and dimensions of the input and
    // output tensors
    device_program->model_name_ = DeviceProgram::GenerateModelName(
        input_names_, output_names_, origin_idims_);
    // Load the cached configuration and model from the buffers which are stored
    // as the tensors in the exec scope
    std::vector<char> model_cache_cfg_buffer;
    std::vector<char> model_cache_bin_buffer;
    ctx_->As<RKNPUContext>().SubgraphModelCacheBuffers(
        exec_scope_,
        device_program->model_name_,
        &model_cache_cfg_buffer,
        &model_cache_bin_buffer);
    VLOG(3) << "[Rockchip NPU] Getting subgraph_model_cache_cfg_buffer: "
            << model_cache_cfg_buffer.size()
            << ", subgraph_model_cache_bin_buffer: "
            << model_cache_bin_buffer.size();
    // Obtain the model cache dir from the Rockchip NPU Context of the subgraph
    // op
    auto model_cache_dir =
        ctx_->As<RKNPUContext>().SubgraphModelCacheDir(exec_scope_);
    VLOG(3) << "[Rockchip NPU] Getting subgraph_model_cache_dir: "
            << model_cache_dir;
    // Check and load if the cached model and configuration file exists
    if (!device_program->LoadCacheFromBufferAndFile(input_names_,
                                                    output_names_,
                                                    origin_idims_,
                                                    origin_itensors_,
                                                    origin_otensors_,
                                                    &model_cache_cfg_buffer,
                                                    &model_cache_bin_buffer,
                                                    model_cache_dir)) {
      // Build the model online, including converting the paddle ops to the RK
      // IR nodes, building the RK IR graph, and generate a execution for
      // inference.
      if (!origin_program_) {
        BuildOriginProgram();
      }
      CHECK(origin_program_)
          << "[Rockchip NPU] The origin program is not initialized!";
      CHECK_GT(origin_program_->instructions().size(), 0)
          << "[Rockchip NPU] No instructions found in the origin program!";
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
    if (!device_program->graph_ || !device_program->execution_) {
      return false;
    }
    device_programs_[origin_idims_] = device_program;
  }
  auto device_program = device_programs_[origin_idims_];
  CHECK(device_program && device_program->graph_ && device_program->execution_);
  return device_program->PrepareInputsOutputs(
      input_names_, output_names_, &origin_itensors_, &origin_otensors_);
}

bool SubgraphEngine::LaunchDeviceProgram() {
  // Roll back to launch the origin program if the device program can't be
  // found or graph/execution isn't initialized.
  if (!device_programs_.count(origin_idims_)) {
    return LaunchOriginProgram();
  }
  auto device_program = device_programs_[origin_idims_];
  if (!device_program->graph_ || !device_program->execution_) {
    return LaunchOriginProgram();
  }
  return device_program->StartExecution();
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

}  // namespace rknpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(subgraph,
                     kRKNPU,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::rknpu::SubgraphCompute,
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
