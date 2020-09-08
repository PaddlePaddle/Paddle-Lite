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

#include "lite/kernels/npu/subgraph_compute.h"
#include <sys/time.h>
#include <time.h>
#include <algorithm>
#include <functional>
#include <utility>
#include "hiai_ir_build.h"  // NOLINT
#include "lite/backends/npu/device.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/paddle_use_bridges.h"
#include "lite/kernels/npu/bridges/utility.h"
#include "lite/utils/io.h"
#include "lite/utils/md5.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace npu {

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
// output tensors of the subgraph op from the cached configuration file and HiAI
// om file
bool DeviceProgram::LoadFromCacheFile(
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<std::vector<int64_t>>& origin_idims,
    const std::string& model_cache_dir) {
  // Generate the model name if not initialized
  if (model_name_.empty()) {
    model_name_ = GenerateModelName(input_names, output_names, origin_idims);
  }
  // Load from the cached model file, return a HiAI model manager client for
  // inference
  auto model_path = model_cache_dir + "/" + model_name_ + ".om";
  VLOG(3) << "[NPU] Load model from " << model_path;
  std::vector<char> model_buffer;
  if (!ReadFile(model_path, &model_buffer)) {
    LOG(WARNING) << "[NPU] Open " << model_path << " for reading failed!";
    return false;
  }
  bool model_comp = false;
  model_client_ =
      lite::npu::Device::Global().Load(model_name_, &model_buffer, &model_comp);
  if (!model_client_) {
    LOG(WARNING) << "[NPU] Load model failed!";
    return false;
  }
  // Rewrite with the compatible model data if the cached
  // model file is incompatible with the current device
  if (!model_comp) {
    VLOG(3) << "[NPU] Export the compatible model to " << model_path;
    if (!WriteFile(model_path, model_buffer)) {
      LOG(WARNING) << "[NPU] Open " << model_path << " for writting failed!";
    }
  }
  // Deserialize the precisions and shapes of the origin output tensors from the
  // cached configuration file
  auto config_path = model_cache_dir + "/" + model_name_ + ".cfg";
  VLOG(3) << "[NPU] Load configuration from " << config_path;
  std::vector<char> config_buffer;
  if (!ReadFile(config_path, &config_buffer)) {
    LOG(WARNING) << "[NPU] read from " << config_path << " failed!";
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
    const std::vector<Tensor*>& origin_otensors,
    const std::string& model_cache_dir) {
  // Generate the model name if not initialized
  if (model_name_.empty()) {
    model_name_ = GenerateModelName(input_names, output_names, origin_idims);
  }
  // Convert all of ops and their input vars and weights to HiAI IR nodes,
  // then added them into the HiAI IR graph
  int status = 0;
  subgraph::npu::Graph graph;
  const auto& bridges = subgraph::SubgraphBridgeRegistry::Instance();
  CHECK(origin_program) << "[NPU] The origin program is not initialized!";
  CHECK_GT(origin_program->instructions(kRootBlockIdx).size(), 0)
      << "[NPU] No instructions found in the origin program!";
  const auto& insts = origin_program->instructions(kRootBlockIdx);
  for (auto& inst : insts) {
    auto op = const_cast<OpLite*>(inst.op());
    CHECK(op);
    op->CheckShape();
    op->InferShape();
    std::string op_type = op->op_info()->Type();
    if (!bridges.Exists(op_type, TARGET(kNPU))) {
      return false;
    }
    auto kernel = inst.kernel();
    status |= bridges.Select(op_type, TARGET(kNPU))(
        reinterpret_cast<void*>(&graph), op, const_cast<KernelBase*>(kernel));
    if (subgraph::CHECK_FAILED(status)) {
      return false;
    }
  }
  // Collect the input and output nodes of the HiAI IR graph
  std::vector<ge::Operator> device_inodes;
  for (size_t i = 0; i < input_names.size(); i++) {
    CHECK(graph.Has(input_names[i]));
    CHECK(graph.Get(input_names[i])->is_data());
    device_inodes.push_back(*graph.Get(input_names[i])->data());
  }
  std::vector<ge::Operator> device_onodes;
  for (size_t i = 0; i < output_names.size(); i++) {
    CHECK(graph.Has(output_names[i]));
    device_onodes.push_back(*graph.Get(output_names[i])->data());
  }
  // Build the HiAI IR graph to the HiAI om model
  std::vector<char> model_buffer;
  if (!lite::npu::Device::Global().Build(
          device_inodes, device_onodes, &model_buffer)) {
    LOG(WARNING) << "[NPU] Build model failed!";
    return false;
  }
  // Load the HiAI om model and create a HiAI model manager client(from HiAI
  // Service) to run inference.
  bool model_comp = true;
  model_client_ =
      lite::npu::Device::Global().Load(model_name_, &model_buffer, &model_comp);
  if (!model_client_) {
    LOG(WARNING) << "[NPU] Load model failed!";
    return false;
  }
  // Do not check model compatibility because it assume that the cached om model
  // is always compatible with the current device
  // Update the precison and dimensions of the origin output tensors
  // Update the precison and dimensions of the origin output tensors
  CHECK_EQ(origin_otensors.size(), output_names.size());
  origin_otypes_.resize(output_names.size());
  origin_odims_.resize(output_names.size());
  for (size_t i = 0; i < output_names.size(); i++) {
    origin_otypes_[i] = graph.Get(output_names[i])->precision();
    origin_odims_[i] = origin_otensors[i]->dims().Vectorize();
  }
  if (!model_cache_dir.empty()) {
    // Save the generated model to file, used for the model caching or the
    // offline model generation
    auto model_path = model_cache_dir + "/" + model_name_ + ".om";
    VLOG(3) << "[NPU] Save model to " << model_path;
    if (!WriteFile(model_path, model_buffer)) {
      LOG(WARNING) << "[NPU] Open " << model_path << " for writting failed!";
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
    VLOG(3) << "[NPU] Save configuration to " << config_path;
    if (!WriteFile(config_path, config_buffer)) {
      LOG(WARNING) << "[NPU] Open " << config_path << " for writting failed!";
    }
  }
  return true;
}

bool DeviceProgram::ShareBufferWithOriginTensors(
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    std::vector<Tensor*>* origin_itensors,
    std::vector<Tensor*>* origin_otensors,
    std::vector<std::shared_ptr<hiai::AiTensor>>* device_itensors,
    std::vector<std::shared_ptr<hiai::AiTensor>>* device_otensors) {
  CHECK(!model_name_.empty() && model_client_);
  // Query the dimensions of the device input and output tensors if not
  // initialized
  if (device_idims_.empty() || device_odims_.empty()) {
    if (model_client_->GetModelIOTensorDim(
            model_name_, device_idims_, device_odims_) != hiai::AI_SUCCESS) {
      LOG(WARNING)
          << "[NPU] Get the dimensions of input and output tensors failed!";
      return false;
    }
  }
  // Check the dimensions of the device tensors and the origin tensors
  CHECK_EQ(device_itensors->size(), input_names.size());
  CHECK_EQ(device_otensors->size(), output_names.size());
  CHECK_EQ(origin_otypes_.size(), output_names.size());
  CHECK_EQ(origin_odims_.size(), output_names.size());
  CHECK_EQ(device_idims_.size(), input_names.size());
  CHECK_EQ(device_odims_.size(), output_names.size());
  for (int i = 0; i < input_names.size(); i++) {
    VLOG(3) << "[NPU] Inputs[" << i << "] name: " << input_names[i]
            << " origin dims:" << (*origin_itensors)[i]->dims().repr()
            << " device dims: {" << device_idims_[i].GetNumber() << ","
            << device_idims_[i].GetChannel() << ","
            << device_idims_[i].GetHeight() << ","
            << device_idims_[i].GetWidth() << "}";
    CHECK_EQ((*origin_itensors)[i]->dims().production(),
             device_idims_[i].GetNumber() * device_idims_[i].GetChannel() *
                 device_idims_[i].GetHeight() * device_idims_[i].GetWidth());
    VLOG(3) << "[NPU] Init the input tensors for the device program and share "
               "their buffers with the origin input tensors";
    // Reinit device tensor will free shared buffer, so copy data to a tmp
    // tensor
    Tensor tmp;
    tmp.CopyDataFrom(*(*origin_itensors)[i]);
    (*device_itensors)[i]->Init(&(device_idims_[i]));

    std::memcpy(
        (*device_itensors)[i]->GetBuffer(), tmp.raw_data(), tmp.memory_size());

    // Share data buf between device_itensor and origin_itensor
    std::shared_ptr<Buffer> buffer =
        std::make_shared<Buffer>((*device_itensors)[i]->GetBuffer(),
                                 lite_api::TargetType::kHost,
                                 (*device_itensors)[i]->GetSize());
    (*origin_itensors)[i]->ResetBuffer(buffer,
                                       (*device_itensors)[i]->GetSize());
  }
  for (int i = 0; i < output_names.size(); i++) {
    (*origin_otensors)[i]->set_precision(origin_otypes_[i]);
    (*origin_otensors)[i]->Resize(origin_odims_[i]);
    VLOG(3) << "[NPU] Outputs[" << i << "] name: " << output_names[i]
            << " origin dims:" << (*origin_otensors)[i]->dims().repr()
            << " device dims: {" << device_odims_[i].GetNumber() << ","
            << device_odims_[i].GetChannel() << ","
            << device_odims_[i].GetHeight() << ","
            << device_odims_[i].GetWidth() << "}";
    CHECK_EQ((*origin_otensors)[i]->dims().production(),
             device_odims_[i].GetNumber() * device_odims_[i].GetChannel() *
                 device_odims_[i].GetHeight() * device_odims_[i].GetWidth());
    (*device_otensors)[i]->Init(&(device_odims_[i]));
    VLOG(3) << "[NPU] Init the output tensors for the device program and share "
               "their buffers with the origin output tensors";
    // Share data buf between device_itensor and origin_itensor
    std::shared_ptr<Buffer> buffer =
        std::make_shared<Buffer>((*device_otensors)[i]->GetBuffer(),
                                 lite_api::TargetType::kHost,
                                 (*device_otensors)[i]->GetSize());
    (*origin_otensors)[i]->ResetBuffer(buffer,
                                       (*device_otensors)[i]->GetSize());
  }
  return true;
}

bool DeviceProgram::ZeroCopyRun(
    std::vector<std::shared_ptr<hiai::AiTensor>>* device_itensors,
    std::vector<std::shared_ptr<hiai::AiTensor>>* device_otensors) {
  CHECK(!model_name_.empty() && model_client_);
  // Run the HiAI model by name
  std::string key = "model_name";  // Note: key seems must be model_name
  hiai::AiContext model_context;
  model_context.AddPara(key, model_name_);
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };
  int istamp;
  auto start_time = GetCurrentUS();
  CHECK_EQ(model_client_->Process(
               model_context, *device_itensors, *device_otensors, 1000, istamp),
           hiai::AI_SUCCESS);
  VLOG(3) << "[NPU] Process cost " << GetCurrentUS() - start_time << " us";
  return true;
}

bool SubgraphEngine::PrepareWorkspaceForDeviceProgram() {
  // Obtain the origin input tensors, and create the origin output
  // tensors(Don't try to access them before launch the device program or the
  // origin program)
  PrepareWorkspaceForOriginProgram();
  // Create the device input and output tensors, but don't initialize them
  // with the dimensions
  device_itensors_.resize(input_names_.size());
  for (int i = 0; i < input_names_.size(); i++) {
    device_itensors_[i].reset(new hiai::AiTensor);
    CHECK(device_itensors_[i]);
  }
  device_otensors_.resize(output_names_.size());
  for (int i = 0; i < output_names_.size(); i++) {
    device_otensors_[i].reset(new hiai::AiTensor);
    CHECK(device_otensors_[i]);
  }
  return true;
}

bool SubgraphEngine::BuildDeviceProgram() {
  // Check if the cache device program exists
  if (!device_programs_.count(origin_idims_)) {
    auto device_program = std::make_shared<DeviceProgram>();
    // Obtain the model cache dir from the NPU Context of the subgraph op
    auto model_cache_dir =
        ctx_->As<NPUContext>().SubgraphModelCacheDir(exec_scope_);
    VLOG(3) << "[NPU] Getting subgraph_model_cache_dir: " << model_cache_dir;
    // Check and load if the cached model and configuration file exists
    if (model_cache_dir.empty() ||
        !device_program->LoadFromCacheFile(
            input_names_, output_names_, origin_idims_, model_cache_dir)) {
      // Build the model online, including converting the paddle ops to the HiAI
      // IR nodes, building the HiAI IR graph to the om model, then load it as a
      // new HiAI model manager client for inference.
      if (!origin_program_) {
        BuildOriginProgram();
      }
      CHECK(origin_program_) << "[NPU] The origin program is not initialized!";
      CHECK_GT(origin_program_->instructions().size(), 0)
          << "[NPU] No instructions found in the origin program!";
      if (!device_program->BuildGraphAndCacheToFile(origin_program_.get(),
                                                    input_names_,
                                                    output_names_,
                                                    origin_idims_,
                                                    origin_otensors_,
                                                    model_cache_dir)) {
        return false;
      }
    }
    if (device_program->model_client_ == nullptr) {
      return false;
    }
    device_programs_[origin_idims_] = device_program;
  }
  auto device_program = device_programs_[origin_idims_];
  CHECK(device_program && device_program->model_client_);
  return device_program->ShareBufferWithOriginTensors(input_names_,
                                                      output_names_,
                                                      &origin_itensors_,
                                                      &origin_otensors_,
                                                      &device_itensors_,
                                                      &device_otensors_);
}

bool SubgraphEngine::LaunchDeviceProgram() {
  // Roll back to launch the origin program if the device program can't be
  // found or the model client isn't initialized.
  if (device_programs_.count(origin_idims_) == 0 ||
      device_programs_[origin_idims_]->model_client_ == nullptr) {
    return LaunchOriginProgram();
  }
  auto device_program = device_programs_[origin_idims_];
  if (!device_program->model_client_) {
    return LaunchOriginProgram();
  }
  return device_program->ZeroCopyRun(&device_itensors_, &device_otensors_);
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

}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(subgraph,
                     kNPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::npu::SubgraphCompute,
                     def)
    .BindInput("Inputs",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Outputs",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .Finalize();
