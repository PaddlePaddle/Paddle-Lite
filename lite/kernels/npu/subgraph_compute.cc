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

namespace paddle {
namespace lite {
namespace kernels {
namespace npu {

void SubgraphEngine::ParseCachedDims(
    const std::vector<std::string>& cached_dims) {
  for (auto& i : cached_dims) {
    auto cached_io_dims = Split<std::string>(i, " ");
    CHECK(cached_io_dims.size() >= 1 && cached_io_dims.size() <= 2);
    // Parsing the precision and dimensions of the output tensors
    std::vector<PrecisionType> origin_otypes;
    std::vector<std::vector<int64_t>> origin_odims;
    auto cached_odims = Split<std::string>(cached_io_dims[0], ";");
    for (auto& j : cached_odims) {
      auto k = Split<std::string>(j, ":");
      CHECK_EQ(k.size(), 2);
      origin_otypes.push_back(static_cast<PrecisionType>(std::stoi(k[0])));
      origin_odims.push_back(Split<int64_t>(k[1], ","));
    }
    // Parsing the dimensions of the input tensors
    std::vector<std::vector<int64_t>> origin_idims;
    if (cached_io_dims.size() > 1) {
      auto cached_idims = Split<std::string>(cached_io_dims[1], ";");
      for (auto& j : cached_idims) {
        auto k = Split<std::string>(j, ":");
        CHECK_EQ(k.size(), 2);
        origin_idims.push_back(Split<int64_t>(k[1], ","));
      }
    }
    // Added as a cached device program, build the model from the
    // cached om model file after a cache hit ocuurs
    if (!device_programs_.count(origin_idims)) {
      device_programs_[origin_idims] =
          std::make_shared<device_program_t>(origin_odims, origin_otypes);
    }
  }
}

void SubgraphEngine::UpdateCachedDims() {
  // Pack the origin i/o dims of all of the cached device programs into a string
  std::vector<std::string> cached_data_dims;
  for (auto& device_program : device_programs_) {
    std::ostringstream os;
    // Pack the precisions and the dimensions of the origin output tensors
    auto& origin_odims = device_program.second->origin_odims_;
    auto& origin_otypes = device_program.second->origin_otypes_;
    CHECK_EQ(origin_odims.size(), origin_otypes.size());
    for (int i = 0; i < origin_odims.size(); i++) {
      os << static_cast<int32_t>(origin_otypes[i]) << ":";
      for (auto dim : origin_odims[i]) {
        os << dim << ",";
      }
      os << ";";
    }
    os << " ";
    // Pack the dimensions of the origin input tensors
    auto& origin_idims = device_program.first;
    for (int i = 0; i < origin_idims.size(); i++) {
      os << static_cast<int32_t>(PRECISION(kUnk))
         << ":";  // Reserved field for the precision type
      for (auto dim : origin_idims[i]) {
        os << dim << ",";
      }
      os << ";";
    }
    cached_data_dims.push_back(os.str());
  }
  // Find the current subgraph op and update its op attribute
  // 'cached_data_shapes'.
  auto block_size = program_desc_->BlocksSize();
  auto* block_desc = program_desc_->GetBlock<cpp::BlockDesc>(block_idx_);
  auto parent_block_idx = block_desc->ParentIdx();
  CHECK(parent_block_idx >= 0 && parent_block_idx < block_size);
  auto* parent_block_desc =
      program_desc_->GetBlock<cpp::BlockDesc>(parent_block_idx);
  auto parent_op_size = parent_block_desc->OpsSize();
  for (int parent_op_idx = 0; parent_op_idx < parent_op_size; parent_op_idx++) {
    auto* parent_op_desc = parent_block_desc->GetOp<cpp::OpDesc>(parent_op_idx);
    CHECK(parent_op_desc);
    auto parent_op_type = parent_op_desc->Type();
    if (parent_op_type != "subgraph") continue;
    if (parent_op_desc->GetAttr<int32_t>("sub_block") != block_idx_) continue;
    parent_op_desc->SetAttr<std::vector<std::string>>("cached_data_dims",
                                                      cached_data_dims);
  }
}  // namespace npu

SubgraphEngine::SubgraphEngine(KernelContext* ctx,
                               int block_idx,
                               cpp::ProgramDesc* program_desc,
                               Scope* exec_scope,
                               const std::vector<std::string>& input_names,
                               const std::vector<std::string>& output_names,
                               const std::vector<std::string>& cached_dims)
    : subgraph::Engine(
          ctx, block_idx, program_desc, exec_scope, input_names, output_names) {
  ParseCachedDims(cached_dims);
}

int SubgraphEngine::BuildDeviceProgram() {
  int status = 0;
  // Obtain the model cache dir from the NPU Context of the subgraph op
  auto model_cache_dir = ctx_->As<NPUContext>().SubgraphModelCacheDir();
  // Check if the cache device program exists
  if (!device_programs_.count(origin_idims_)) {
    device_programs_[origin_idims_] = std::make_shared<device_program_t>();
  }
  std::shared_ptr<device_program_t> device_program =
      device_programs_[origin_idims_];
  if (device_program->model_name_.empty()) {
    device_program->model_name_ = "model_" + std::to_string(block_idx_) + ".om";
  }
  // Check if the cache om model file exists
  if (!device_program->model_client_ && !model_cache_dir.empty()) {
    device_program->model_client_ = lite::npu::Device::Global().Build(
        device_program->model_name_, model_cache_dir);
  }
  // Build the device program online, including converting the paddle ops to
  // HiAI IR node, building HiAI IR graph to the om model, then load it as a
  // new model client.
  if (!device_program->model_client_) {
    // Build the origin program to attach all of ops of the subblock
    if (!origin_program_) {
      BuildOriginProgram();
    }
    CHECK(origin_program_);
    // Convert all of ops and their input vars and weights to HiAI IR nodes,
    // then added them into the HiAI IR graph
    subgraph::npu::Graph graph;
    const auto& bridges = subgraph::Registry::Instance();
    for (auto& inst : origin_program_->instructions()) {
      auto op = const_cast<OpLite*>(inst.op());
      CHECK(op);
      op->CheckShape();
      op->InferShape();
      std::string op_type = op->op_info()->Type();
      if (!bridges.Exists(op_type, TARGET(kNPU))) {
        return subgraph::FAILED;
      }
      auto kernel = inst.kernel();
      status |= bridges.Select(op_type, TARGET(kNPU))(
          reinterpret_cast<void*>(&graph), op, const_cast<KernelBase*>(kernel));
      if (subgraph::CHECK_FAILED(status)) {
        return subgraph::FAILED;
      }
    }
    // Collect the input and output nodes in the HiAI IR graph
    std::vector<ge::Operator> device_inodes;
    for (size_t i = 0; i < input_names_.size(); i++) {
      CHECK(graph.Has(input_names_[i]) &&
            graph.Get(input_names_[i])->is_data());
      device_inodes.push_back(*graph.Get(input_names_[i])->data());
    }
    std::vector<ge::Operator> device_onodes;
    for (size_t i = 0; i < output_names_.size(); i++) {
      CHECK(graph.Has(output_names_[i]));
      device_onodes.push_back(*graph.Get(output_names_[i])->data());
    }

    // Build the HiAI IR graph to the HiAI om model, then create the model
    // client to load the om model
    device_program->model_client_ =
        lite::npu::Device::Global().Build(device_program->model_name_,
                                          device_inodes,
                                          device_onodes,
                                          model_cache_dir);
    if (!device_program->model_client_) {
      LOG(WARNING) << "[NPU] Build model failed!";
      return subgraph::FAILED;
    }
    // Update the dimensions of the origin output tensors
    if (!device_program->origin_odims_.size()) {
      device_program->origin_odims_.resize(origin_otensors_.size());
      for (size_t i = 0; i < origin_otensors_.size(); i++) {
        device_program->origin_odims_[i] =
            origin_otensors_[i]->dims().Vectorize();
      }
    }
    // Update the data type of the origin output tensors
    if (!device_program->origin_otypes_.size()) {
      device_program->origin_otypes_.resize(output_names_.size());
      for (size_t i = 0; i < output_names_.size(); i++) {
        device_program->origin_otypes_[i] =
            graph.Get(output_names_[i])->precision();
      }
    }
    // Update the attr 'cached_data_dims' of the subgraph op info if
    // the environment variable 'SUBGRAPH_DISABLE_ONLINE_MODE' is set to true
    if (GetBoolFromEnv(SUBGRAPH_DISABLE_ONLINE_MODE)) {
      UpdateCachedDims();
    }
  }
  // Query the dimensions of the device input and output tensors if they are
  // not initialized
  if (!device_program->device_idims_.size() ||
      !device_program->device_odims_.size()) {
    if (device_program->model_client_->GetModelIOTensorDim(
            device_program->model_name_,
            device_program->device_idims_,
            device_program->device_odims_) != hiai::AI_SUCCESS) {
      LOG(WARNING)
          << "[NPU] Get the dimensions of input and output tensors failed!";
      return subgraph::FAILED;
    }
  }
  // Check the dimensions of the device tensors and the origin tensors
  CHECK_EQ(device_program->origin_odims_.size(), output_names_.size());
  CHECK_EQ(device_program->origin_otypes_.size(), output_names_.size());
  CHECK_EQ(device_program->device_idims_.size(), input_names_.size());
  CHECK_EQ(device_program->device_odims_.size(), output_names_.size());
  for (int i = 0; i < input_names_.size(); i++) {
    VLOG(3) << "[NPU] Inputs[" << i << "] name: " << input_names_[i]
            << " dims: {" << device_program->device_idims_[i].GetNumber() << ","
            << device_program->device_idims_[i].GetChannel() << ","
            << device_program->device_idims_[i].GetHeight() << ","
            << device_program->device_idims_[i].GetWidth() << "}";
    CHECK_EQ(std::accumulate(origin_idims_[i].begin(),
                             origin_idims_[i].end(),
                             1,
                             std::multiplies<int64_t>()),
             device_program->device_idims_[i].GetNumber() *
                 device_program->device_idims_[i].GetChannel() *
                 device_program->device_idims_[i].GetHeight() *
                 device_program->device_idims_[i].GetWidth());
    VLOG(3) << "[NPU] Init the input tensors for the device program and share "
               "their buffers with the origin input tensors";
    device_itensors_[i]->Init(&(device_program->device_idims_[i]));
    std::memcpy(device_itensors_[i]->GetBuffer(),
                origin_itensors_[i]->raw_data(),
                origin_itensors_[i]->memory_size());
    // Share data buf between device_itensor and origin_itensor
    std::shared_ptr<Buffer> buffer =
        std::make_shared<Buffer>(device_itensors_[i]->GetBuffer(),
                                 lite_api::TargetType::kHost,
                                 device_itensors_[i]->GetSize());
    origin_itensors_[i]->ResetBuffer(buffer, device_itensors_[i]->GetSize());
  }
  for (int i = 0; i < output_names_.size(); i++) {
    VLOG(3) << "[NPU] Outputs[" << i << "] name: " << output_names_[i]
            << " dims: {" << device_program->device_odims_[i].GetNumber() << ","
            << device_program->device_odims_[i].GetChannel() << ","
            << device_program->device_odims_[i].GetHeight() << ","
            << device_program->device_odims_[i].GetWidth() << "}";
    CHECK_EQ(std::accumulate(device_program->origin_odims_[i].begin(),
                             device_program->origin_odims_[i].end(),
                             1,
                             std::multiplies<int64_t>()),
             device_program->device_odims_[i].GetNumber() *
                 device_program->device_odims_[i].GetChannel() *
                 device_program->device_odims_[i].GetHeight() *
                 device_program->device_odims_[i].GetWidth());
    device_otensors_[i]->Init(&(device_program->device_odims_[i]));
    VLOG(3) << "[NPU] Init the output tensors for the device program and share "
               "their buffers with the origin output tensors";
    // Share data buf between device_itensor and origin_itensor
    origin_otensors_[i]->set_precision(device_program->origin_otypes_[i]);
    origin_otensors_[i]->Resize(device_program->origin_odims_[i]);
    std::shared_ptr<Buffer> buffer =
        std::make_shared<Buffer>(device_otensors_[i]->GetBuffer(),
                                 lite_api::TargetType::kHost,
                                 device_otensors_[i]->GetSize());
    origin_otensors_[i]->ResetBuffer(buffer, device_otensors_[i]->GetSize());
  }
  return status;
}

int SubgraphEngine::PrepareForLaunchDeviceProgram() {
  // Obtain the origin input tensors, and create the origin output
  // tensors(Don't try to access them before launch the device program or the
  // origin program)
  PrepareForLaunchOriginProgram();
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
  return subgraph::SUCCESS;
}

int SubgraphEngine::LaunchDeviceProgram() {
  // Roll back to launch the origin program if the device program can't be
  // found or the model client isn't initialized.
  if (!device_programs_.count(origin_idims_)) {
    return LaunchOriginProgram();
  }
  auto device_program = device_programs_[origin_idims_];
  if (!device_program->model_client_) {
    return LaunchOriginProgram();
  }
  // Run the HiAI model by name
  std::string key = "model_name";  // Note: key seems must be model_name
  hiai::AiContext model_context;
  model_context.AddPara(key, device_program->model_name_);
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };
  int istamp;
  auto start_time = GetCurrentUS();
  CHECK_EQ(device_program->model_client_->Process(
               model_context, device_itensors_, device_otensors_, 1000, istamp),
           hiai::AI_SUCCESS);
  VLOG(3) << "[NPU] Process cost " << GetCurrentUS() - start_time << " us";
  return subgraph::SUCCESS;
}

void SubgraphCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  engine_.reset(new SubgraphEngine(ctx_.get(),
                                   param.block_idx,
                                   param.program_desc,
                                   param.exec_scope,
                                   param.input_data_names,
                                   param.output_data_names,
                                   param.cached_data_dims));
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
