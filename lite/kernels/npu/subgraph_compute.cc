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

std::string SubgraphEngine::GenerateModelCacheName() const {
  auto inames = device_inames_;
  auto onames = device_onames_;
  std::stable_sort(inames.begin(), inames.end());

  std::string model_cache_name = "subgraph_" + std::to_string(block_idx_);
  for (auto iname : inames) {
    model_cache_name += "_";
    auto itensor = scope_->FindTensor(iname);
    int tmp = 0;
    for (auto i : itensor->dims().Vectorize()) {
      tmp += i * i;
    }
    model_cache_name += std::to_string(tmp % 1999);
  }
  model_cache_name += "_.om";

  return model_cache_name;
}

int SubgraphEngine::BuildDeviceProgram() {
  int status = 0;
  // Convert all of ops and their input vars and weights and added into the NPU
  // HiAI IR graph
  subgraph::npu::Graph graph;
  const auto& bridges = subgraph::Registry::Instance();
  for (auto& inst : origin_program_) {
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
  // Collect the valid input and output nodes in the HiAI IR graph and update
  // the input and output names
  device_inames_.clear();
  device_onames_.clear();
  std::vector<ge::Operator> device_inodes;
  std::vector<ge::Operator> device_onodes;
  for (auto& input_name : input_names_) {
    if (graph.Has(input_name)) {
      if (graph.Get(input_name)->is_data()) {
        device_inodes.push_back(*graph.Get(input_name)->data());
        device_inames_.push_back(input_name);
      } else {
        LOG(WARNING) << "[NPU] Input node " << input_name
                     << " is ignored because it is not a data node.";
      }
    } else {
      LOG(WARNING) << "[NPU] Input node " << input_name
                   << " is ignored because it does not exist.";
    }
  }
  for (auto& output_name : output_names_) {
    if (graph.Has(output_name)) {
      device_onodes.push_back(*graph.Get(output_name)->data());
      device_onames_.push_back(output_name);
    } else {
      LOG(WARNING) << "[NPU] Output node " << output_name
                   << " is ignored because it does not exist.";
    }
  }
  CHECK(!device_inames_.empty())
      << "[NPU] No input nodes found for building NPU model";
  CHECK(!device_onames_.empty())
      << "[NPU] No output nodes found for building NPU model";

  // Build the HiAI IR graph to HiAI om model as the device program
  if (device_program_map_.count(inputs_shape_) > 0) {
    return status;
  }
  std::string model_cache_full_dir =
      model_cache_dir_.empty() ? "" : model_cache_dir_ + "/" +
                                          GenerateModelCacheName();
  auto device_client = lite::npu::Device::Global().Build(
      model_name_, device_inodes, device_onodes, model_cache_full_dir);
  if (device_client == nullptr) {
    LOG(WARNING) << "[NPU] Build model failed!";
    return subgraph::FAILED;
  }
  auto device_program = std::make_shared<device_program_t>(device_client);
  if (!inputs_shape_.empty()) {
    device_program_map_[inputs_shape_] = device_program;
  }

  // Query and check the dimensions of valid input and output tensors
  std::vector<hiai::TensorDimension> device_idims, device_odims;
  if (device_program->client->GetModelIOTensorDim(
          model_name_, device_idims, device_odims) != hiai::AI_SUCCESS) {
    LOG(WARNING)
        << "[NPU] Get the dimensions of input and output tensors failed!";
    return subgraph::FAILED;
  }
  device_program->device_idims = device_idims;
  device_program->device_odims = device_odims;

  CHECK_EQ(device_idims.size(), device_inames_.size());
  CHECK_EQ(device_odims.size(), device_onames_.size());
  origin_idims_.resize(device_inames_.size());
  origin_itensors_.resize(device_inames_.size());
  device_itensors_.resize(device_inames_.size());
  origin_odims_.resize(device_onames_.size());
  origin_otensors_.resize(device_onames_.size());
  device_otensors_.resize(device_onames_.size());

  for (int i = 0; i < device_inames_.size(); i++) {
    auto node = graph.Get(device_inames_[i]);
    auto precision = node->precision();
    auto layout = node->layout();
    origin_itensors_[i] = scope_->FindMutableTensor(device_inames_[i]);
    CHECK(origin_itensors_[i]);
    origin_idims_[i] = origin_itensors_[i]->dims();
    VLOG(3) << "[NPU] Inputs[" << i << "] name: " << device_inames_[i]
            << " precision: " << PrecisionToStr(precision)
            << " layout: " << DataLayoutToStr(layout) << " dims: {"
            << device_idims[i].GetNumber() << ","
            << device_idims[i].GetChannel() << ","
            << device_idims[i].GetHeight() << "," << device_idims[i].GetWidth()
            << "}";
    // Prepare the device input tensors
    CHECK_EQ(origin_idims_[i].production(),
             device_idims[i].GetNumber() * device_idims[i].GetChannel() *
                 device_idims[i].GetHeight() * device_idims[i].GetWidth());
    device_itensors_[i].reset(new hiai::AiTensor);
    device_itensors_[i]->Init(&(device_idims[i]));
  }
  device_program->origin_idims = origin_idims_;

  for (int i = 0; i < device_onames_.size(); i++) {
    auto node = graph.Get(device_onames_[i]);
    auto precision = node->precision();
    auto layout = node->layout();
    origin_otensors_[i] = scope_->FindMutableTensor(device_onames_[i]);
    CHECK(origin_otensors_[i]);
    origin_odims_[i] = origin_otensors_[i]->dims();
    VLOG(3) << "[NPU] Outputs[" << i << "] name: " << device_onames_[i]
            << " precision: " << PrecisionToStr(precision)
            << " layout: " << DataLayoutToStr(layout) << " dims: {"
            << device_odims[i].GetNumber() << ","
            << device_odims[i].GetChannel() << ","
            << device_odims[i].GetHeight() << "," << device_odims[i].GetWidth()
            << "}";
    // Prepare the device output tensors
    switch (precision) {
      case PRECISION(kFloat):
        origin_otensors_[i]->mutable_data<float>();
        break;
      case PRECISION(kBool):
        origin_otensors_[i]->mutable_data<bool>();
        break;
      case PRECISION(kInt8):
        origin_otensors_[i]->mutable_data<int8_t>();
        break;
      case PRECISION(kInt16):
        origin_otensors_[i]->mutable_data<int16_t>();
        break;
      case PRECISION(kInt32):
        origin_otensors_[i]->mutable_data<int32_t>();
        break;
      case PRECISION(kInt64):
        origin_otensors_[i]->mutable_data<int64_t>();
        break;
      default:
        LOG(FATAL) << "[NPU] " << device_onames_[i]
                   << " can't mutable data with precision type "
                   << PrecisionToStr(precision);
        break;
    }
    device_program->origin_odims = origin_odims_;

    CHECK_EQ(origin_odims_[i].production(),
             device_odims[i].GetNumber() * device_odims[i].GetChannel() *
                 device_odims[i].GetHeight() * device_odims[i].GetWidth());
    device_otensors_[i].reset(new hiai::AiTensor);
    device_otensors_[i]->Init(&(device_odims[i]));
  }
  return status;
}

int SubgraphEngine::LaunchDeviceProgram() {
  // Copy the data of origin input tensors to the buffer of input HiAI tensors
  // init device_itensors_, device_otensors_, origin_otensors_
  auto device_program = device_program_map_[inputs_shape_];

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
  CHECK_EQ(device_program->client->Process(
               model_context, device_itensors_, device_otensors_, 1000, istamp),
           hiai::AI_SUCCESS);
  VLOG(3) << "[NPU] Process cost " << GetCurrentUS() - start_time << " us";

  return 0;
}

int SubgraphEngine::Build() {
  if (device_program_map_.count(inputs_shape_) > 0) {
    return subgraph::SUCCESS;
  }
  // In order to attach all of the ops of the block desc, we need to build the
  // original program firstly.
  BuildOriginProgram();
  // Run InferShape() of all of ops, and convert Paddle ops to NPU/XPU IR graph
  build_device_program_status_ = BuildDeviceProgram();
  return build_device_program_status_;
}

void SubgraphEngine::InitDeviceTensor() {
  auto device_program = device_program_map_[inputs_shape_];
  for (size_t i = 0; i < device_itensors_.size(); i++) {
    if (device_itensors_[i]->GetBuffer() != origin_itensors_[i]->raw_data()) {
      VLOG(3) << "init device_itensors and share input tensor buf between "
                 "device and host";
      device_itensors_[i]->Init(&(device_program->device_idims[i]));
      std::memcpy(device_itensors_[i]->GetBuffer(),
                  origin_itensors_[i]->raw_data(),
                  origin_itensors_[i]->memory_size());
      // share data buf between device_itensor and origin_itensor
      std::shared_ptr<Buffer> buffer =
          std::make_shared<Buffer>(device_itensors_[i]->GetBuffer(),
                                   lite_api::TargetType::kHost,
                                   device_itensors_[i]->GetSize());
      origin_itensors_[i]->ResetBuffer(buffer, device_itensors_[i]->GetSize());
    }
  }
  for (size_t i = 0; i < device_otensors_.size(); i++) {
    if (device_otensors_[i]->GetBuffer() != origin_otensors_[i]->raw_data()) {
      VLOG(3) << "init device_otensors and share output tensor buf between "
                 "device and host";
      device_otensors_[i]->Init(&(device_program->device_odims[i]));
      // share data buf between device_itensor and origin_itensor
      origin_otensors_[i]->Resize(device_program->origin_odims[i]);
      std::shared_ptr<Buffer> buffer =
          std::make_shared<Buffer>(device_otensors_[i]->GetBuffer(),
                                   lite_api::TargetType::kHost,
                                   device_otensors_[i]->GetSize());
      origin_otensors_[i]->ResetBuffer(buffer, device_otensors_[i]->GetSize());
    }
  }
}

bool SubgraphEngine::InputShapeChanged() {
  std::vector<std::vector<int64_t>> new_shape;
  for (auto origin_itensor : origin_itensors_) {
    new_shape.push_back(origin_itensor->dims().Vectorize());
  }
  if (inputs_shape_ == new_shape) {
    return false;
  }
  inputs_shape_ = new_shape;
  return true;
}

void SubgraphCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  engine_.reset(new SubgraphEngine(ctx_.get(),
                                   param.sub_block_idx,
                                   param.sub_block_desc,
                                   param.input_data_names,
                                   param.output_data_names,
                                   param.scope,
                                   NPUContext::SubgraphModelCacheDir()));
  CHECK(engine_);
  engine_->Build();
}

void SubgraphCompute::Run() {
  CHECK(engine_);
  engine_->Launch();
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
