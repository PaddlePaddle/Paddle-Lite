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
#include <utility>
#include "ai_ddk_lib/include/hiai_ir_build.h"
#include "lite/backends/npu/device.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/paddle_use_bridges.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace npu {

int SubgraphEngine::BuildDeviceProgram() {
  int status = 0;
  // Convert all of ops and their input vars and weights and added into the NPU
  // HiAI IR graph
  subgraph::npu::Graph graph;
  const auto& bridges = subgraph::Registry::Instance();
  for (auto& inst : origin_program_) {
    auto op = inst.op();
    CHECK(op);
    op->CheckShape();
    op->InferShape();
    std::string op_type = op->op_info()->Type();
    if (!bridges.Exists("NPU", op_type)) {
      return subgraph::FAILED;
    }
    auto kernel = inst.kernel();
    status |= bridges.Select("NPU", op_type)(reinterpret_cast<void*>(&graph),
                                             const_cast<OpLite*>(op),
                                             const_cast<KernelBase*>(kernel));
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
    if (graph.HasNode(input_name)) {
      if (!graph.GetType(input_name).persistable()) {
        device_inodes.push_back(*graph.GetNode(input_name));
        device_inames_.push_back(input_name);
      } else {
        LOG(WARNING) << "[NPU] Input node " << input_name
                     << " is skipped because it is a persistable node.";
      }
    } else {
      LOG(WARNING) << "[NPU] Input node " << input_name
                   << " is skipped because it does not exist.";
    }
  }
  for (auto& output_name : output_names_) {
    if (graph.HasNode(output_name)) {
      device_onodes.push_back(*graph.GetNode(output_name));
      device_onames_.push_back(output_name);
    } else {
      LOG(WARNING) << "[NPU] Output node " << output_name
                   << " is skipped because it does not exist.";
    }
  }
  CHECK(!device_inames_.empty())
      << "[NPU] No input nodes found for building NPU model";
  CHECK(!device_onames_.empty())
      << "[NPU] No output nodes found for building NPU model";
  // Build the HiAI IR graph to HiAI om model as the device program
  device_program_ = lite::npu::Device::Global().Build(
      model_name_, device_inodes, device_onodes);
  if (device_program_ == nullptr) {
    LOG(WARNING) << "[NPU] Build model failed!";
    return subgraph::FAILED;
  }

  // Query and check the dimensions of valid input and output tensors
  std::vector<hiai::TensorDimension> device_idims, device_odims;
  if (device_program_->GetModelIOTensorDim(
          model_name_, device_idims, device_odims) != hiai::AI_SUCCESS) {
    LOG(WARNING)
        << "[NPU] Get the dimensions of input and output tensors failed!";
    return subgraph::FAILED;
  }
  CHECK_EQ(device_idims.size(), device_inames_.size());
  CHECK_EQ(device_odims.size(), device_onames_.size());
  origin_idims_.resize(device_inames_.size());
  origin_itensors_.resize(device_inames_.size());
  device_itensors_.resize(device_inames_.size());
  origin_odims_.resize(device_onames_.size());
  origin_otensors_.resize(device_onames_.size());
  device_otensors_.resize(device_onames_.size());
  for (int i = 0; i < device_inames_.size(); i++) {
    auto type = graph.GetType(device_inames_[i]);
    auto precision = type.precision();
    auto layout = type.layout();
    origin_itensors_[i] = scope_->FindMutableTensor(device_inames_[i]);
    CHECK(origin_itensors_[i]);
    origin_idims_[i] = origin_itensors_[i]->dims();
    VLOG(3) << "[NPU] Inputs[" << i
            << "] precision: " << PrecisionToStr(precision)
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
  for (int i = 0; i < device_onames_.size(); i++) {
    auto type = graph.GetType(device_onames_[i]);
    auto precision = type.precision();
    auto layout = type.layout();
    origin_otensors_[i] = scope_->FindMutableTensor(device_onames_[i]);
    CHECK(origin_otensors_[i]);
    origin_odims_[i] = origin_otensors_[i]->dims();
    VLOG(3) << "[NPU] Outputs[" << i
            << "] precision: " << PrecisionToStr(precision)
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
  for (size_t i = 0; i < device_itensors_.size(); i++) {
    std::memcpy(device_itensors_[i]->GetBuffer(),
                origin_itensors_[i]->raw_data(),
                origin_itensors_[i]->memory_size());
  }
  // Run the HiAI model by name
  std::string key = "model_name";  // Note: key seems must be model_name
  model_context_.AddPara(key, model_name_);
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };
  int istamp;
  auto start_time = GetCurrentUS();
  CHECK_EQ(
      device_program_->Process(
          model_context_, device_itensors_, device_otensors_, 1000, istamp),
      hiai::AI_SUCCESS);
  VLOG(3) << "[NPU] Process cost " << GetCurrentUS() - start_time << " us";
  // Copy the data of output HiAI tensor to the buffer of origin output tensors
  for (size_t i = 0; i < device_otensors_.size(); i++) {
    std::memcpy(const_cast<void*>(origin_otensors_[i]->raw_data()),
                device_otensors_[i]->GetBuffer(),
                device_otensors_[i]->GetSize());
  }
  return 0;
}

void SubgraphCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  engine_.reset(new SubgraphEngine(param.sub_block_idx,
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

}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(subgraph,
                     kNPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::npu::SubgraphCompute,
                     def)
    .BindInput("Inputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Outputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
