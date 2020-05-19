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
#include "rknpu/rknpu_pub.h"  // NOLINT

namespace paddle {
namespace lite {
namespace kernels {
namespace rknpu {

int SubgraphEngine::BuildDeviceProgram() {
  LOG(INFO) << "[RKNPU]:BuildDeviceProgram";
  int status = 0;
  // Convert all of ops and their input vars and weights and added into the NPU
  // RKNPU IR graph
  subgraph::rknpu::Graph graph;
  const auto& bridges = subgraph::Registry::Instance();
  for (auto& inst : origin_program_) {
    auto op = const_cast<OpLite*>(inst.op());
    CHECK(op);
    op->CheckShape();
    op->InferShape();
    std::string op_type = op->op_info()->Type();
    if (!bridges.Exists(op_type, TARGET(kRKNPU))) {
      return subgraph::FAILED;
    }
    auto kernel = inst.kernel();
    status |= bridges.Select(op_type, TARGET(kRKNPU))(
        reinterpret_cast<void*>(&graph), op, const_cast<KernelBase*>(kernel));
    if (subgraph::CHECK_FAILED(status)) {
      return subgraph::FAILED;
    }
  }
  // Collect the valid input and output nodes in the RKNPU IR graph and update
  // the input and output names
  device_inames_.clear();
  device_onames_.clear();

  for (auto& input_name : input_names_) {
    LOG(INFO) << "[RKNPU] Input node " << input_name;
    if (graph.Has(input_name)) {
      LOG(INFO) << input_name << " Precision "
                << PrecisionToStr(graph.Get(input_name)->precision());
      device_itensors_.push_back(graph.Get(input_name)->data());
      device_inames_.push_back(input_name);
    } else {
      LOG(WARNING) << "[RKNPU] Input node " << input_name
                   << " is ignored because it does not exist.";
    }
  }

  for (auto& output_name : output_names_) {
    LOG(INFO) << "[RKNPU] Output node " << output_name;
    if (graph.Has(output_name)) {
      auto tensor = scope_->FindMutableTensor(output_name);
      LOG(INFO) << output_name << " Precision "
                << PrecisionToStr(tensor->precision());
      device_otensors_.push_back(graph.Get(output_name)->data());
      device_onames_.push_back(output_name);
    } else {
      LOG(WARNING) << "[RKNPU] Output node " << output_name
                   << " is ignored because it does not exist.";
    }
  }
  CHECK(!device_inames_.empty())
      << "[RKNPU] No input nodes found for building NPU model";
  CHECK(!device_onames_.empty())
      << "[RKNPU] No output nodes found for building NPU model";

  device_program_ = lite::rknpu::Device::Global().Build(
      model_name_, graph.GetHandle(), device_itensors_, device_otensors_);
  if (device_program_ == nullptr) {
    LOG(WARNING) << "[RKNPU] Build model failed!";
    return subgraph::FAILED;
  }

  // input
  origin_idims_.resize(input_names_.size());
  origin_itensors_.resize(input_names_.size());
  for (size_t i = 0; i < input_names_.size(); i++) {
    origin_itensors_[i] = scope_->FindMutableTensor(input_names_[i]);
    CHECK(origin_itensors_[i]);
    origin_idims_[i] = origin_itensors_[i]->dims();
  }
  // output
  origin_odims_.resize(output_names_.size());
  origin_otensors_.resize(output_names_.size());
  for (size_t i = 0; i < output_names_.size(); i++) {
    origin_otensors_[i] = scope_->FindMutableTensor(output_names_[i]);
    CHECK(origin_otensors_[i]);
    origin_odims_[i] = origin_otensors_[i]->dims();

    auto output_dims = origin_otensors_[i]->dims();
  }

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

    LOG(INFO) << "[RKNPU] Inputs[" << i << "] name: " << device_inames_[i]
              << " precision: " << PrecisionToStr(precision)
              << " layout: " << DataLayoutToStr(layout);
  }
  for (int i = 0; i < device_onames_.size(); i++) {
    auto node = graph.Get(device_onames_[i]);
    auto precision = node->precision();
    auto layout = node->layout();
    origin_otensors_[i] = scope_->FindMutableTensor(device_onames_[i]);
    CHECK(origin_otensors_[i]);
    origin_odims_[i] = origin_otensors_[i]->dims();
    LOG(INFO) << "[RKNPU] Outputs[" << i << "] name: " << device_onames_[i]
              << " precision: " << PrecisionToStr(precision)
              << " layout: " << DataLayoutToStr(layout);
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
        LOG(FATAL) << "[RKNPU] " << device_onames_[i]
                   << " can't mutable data with precision type "
                   << PrecisionToStr(precision);
        break;
    }
  }
  return status;
}

int SubgraphEngine::LaunchDeviceProgram() {
  LOG(INFO) << "[RKNPU]:LaunchDeviceProgram";
  std::vector<rk::nn::InputInfo> inputs;
  std::vector<rk::nn::OutputInfo> outputs;

  inputs.resize(device_itensors_.size());
  for (size_t i = 0; i < device_itensors_.size(); i++) {
    inputs[i].index = i;
    inputs[i].buf = const_cast<void*>(origin_itensors_[i]->raw_data());
    inputs[i].size = origin_itensors_[i]->memory_size();
    inputs[i].pass_through = false;
    inputs[i].type =
        subgraph::rknpu::ToRknpuPrecisionType(origin_itensors_[i]->precision());
    inputs[i].layout = rk::nn::DataLayoutType::NCHW;
  }

  outputs.resize(device_otensors_.size());
  for (size_t i = 0; i < device_otensors_.size(); i++) {
    outputs[i].index = i;
    outputs[i].buf = const_cast<void*>(origin_otensors_[i]->raw_data());
    outputs[i].size = origin_otensors_[i]->memory_size();
    outputs[i].want_float = false;
  }

  device_program_->SetInputs(inputs);
  device_program_->Run();
  device_program_->GetOutputs(outputs);
  return 0;
}

void SubgraphCompute::PrepareForRun() {
  LOG(INFO) << "[RKNPU]:PrepareForRun";
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
  LOG(INFO) << "[RKNPU]:Run";
  CHECK(engine_);
  engine_->Launch();
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
