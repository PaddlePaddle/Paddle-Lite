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

bool SubgraphEngine::BuildDeviceProgram() {
  LOG(INFO) << "[RKNPU]:BuildDeviceProgram";
  int status = 0;
  // Convert all of ops and their input vars and weights and added into the NPU
  // RKNPU IR graph
  subgraph::rknpu::Graph graph;
  const auto& bridges = subgraph::SubgraphBridgeRegistry::Instance();
  if (!origin_program_) {
    BuildOriginProgram();
  }
  const auto& insts = origin_program_->instructions(kRootBlockIdx);
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
  // Collect the valid input and output nodes in the RKNPU IR graph and update
  // the input and output names
  device_itensors_.clear();
  device_otensors_.clear();
  for (size_t i = 0; i < input_names_.size(); i++) {
    CHECK(graph.Has(input_names_[i])) << "[RKNPU] Failed to find input node "
                                      << input_names_[i];
    auto node = graph.Get(input_names_[i]);
    auto precision = node->precision();
    auto layout = node->layout();
    LOG(INFO) << "[RKNPU] Inputs[" << i << "] name: " << input_names_[i]
              << " precision: " << PrecisionToStr(precision)
              << " layout: " << DataLayoutToStr(layout);
    device_itensors_.push_back(node->data());
  }
  for (size_t i = 0; i < output_names_.size(); i++) {
    CHECK(graph.Has(output_names_[i])) << "[RKNPU] Failed to find output node "
                                       << output_names_[i];
    auto node = graph.Get(output_names_[i]);
    auto precision = node->precision();
    auto layout = node->layout();
    LOG(INFO) << "[RKNPU] Outputs[" << i << "] name: " << output_names_[i]
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
        LOG(FATAL) << "[RKNPU] " << output_names_[i]
                   << " can't mutable data with precision type "
                   << PrecisionToStr(precision);
        break;
    }
    device_otensors_.push_back(node->data());
  }
  // Create the RKNPU model and set the input and output nodes
  device_program_ = lite::rknpu::Device::Global().Build(
      model_name_, graph.GetHandle(), device_itensors_, device_otensors_);
  if (device_program_ == nullptr) {
    LOG(WARNING) << "[RKNPU] Build model failed!";
    return false;
  }
  return true;
}

bool SubgraphEngine::LaunchDeviceProgram() {
  LOG(INFO) << "[RKNPU]:LaunchDeviceProgram";
  std::vector<rk::nn::InputInfo> inputs;
  std::vector<rk::nn::OutputInfo> outputs;

  inputs.resize(origin_itensors_.size());
  for (size_t i = 0; i < origin_itensors_.size(); i++) {
    inputs[i].index = i;
    inputs[i].buf = const_cast<void*>(origin_itensors_[i]->raw_data());
    inputs[i].size = origin_itensors_[i]->memory_size();
    inputs[i].pass_through = false;
    inputs[i].type =
        subgraph::rknpu::ToRknpuPrecisionType(origin_itensors_[i]->precision());
    inputs[i].layout = rk::nn::DataLayoutType::NCHW;
  }

  outputs.resize(origin_otensors_.size());
  for (size_t i = 0; i < origin_otensors_.size(); i++) {
    outputs[i].index = i;
    outputs[i].buf = const_cast<void*>(origin_otensors_[i]->raw_data());
    outputs[i].size = origin_otensors_[i]->memory_size();
    outputs[i].want_float = false;
  }

  device_program_->SetInputs(inputs);
  device_program_->Run();
  device_program_->GetOutputs(outputs);
  return true;
}

void SubgraphCompute::PrepareForRun() {
  LOG(INFO) << "[RKNPU]:PrepareForRun";
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
  LOG(INFO) << "[RKNPU]:Run";
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
