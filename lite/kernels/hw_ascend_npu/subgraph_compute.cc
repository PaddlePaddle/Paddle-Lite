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

#include "lite/kernels/hw_ascend_npu/subgraph_compute.h"
#include <sys/time.h>
#include <time.h>
#include <utility>
#include "lite/backends/hw_ascend_npu/device.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/hw_ascend_npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/paddle_use_bridges.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace hw_ascend_npu {

int SubgraphEngine::BuildDeviceProgram() {
  int status = 0;
  // Convert all of ops and their input vars and weights and added into
  // the HWAscendNPU IR graph
  subgraph::hw_ascend_npu::Graph graph;
  const auto& bridges = subgraph::Registry::Instance();
  for (auto& inst : origin_program_) {
    auto op = const_cast<OpLite*>(inst.op());
    CHECK(op);
    op->CheckShape();
    op->InferShape();
    std::string op_type = op->op_info()->Type();
    if (!bridges.Exists(op_type, TARGET(kHWAscendNPU))) {
      return subgraph::FAILED;
    }
    auto kernel = inst.kernel();
    status |= bridges.Select(op_type, TARGET(kHWAscendNPU))(
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
        LOG(WARNING) << "[HWAscendNPU] Input node " << input_name
                     << " is ignored because it is not a data node.";
      }
    } else {
      LOG(WARNING) << "[HWAscendNPU] Input node " << input_name
                   << " is ignored because it does not exist.";
    }
  }
  for (auto& output_name : output_names_) {
    if (graph.Has(output_name)) {
      device_onodes.push_back(*graph.Get(output_name)->data());
      device_onames_.push_back(output_name);
    } else {
      LOG(WARNING) << "[HWAscendNPU] Output node " << output_name
                   << " is ignored because it does not exist.";
    }
  }
  CHECK(!device_inames_.empty())
      << "[HWAscendNPU] No input nodes found for building NPU model";
  CHECK(!device_onames_.empty())
      << "[HWAscendNPU] No output nodes found for building NPU model";

  // Build the IR graph to om model as the device program
  if (device_program_map_.count(inputs_shape_) > 0) {
    return status;
  }
  auto device_client =
      lite::hw_ascend_npu::Device::Global().Build(device_inodes, device_onodes);
  if (device_client == nullptr) {
    LOG(WARNING) << "[HWAscendNPU] Build model failed!";
    return subgraph::FAILED;
  }
  auto device_program = std::make_shared<device_program_t>(device_client);
  device_program_map_[inputs_shape_] = device_program;

  // Query and check the dimensions of valid input and output tensors
  std::vector<TensorDesc> device_idims, device_odims;
  if (device_program->client->GetModelIOTensorDim(&device_idims,
                                                  &device_odims) != 0) {
    LOG(WARNING) << "[HWAscendNPU] Get the dimensions of input and output "
                    "tensors failed!";
    return subgraph::FAILED;
  }
  device_program->device_idims = device_idims;
  device_program->device_odims = device_odims;

  CHECK_EQ(device_idims.size(), device_inames_.size());
  CHECK_EQ(device_odims.size(), device_onames_.size());
  origin_idims_.resize(device_inames_.size());
  origin_itensors_.resize(device_inames_.size());
  origin_odims_.resize(device_onames_.size());
  origin_otensors_.resize(device_onames_.size());

  for (size_t i = 0; i < device_inames_.size(); i++) {
    auto node = graph.Get(device_inames_[i]);
    auto precision = node->precision();
    auto layout = node->layout();
    origin_itensors_[i] = scope_->FindMutableTensor(device_inames_[i]);
    CHECK(origin_itensors_[i]);
    origin_idims_[i] = origin_itensors_[i]->dims();
    VLOG(3) << "[HWAscendNPU] Inputs[" << i << "] name: " << device_inames_[i]
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
  }
  device_program->origin_idims = origin_idims_;

  for (size_t i = 0; i < device_onames_.size(); i++) {
    auto node = graph.Get(device_onames_[i]);
    auto precision = node->precision();
    auto layout = node->layout();
    origin_otensors_[i] = scope_->FindMutableTensor(device_onames_[i]);
    CHECK(origin_otensors_[i]);
    origin_odims_[i] = origin_otensors_[i]->dims();
    VLOG(3) << "[HWAscendNPU] Outputs[" << i << "] name: " << device_onames_[i]
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
        LOG(FATAL) << "[HWAscendNPU] " << device_onames_[i]
                   << " can't mutable data with precision type "
                   << PrecisionToStr(precision);
        break;
    }
    device_program->origin_odims = origin_odims_;

    CHECK_EQ(origin_odims_[i].production(),
             device_odims[i].GetNumber() * device_odims[i].GetChannel() *
                 device_odims[i].GetHeight() * device_odims[i].GetWidth());
  }
  return status;
}

int SubgraphEngine::LaunchDeviceProgram() {
  // Copy the data of origin input tensors to the buffer of input HWAscendNPU
  // tensors
  auto device_program = device_program_map_[inputs_shape_];
  int ret = 0;

  ret = device_program->client->SetInput(origin_itensors_,
                                         device_program->origin_idims);
  if (ret != 0) {
    return ret;
  }

  device_program->client->CreateOutput(device_program->origin_odims);

  // run inference
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };
  auto start_time = GetCurrentUS();
  CHECK_EQ(device_program->client->Process(), 0);
  VLOG(3) << "[HWAscendNPU] Process cost " << GetCurrentUS() - start_time
          << " us";

  device_program->client->GetOutput(&origin_otensors_);

  return 0;
}

bool SubgraphEngine::InputShapeChanged() {
  std::vector<std::vector<int64_t>> new_shape;
  for (auto origin_itensor : origin_itensors_) {
    new_shape.push_back(origin_itensor->dims().Vectorize());
  }
  inputs_shape_ = new_shape;
  if (device_program_map_.count(inputs_shape_) > 0) {
    return false;
  }
  return true;
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

}  // namespace hw_ascend_npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(subgraph,
                     kHWAscendNPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::hw_ascend_npu::SubgraphCompute,
                     def)
    .BindInput("Inputs",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Outputs",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .Finalize();
