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

#include "lite/kernels/xpu/subgraph_compute.h"
#include <sys/time.h>
#include <time.h>
#include <utility>
#include "lite/backends/xpu/device.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/xpu/bridges/graph.h"
#include "lite/kernels/xpu/bridges/paddle_use_bridges.h"
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

int SubgraphEngine::BuildDeviceProgram() {
  int status = 0;
  // Convert all of ops and their input vars and weights and added into the XPU
  // IR graph
  subgraph::xpu::Graph graph;
  const auto& bridges = subgraph::Registry::Instance();
  for (auto& inst : origin_program_) {
    auto op = inst.op();
    CHECK(op);
    op->CheckShape();
    op->InferShape();
    std::string op_type = op->op_info()->Type();
    if (!bridges.Exists("XPU", op_type)) {
      return subgraph::FAILED;
    }
    auto kernel = inst.kernel();
    status |= bridges.Select("XPU", op_type)(reinterpret_cast<void*>(&graph),
                                             const_cast<OpLite*>(op),
                                             const_cast<KernelBase*>(kernel));
    if (subgraph::CHECK_FAILED(status)) {
      return subgraph::FAILED;
    }
  }
  // Obtain the output nodes of the XPU IR graph and build the graph to the XPU
  // runtime
  device_inames_.clear();
  device_onames_.clear();
  std::vector<xtcl::xExpr*> device_inodes;
  std::vector<xtcl::xExpr*> device_onodes;
  for (auto& input_name : input_names_) {
    if (graph.HasNode(input_name)) {
      if (!graph.GetType(input_name).persistable()) {
        device_inodes.push_back(graph.GetNode(input_name).get());
        device_inames_.push_back(input_name);
      } else {
        LOG(WARNING) << "[XPU] Input node " << input_name
                     << " is skipped because it is a persistable node.";
      }
    } else {
      LOG(WARNING) << "[XPU] Input node " << input_name
                   << " is skipped because it does not exist.";
    }
  }
  for (auto& output_name : output_names_) {
    if (graph.HasNode(output_name)) {
      device_onodes.push_back(graph.GetNode(output_name).get());
      device_onames_.push_back(output_name);
    } else {
      LOG(WARNING) << "[XPU] Output node " << output_name
                   << " is skipped because it does not exist.";
    }
  }
  CHECK(!device_inames_.empty())
      << "[XPU] No input nodes found for building XPU model";
  CHECK(!device_onames_.empty())
      << "[XPU] No output nodes found for building XPU model";
  device_program_ = lite::xpu::Device::Global().Build(
      &graph.builder_, &graph.params_, &device_onodes);
  if (device_program_ == nullptr) {
    LOG(WARNING) << "[XPU] Build model failed!";
    return subgraph::FAILED;
  }

  // Query and check the dimensions of input and output tensors
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
    VLOG(3) << "[XPU] Inputs[" << i
            << "] precision: " << PrecisionToStr(precision)
            << " layout: " << DataLayoutToStr(layout)
            << " dims: " << origin_idims_[i];
    // Prepare the device input tensors which share data with the origin input
    // tensors
    device_itensors_[i].data = nullptr;
    device_itensors_[i].ctx.device_type =
        subgraph::xpu::CvtDLDeviceType(TARGET(kHost));
    device_itensors_[i].ctx.device_id = 0;
    device_itensors_[i].ndim = origin_idims_[i].size();
    device_itensors_[i].dtype = subgraph::xpu::CvtDLDataType(precision);
    device_itensors_[i].shape = const_cast<int64_t*>(
        static_cast<const int64_t*>(origin_idims_[i].data().data()));
    device_itensors_[i].strides = nullptr;
    device_itensors_[i].byte_offset = 0;
  }
  for (int i = 0; i < device_onames_.size(); i++) {
    auto type = graph.GetType(device_onames_[i]);
    auto precision = type.precision();
    auto layout = type.layout();
    origin_otensors_[i] = scope_->FindMutableTensor(device_onames_[i]);
    CHECK(origin_otensors_[i]);
    origin_odims_[i] = origin_otensors_[i]->dims();
    VLOG(3) << "[XPU] Outputs[" << i
            << "] precision: " << PrecisionToStr(precision)
            << " layout: " << DataLayoutToStr(layout)
            << " dims: " << origin_odims_[i];
    // Prepare the device output tensors which share data with the origin output
    // tensors
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
        LOG(FATAL) << "[XPU] " << device_onames_[i]
                   << " can't mutable data with precision type "
                   << PrecisionToStr(precision);
        break;
    }
    device_otensors_[i].data = nullptr;
    device_otensors_[i].ctx.device_type =
        subgraph::xpu::CvtDLDeviceType(TARGET(kHost));
    device_otensors_[i].ctx.device_id = 0;
    device_otensors_[i].ndim = origin_odims_[i].size();
    device_otensors_[i].dtype = subgraph::xpu::CvtDLDataType(precision);
    device_otensors_[i].shape = const_cast<int64_t*>(
        static_cast<const int64_t*>(origin_odims_[i].data().data()));
    device_otensors_[i].strides = nullptr;
    device_otensors_[i].byte_offset = 0;
  }
  return status;
}

int SubgraphEngine::LaunchDeviceProgram() {
  for (size_t i = 0; i < device_itensors_.size(); i++) {
    // Update the data pointer of DLTensor to track the origin input tensors
    device_itensors_[i].data =
        const_cast<void*>(origin_itensors_[i]->raw_data());
    device_program_->SetInputZeroCopy(device_inames_[i], &device_itensors_[i]);
  }
  // Run the XPU model
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };
  auto start_time = GetCurrentUS();
  device_program_->Run();
  VLOG(3) << "[XPU] Process cost " << GetCurrentUS() - start_time << " us";
  for (size_t i = 0; i < device_otensors_.size(); i++) {
    // Update the data pointer of DLTensor to track the origin output tensors
    device_otensors_[i].data =
        const_cast<void*>(origin_otensors_[i]->raw_data());
    device_program_->CopyOutputTo(i, &device_otensors_[i]);
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

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(subgraph,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::SubgraphCompute,
                     def)
    .BindInput("Inputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Outputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
