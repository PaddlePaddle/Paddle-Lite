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

bool SubgraphEngine::BuildDeviceProgram() {
  int status = 0;
  if (!origin_program_) {
    BuildOriginProgram();
  }

  // Convert all of ops and their input vars and weights and added into the XPU
  // IR graph
  subgraph::xpu::Graph graph;
  const auto& bridges = subgraph::SubgraphBridgeRegistry::Instance();
  const auto& insts = origin_program_->instructions(kRootBlockIdx);
  for (auto& inst : insts) {
    auto op = const_cast<OpLite*>(inst.op());
    CHECK(op);
    op->CheckShape();
    op->InferShape();
    std::string op_type = op->op_info()->Type();
    if (!bridges.Exists(op_type, TARGET(kXPU))) {
      return false;
    }
    auto kernel = inst.kernel();
    status |= bridges.Select(op_type, TARGET(kXPU))(
        reinterpret_cast<void*>(&graph), op, const_cast<KernelBase*>(kernel));
    if (subgraph::CHECK_FAILED(status)) {
      return false;
    }
  }

  // Collect the input and output nodes of the XPU IR graph
  std::vector<xtcl::xExpr*> device_inodes;
  std::vector<xtcl::xExpr*> device_onodes;
  for (size_t i = 0; i < input_names_.size(); i++) {
    CHECK(graph.Has(input_names_[i]));
    CHECK(graph.Get(input_names_[i])->is_data());
    device_inodes.push_back(graph.Get(input_names_[i])->data().get());
  }
  for (size_t i = 0; i < output_names_.size(); i++) {
    if (graph.Has(output_names_[i])) {
      device_onodes.push_back(graph.Get(output_names_[i])->data().get());
    } else {
      // update output_names_ and origin_otensors because some outputs may be
      // useless
      output_names_.erase(output_names_.begin() + i);
      origin_otensors_.erase(origin_otensors_.begin() + i);
      i--;
    }
  }
  CHECK_GT(output_names_.size(), 0);
  CHECK_EQ(output_names_.size(), origin_otensors_.size());

  // Build the XPU IR graph to the XPU runtime for inference
  device_program_ = lite::xpu::Device::Global().Build(
      &graph.builder_, &graph.params_, &device_onodes);
  if (device_program_ == nullptr) {
    LOG(WARNING) << "[XPU] Build model failed!";
    return false;
  }
  origin_otypes_.resize(output_names_.size());
  origin_odims_.resize(output_names_.size());
  for (size_t i = 0; i < output_names_.size(); i++) {
    origin_otypes_[i] = graph.Get(output_names_[i])->precision();
    origin_odims_[i] = origin_otensors_[i]->dims().Vectorize();
  }

  // Query and check the dimensions of input and output tensors
  device_itensors_.resize(input_names_.size());
  device_otensors_.resize(output_names_.size());
  for (size_t i = 0; i < input_names_.size(); i++) {
    VLOG(3) << "[XPU] Inputs[" << i << "] name: " << input_names_[i]
            << " dims: " << DDim(origin_idims_[i]).repr();
    // Prepare the device input tensors which share data with the origin input
    // tensors
    device_itensors_[i].data = nullptr;
    device_itensors_[i].ctx.device_type =
        subgraph::xpu::CvtDLDeviceType(TARGET(kHost));
    device_itensors_[i].ctx.device_id = 0;
    device_itensors_[i].ndim = origin_idims_[i].size();
    device_itensors_[i].dtype =
        subgraph::xpu::CvtDLDataType(origin_itensors_[i]->precision());
    device_itensors_[i].shape = const_cast<int64_t*>(
        static_cast<const int64_t*>(origin_idims_[i].data()));
    device_itensors_[i].strides = nullptr;
    device_itensors_[i].byte_offset = 0;
  }
  for (size_t i = 0; i < output_names_.size(); i++) {
    VLOG(3) << "[XPU] Outputs[" << i << "] name: " << output_names_[i]
            << " dims: " << DDim(origin_odims_[i]).repr();
    // Prepare the device output tensors which share data with the origin output
    // tensors
    origin_otensors_[i]->Resize(origin_odims_[i]);
    auto& precision = origin_otypes_[i];
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
        LOG(FATAL) << "[XPU] " << output_names_[i]
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
        static_cast<const int64_t*>(origin_odims_[i].data()));
    device_otensors_[i].strides = nullptr;
    device_otensors_[i].byte_offset = 0;
  }
  return true;
}

bool SubgraphEngine::LaunchDeviceProgram() {
  for (size_t i = 0; i < device_itensors_.size(); i++) {
    // Update the data pointer of DLTensor to track the origin input tensors
    device_itensors_[i].data =
        const_cast<void*>(origin_itensors_[i]->raw_data());
    device_program_->SetInput(input_names_[i], &device_itensors_[i]);
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
  return true;
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

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(subgraph,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::SubgraphCompute,
                     def)
    .BindInput("Inputs",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Outputs",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .Finalize();
