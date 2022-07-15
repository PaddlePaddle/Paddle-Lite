// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/increment_compute.h"
#include <algorithm>
#include <vector>
#include "lite/backends/xpu/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void IncrementCompute::PrepareForRun() {
  auto& param = this->template Param<operators::IncrementParam>();
  auto* x = param.X;
  step_guard_ = TargetWrapperXPU::MallocScratchPad(1 * 4);
  cast_out_guard_ = TargetWrapperXPU::MallocScratchPad(1 * 4);
  switch (x->precision()) {
    case PRECISION(kFloat): {
      float step = static_cast<float>(param.step);
      XPU_CALL(xpu_memcpy(
          step_guard_->addr_, &step, sizeof(float), XPU_HOST_TO_DEVICE));
      break;
    }
    case PRECISION(kInt32): {
      int step = static_cast<int>(param.step);
      XPU_CALL(xpu_memcpy(
          step_guard_->addr_, &step, sizeof(int), XPU_HOST_TO_DEVICE));
      break;
    }
    case PRECISION(kInt64): {
      int step = static_cast<int>(param.step);
      XPU_CALL(xpu_memcpy(
          step_guard_->addr_, &step, sizeof(int), XPU_HOST_TO_DEVICE));
      break;
    }
    default: {
      LOG(FATAL) << "unsupport input type: " << PrecisionToStr(x->precision());
    }
  }
}

void IncrementCompute::Run() {
  auto& param = this->template Param<operators::IncrementParam>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto* x = param.X;
  auto* out = param.Out;

  switch (x->precision()) {
    case PRECISION(kFloat): {
      int ret = xdnn::broadcast_add<float>(
          ctx.GetRawContext(),
          x->data<float>(),
          reinterpret_cast<float*>(step_guard_->addr_),
          out->mutable_data<float>(TARGET(kXPU)),
          {static_cast<int>(x->numel())},
          {1});
      CHECK_EQ(ret, 0);
      break;
    }
    case PRECISION(kInt32): {
      int ret =
          xdnn::broadcast_add<int>(ctx.GetRawContext(),
                                   x->data<int>(),
                                   reinterpret_cast<int*>(step_guard_->addr_),
                                   out->mutable_data<int>(TARGET(kXPU)),
                                   {static_cast<int>(x->numel())},
                                   {1});
      CHECK_EQ(ret, 0);
      break;
    }
    case PRECISION(kInt64): {
      cast_out_guard_->Reserve(x->numel() * sizeof(int));
      int ret = xdnn::cast_v2<int64_t, int>(
          ctx.GetRawContext(),
          x->data<int64_t>(),
          reinterpret_cast<int*>(cast_out_guard_->addr_),
          static_cast<int>(x->numel()));
      CHECK_EQ(ret, 0);
      ret = xdnn::broadcast_add<int>(
          ctx.GetRawContext(),
          reinterpret_cast<int*>(cast_out_guard_->addr_),
          reinterpret_cast<int*>(step_guard_->addr_),
          reinterpret_cast<int*>(cast_out_guard_->addr_),
          {static_cast<int>(x->numel())},
          {1});
      CHECK_EQ(ret, 0);
      ret = xdnn::cast_v2<int, int64_t>(
          ctx.GetRawContext(),
          reinterpret_cast<int*>(cast_out_guard_->addr_),
          out->mutable_data<int64_t>(TARGET(kXPU)),
          static_cast<int>(x->numel()));
      CHECK_EQ(ret, 0);
      break;
    }
    default:
      LOG(FATAL) << "unsupport input type: " << PrecisionToStr(x->precision());
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(increment,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::IncrementCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();
