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

#include "lite/kernels/xpu/cast_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename InType>
void CastCompute<InType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto* x = param.X;
  auto* out = param.Out;
  int out_dtype = param.out_dtype;
  auto* in_data = x->template data<InType>();
  int numel = x->numel();
  if (numel <= 0) {
    return;
  }

  int r = -1;
  // BOOL = 0;INT16 = 1;INT32 = 2;INT64 = 3;FP16 = 4;FP32 = 5;FP64 = 6;
  // SIZE_T = 19;UINT8 = 20;INT8 = 21;
  if (out_dtype == 5) {
    auto* out_data = out->template mutable_data<float>(TARGET(kXPU));
    r = xdnn::cast_v2<InType, float>(
        ctx.GetRawContext(), in_data, out_data, numel);
  } else if (out_dtype == 2) {
    auto* out_data = out->template mutable_data<int>(TARGET(kXPU));
    r = xdnn::cast_v2<InType, int>(
        ctx.GetRawContext(), in_data, out_data, numel);
  } else if (out_dtype == 3) {
    auto* out_data = out->template mutable_data<int64_t>(TARGET(kXPU));
    r = xdnn::cast_v2<InType, int64_t>(
        ctx.GetRawContext(), in_data, out_data, numel);
  } else {
    LOG(FATAL) << "cast from in_type("
               << lite_api::PrecisionToStr(
                      lite_api::PrecisionTypeTrait<InType>::Type())
               << ") to out_type(" << out_dtype << ") is not supported.";
  }
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(cast,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::CastCompute<float>,
                     cast_fp32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(cast,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::CastCompute<int>,
                     cast_i32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(cast,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::CastCompute<int64_t>,
                     cast_i64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(cast,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::CastCompute<uint8_t>,
                     cast_u8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kUInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(cast,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::CastCompute<int8_t>,
                     cast_i8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(cast,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::CastCompute<int8_t>,
                     cast_bool)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();
