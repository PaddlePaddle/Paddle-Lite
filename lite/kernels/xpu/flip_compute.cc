// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/flip_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename InType>
void FlipCompute<InType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto* x = param.X;
  auto* out = param.Out;
  auto& axis = param.axis;
  int ndims = axis.size();
  for (int i = 0; i < ndims; i++) {
    if (axis[i] < 0) {
      axis[i] = axis[i] + ndims;
    }
  }
  const auto x_dims = x->dims();
  std::vector<int> x_shape_host(x_dims.size(), 0);
  for (int i = 0; i < x_dims.size(); ++i) {
    x_shape_host[i] = x_dims[i];
  }

  int numel = x->numel();
  if (numel <= 0) {
    return;
  }

  int r = xdnn::flip<InType>(ctx.GetRawContext(),
                             x->template data<InType>(),
                             out->template mutable_data<InType>(TARGET(kXPU)),
                             x_shape_host,
                             axis);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(flip,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::FlipCompute<float>,
                     xflip_fp32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(flip,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::FlipCompute<float16>,
                     xflip_fp16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(flip,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::FlipCompute<int>,
                     xflip_i32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(flip,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::FlipCompute<int64_t>,
                     xflip_i64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(flip,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::FlipCompute<int16_t>,
                     xflip_i16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(flip,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::FlipCompute<int8_t>,
                     xflip_i8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(flip,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::FlipCompute<int8_t>,
                     xflip_bool)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();
