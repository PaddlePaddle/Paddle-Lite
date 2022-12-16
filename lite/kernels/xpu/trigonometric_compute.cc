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

#include "lite/kernels/xpu/trigonometric_compute.h"
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T, PrecisionType PType>
void CosCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::cos(ctx.GetRawContext(),
                    param.X->template data<T>(),
                    param.Out->template mutable_data<T>(TARGET(kXPU)),
                    param.X->numel());
  CHECK_EQ(r, 0);
}

template <typename T, PrecisionType PType>
void SinCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::sin(ctx.GetRawContext(),
                    param.X->template data<T>(),
                    param.Out->template mutable_data<T>(TARGET(kXPU)),
                    param.X->numel());
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;

using cos_FP32 = xpu::CosCompute<float, PRECISION(kFloat)>;
using cos_FP16 = xpu::CosCompute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(cos, kXPU, kFloat, kNCHW, cos_FP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(cos, kXPU, kFP16, kNCHW, cos_FP16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

using sin_FP32 = xpu::SinCompute<float, PRECISION(kFloat)>;
using sin_FP16 = xpu::SinCompute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(sin, kXPU, kFloat, kNCHW, sin_FP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(sin, kXPU, kFP16, kNCHW, sin_FP16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
