// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/__xpu__quick_gelu_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T, PrecisionType PType>
void QuickGeluCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::quick_gelu(ctx.GetRawContext(),
                           param.X->template data<T>(),
                           param.Out->template mutable_data<T>(TARGET(kXPU)),
                           param.X->numel());
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using quick_gelu_FP32 =
    paddle::lite::kernels::xpu::QuickGeluCompute<float, PRECISION(kFloat)>;
using qucik_gelu_FP16 =
    paddle::lite::kernels::xpu::QuickGeluCompute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(
    __xpu__quick_gelu, kXPU, kFloat, kNCHW, quick_gelu_FP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(
    __xpu__quick_gelu, kXPU, kFP16, kNCHW, qucik_gelu_FP16, qucik_gelu_FP16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
