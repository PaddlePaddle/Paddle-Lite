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

#include "lite/kernels/xpu/layer_norm_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename InType, PrecisionType PType>
void LayerNormCompute<InType, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto x_dims = param.X->dims();
  auto axis = param.begin_norm_axis;
  auto matrix_dim = x_dims.Flatten2D(axis);
  float epsilon = param.epsilon;

  int r = xdnn::layer_norm<InType>(
      ctx.GetRawContext(),                                  /* context */
      param.X->template data<InType>(),                     /* in */
      param.Y->template mutable_data<InType>(TARGET(kXPU)), /* out */
      matrix_dim[0],                                        /* m */
      matrix_dim[1],                                        /* n */
      epsilon,                                              /* epsilon */
      param.Scale->template data<float>(),                  /* scale */
      param.Bias->template data<float>(),                   /* bias */
      nullptr,
      nullptr);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;

using LayerNorm_FP32 = xpu::LayerNormCompute<float, PRECISION(kFloat)>;
using LayerNorm_FP16 = xpu::LayerNormCompute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(layer_norm, kXPU, kFloat, kNCHW, LayerNorm_FP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Mean", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Variance", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    layer_norm, kXPU, kFP16, kNCHW, LayerNorm_FP16, DISABLE_XPU1_fp16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Mean", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Variance",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
