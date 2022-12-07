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

#include "lite/kernels/xpu/group_norm_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename InType, PrecisionType PType>
void GroupNormCompute<InType, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  const float* scale =
      param.scale == nullptr ? nullptr : param.scale->template data<float>();
  const float* bias =
      param.bias == nullptr ? nullptr : param.bias->template data<float>();
  float epsilon = param.epsilon;
  int groups = param.groups;

  int n = param.x->dims()[0];
  int c = param.x->dims()[1];
  int height = param.x->dims()[2];
  int width = param.x->dims()[3];

  int r = xdnn::group_norm<InType>(
      ctx.GetRawContext(),                                    /* context */
      param.x->template data<InType>(),                       /* in */
      param.out->template mutable_data<InType>(TARGET(kXPU)), /* out */
      n,                                                      /* n */
      c,                                                      /* c */
      height,                                                 /* h */
      width,                                                  /* w */
      groups,                                                 /* groups */
      epsilon,                                                /* epsilon */
      scale,                                                  /* scale */
      bias,                                                   /* bias */
      nullptr,                                                /* mean */
      nullptr,                                                /* var */
      true);                                                  /* is_nchw */

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;

using GroupNorm_FP32 = xpu::GroupNormCompute<float, PRECISION(kFloat)>;
using GroupNorm_FP16 = xpu::GroupNormCompute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(group_norm, kXPU, kFloat, kNCHW, GroupNorm_FP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Mean", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Variance", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(group_norm, kXPU, kFP16, kNCHW, GroupNorm_FP16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Mean", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Variance", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
