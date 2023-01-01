// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/deformable_conv_compute.h"

#include <vector>

#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/kernels/host/deformable_conv_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename TGEMM,
          typename TW,
          typename TX,
          typename TY,
          PrecisionType PType>
void DeformableConvCompute<TGEMM, TW, TX, TY, PType>::Run() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  const auto& param = this->template Param<param_t>();

  // this implementation only support v2
  // to support v1, you could follow
  // "paddle/fluid/operators/deformable_conv_v1_op.h"
  const auto* input = param.x->template data<TX>();
  const auto* mask = param.mask->template data<float>();
  const auto* offset = param.offset->template data<float>();
  auto* filter = param.conv_param.filter->template data<TW>();

  auto* out = param.output;
  const int groups = param.conv_param.groups;
  const int deformable_groups = param.deformable_groups;
  const std::vector<int>& strides = param.conv_param.strides;
  const std::vector<int>& paddings = *param.conv_param.paddings;
  const std::vector<int>& dilations = *param.conv_param.dilations;
  std::vector<int> ksize = {
      static_cast<int>(param.conv_param.filter->dims()[2]),
      static_cast<int>(param.conv_param.filter->dims()[3])};
  int f = static_cast<int>(param.conv_param.filter->dims()[0]);
  int n = static_cast<int>(param.x->dims()[0]);
  int c = static_cast<int>(param.x->dims()[1]);
  int h = static_cast<int>(param.x->dims()[2]);
  int w = static_cast<int>(param.x->dims()[3]);
  int r = xdnn::deformable_conv<TX, TW, TY, TGEMM>(
      ctx.GetRawContext(),
      input,
      filter,
      offset,
      mask,
      out->template mutable_data<TY>(TARGET(kXPU)),
      n,
      c,
      h,
      w,
      f,
      ksize,
      strides,
      paddings,
      dilations,
      groups,
      deformable_groups,
      nullptr,
      nullptr,
      nullptr,
      true);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;

using DeformableConvFp32 =
    xpu::DeformableConvCompute<int, float, float, float, PRECISION(kFloat)>;

using DeformableConvFp16 = xpu::
    DeformableConvCompute<int16_t, float, float16, float16, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(
    deformable_conv, kXPU, kFloat, kNCHW, DeformableConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Offset", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    deformable_conv, kXPU, kFP16, kNCHW, DeformableConvFp16, fp16)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Offset", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
