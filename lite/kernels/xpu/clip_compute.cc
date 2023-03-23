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

#include "lite/kernels/xpu/clip_compute.h"
#include <cmath>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename InType, PrecisionType PType>
void ClipCompute<InType, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto min_tensor = param.min_tensor;
  auto max_tensor = param.max_tensor;
  float min = param.min;
  float max = param.max;
  if (min_tensor != nullptr) {
    min = min_tensor->template data<float>()[0];
  }
  if (max_tensor != nullptr) {
    max = max_tensor->template data<float>()[0];
  }
  int r = xdnn::clip_v2<InType>(
      ctx.GetRawContext(),
      param.x->template data<InType>(),
      param.out->template mutable_data<InType>(TARGET(kXPU)),
      param.x->numel(),
      min,
      max);
  CHECK_EQ(r, 0);
}

template <>
void ClipCompute<int8_t, PRECISION(kInt8)>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto min_tensor = param.min_tensor;
  auto max_tensor = param.max_tensor;
  float min = param.min;
  float max = param.max;
  if (min_tensor != nullptr) {
    min = min_tensor->template data<float>()[0];
  }
  if (max_tensor != nullptr) {
    max = max_tensor->template data<float>()[0];
  }

  CHECK_LT(min, param.input_scale);
  int8_t max_quantized =
      std::abs(max) < param.input_scale
          ? static_cast<int8_t>(max / param.input_scale * 127.f)
          : 127;
  int8_t min_quantized =
      std::abs(min) < param.input_scale
          ? static_cast<int8_t>(min / param.input_scale * 127.f)
          : -127;

  int r = xdnn::clip_v2<int8_t>(
      ctx.GetRawContext(),
      param.x->template data<int8_t>(),
      param.out->template mutable_data<int8_t>(TARGET(kXPU)),
      param.x->numel(),
      min_quantized,
      max_quantized);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using clip_fp32 =
    paddle::lite::kernels::xpu::ClipCompute<float, PRECISION(kFloat)>;
using clip_int8 =
    paddle::lite::kernels::xpu::ClipCompute<int8_t, PRECISION(kInt8)>;

REGISTER_LITE_KERNEL(clip, kXPU, kFloat, kNCHW, clip_fp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Min", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Max", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(clip, kXPU, kInt8, kNCHW, clip_int8, clip_INT8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindInput("Min", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Max", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .Finalize();
