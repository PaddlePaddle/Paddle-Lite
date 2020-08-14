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

#include "lite/kernels/xpu/conv2d_transpose_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <>
void Conv2dTransposeCompute<PRECISION(kFloat)>::PrepareForRun() {
  maxs_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(8 * sizeof(float), false /* use_l3 */);

  auto& ctx = this->ctx_->As<XPUContext>();
  auto& param = this->Param<param_t>();
  float* max_filter_ptr = reinterpret_cast<float*>(maxs_xpu_guard_->addr_);
  int filter_size = param.filter->numel();
  int r = xdnn::findmax<float>(ctx.GetRawContext(),
                               param.filter->data<float>(),
                               filter_size,
                               max_filter_ptr);
  CHECK_EQ(r, 0);
}

template <>
void Conv2dTransposeCompute<PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto& out_dims = param.output->dims();
  auto& w_dims = param.filter->dims();
  auto& in_dims = param.x->dims();

  int groups = param.groups;
  auto& strides = param.strides;
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;

  float* max_filter_ptr = reinterpret_cast<float*>(maxs_xpu_guard_->addr_);
  float* max_image_ptr = max_filter_ptr + 4;
  int image_size = param.x->numel();

  // find image max
  int r = xdnn::findmax<float>(
      ctx.GetRawContext(), param.x->data<float>(), image_size, max_image_ptr);
  CHECK_EQ(r, 0);

  r = xdnn::conv2d_backward_int16(
      ctx.GetRawContext(),
      out_dims[0],
      out_dims[1],
      out_dims[2],
      out_dims[3],
      in_dims[1],
      w_dims[2],
      w_dims[3],
      strides[0],
      strides[1],
      paddings[0],
      paddings[1],
      dilations[0],
      dilations[1],
      groups,
      param.x->data<float>(),
      param.filter->data<float>(),
      param.output->mutable_data<float>(TARGET(kXPU)),
      max_image_ptr,
      max_filter_ptr);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;
using Conv2dTransposeFp32 = xpu::Conv2dTransposeCompute<PRECISION(kFloat)>;

REGISTER_LITE_KERNEL(
    conv2d_transpose, kXPU, kFloat, kNCHW, Conv2dTransposeFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
