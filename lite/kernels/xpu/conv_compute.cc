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

#include "lite/kernels/xpu/conv_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <>
void Conv2dCompute<PRECISION(kFloat)>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto& x_dims = param.x->dims();
  auto& w_dims = param.filter->dims();
  int groups = param.groups;
  auto& strides = param.strides;
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;

  int r = xdnn::conv2d<float, float, float, int16_t>(
      ctx.GetRawContext(), /* context */
      param.x->data<float>(),
      param.filter->data<float>(), /* weight */
      param.output->mutable_data<float>(TARGET(kXPU)),
      x_dims[0], /* num */
      x_dims[1], /* input_c */
      x_dims[2], /* input_h */
      x_dims[3], /* input_w */
      w_dims[0], /* num_filter */
      std::vector<int>{static_cast<int>(w_dims[2]),
                       static_cast<int>(w_dims[3])}, /* kernel size*/
      strides,
      paddings,
      dilations,
      groups,
      nullptr,
      nullptr,
      nullptr,
      true /*is_nchw*/);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;
using Conv2dFp32 = xpu::Conv2dCompute<PRECISION(kFloat)>;

REGISTER_LITE_KERNEL(conv2d, kXPU, kFloat, kNCHW, Conv2dFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindPaddleOpVersion("conv2d", 1)
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d, kXPU, kFloat, kNCHW, Conv2dFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindPaddleOpVersion("depthwise_conv2d", 1)
    .Finalize();
