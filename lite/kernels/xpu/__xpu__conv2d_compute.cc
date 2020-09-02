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

#include "lite/kernels/xpu/__xpu__conv2d_compute.h"
#include <string>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUConv2dCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto& input_dims = param.Input->dims();
  auto& filter_dims = param.Filter->dims();
  int batch = static_cast<int>(input_dims[0]);
  int img_c = static_cast<int>(input_dims[1]);
  int img_h = static_cast<int>(input_dims[2]);
  int img_w = static_cast<int>(input_dims[3]);
  int filter_num = static_cast<int>(filter_dims[0]);
  int win_h = static_cast<int>(filter_dims[2]);
  int win_w = static_cast<int>(filter_dims[3]);

  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  int paddings_h = paddings[0];
  int paddings_w = paddings[1];
  int dilations_h = dilations[0];
  int dilations_w = dilations[1];

  std::string filter_type = param.filter_type;
  int groups = param.groups;

  int act_type = (param.act_type == -1) ? xdnn::Activation_t::RELU
                                        : param.act_type;  // -1 means not init
  const auto* bias = param.Bias ? param.Bias->data<float>() : nullptr;
  const auto* branch = param.Branch ? param.Branch->data<float>() : nullptr;
  const float* input_max =
      param.InputMax ? param.InputMax->data<float>() : nullptr;
  float* output_max = param.OutputMax
                          ? param.OutputMax->mutable_data<float>(TARGET(kXPU))
                          : nullptr;
  float* output = param.Output->mutable_data<float>(TARGET(kXPU));

  // TODO(luohang): now support for resnet50 first
  CHECK_EQ(act_type, xdnn::Activation_t::RELU);
  CHECK_EQ(groups, 1);
  CHECK_EQ(filter_type, "int16");

  xdnn::Activation_t act((xdnn::Activation_t::act_enum)act_type);
  int r = xdnn::conv2d_forward_int16<float, int16_t, float, float>(
      ctx.GetRawContext(),            /* context */
      batch,                          /* batch */
      img_c,                          /* input_c */
      img_h,                          /* input_h */
      img_w,                          /* input_w */
      filter_num,                     /* num_filter */
      win_h,                          /* kernel_h */
      win_w,                          /* kernel_w */
      stride_h,                       /* stride_h */
      stride_w,                       /* stride_w */
      paddings_h,                     /* pad_h */
      paddings_w,                     /* pad_w */
      dilations_h,                    /* dilation_h */
      dilations_w,                    /* dilation_w */
      groups,                         /* group */
      param.Input->data<float>(),     /* input bottom */
      param.Filter->data<int16_t>(),  /* filter weight */
      output,                         /* output top */
      bias,                           /* bias */
      branch,                         /* branch */
      act,                            /* act type */
      input_max,                      /* max_image_ptr */
      param.FilterMax->data<float>(), /* max_filter_ptr */
      output_max /* max_result_ptr */);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__conv2d,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUConv2dCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FilterMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Branch", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
