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

#include "lite/kernels/x86/conv_compute.h"
#include <utility>
#include "lite/kernels/x86/conv_depthwise.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <>
void Conv2dCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  auto& param = this->Param<param_t>();

  const int input_channel = param.x->dims()[1];
  const int output_channel = param.filter->dims()[0];
  const int groups = param.groups;

  const int kernel_h = param.filter->dims()[2];
  const int kernel_w = param.filter->dims()[3];

  const int stride_h = param.strides[0];
  const int stride_w = param.strides[1];
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  bool dw_kernel = (input_channel == groups && output_channel == groups);
  bool pads_equal =
      ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));
  bool pads_all_equal = (pads_equal && paddings[0] == paddings[2]);
  bool ks_equal = (stride_h == stride_w) && (kernel_h == kernel_w);
  bool no_dilation = (dilations[0] == 1) && (dilations[1] == 1);
  bool kps_equal = (pad_h == pad_w) && ks_equal;
  bool flag_dw_3x3 = (kw == 3) && (kh == 3) && (stride == 1 || stride == 2);
  bool flag_dw_5x5 = (kw == 5) && (kh == 5) && (stride == 1 || stride == 2);
  // todo add conv_5x5_depthwise implement
  flag_dw_5x5 = false;
  bool flag_dw = flag_dw_3x3 || flag_dw_5x5;

  /// select conv impl
  if (dw_kernel && ks_equal && no_dilation && flag_dw && (groups & 3) == 0) {
    impl_ = new DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>;
  }

  if (impl_) {
    impl_->SetContext(std::move(this->ctx_));
    impl_->SetParam(param);
    impl_->PrepareForRun();
    is_first_epoch_ = false;
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
typedef paddle::lite::kernels::x86::Conv2dCompute<PRECISION(kFloat),
                                                  PRECISION(kFloat)>
    ConvFp32;

REGISTER_LITE_KERNEL(conv2d, kX86, kFloat, kNCHW, ConvFp32, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("SecondInput",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindPaddleOpVersion("conv2d", 1)
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d, kX86, kFloat, kNCHW, ConvFp32, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindPaddleOpVersion("depthwise_conv2d", 1)
    .Finalize();
