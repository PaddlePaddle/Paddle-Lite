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
void Conv2dCompute<float>::PrepareForRun() {
#ifdef LITE_WITH_AVX
  auto& param = this->Param<param_t>();

  const int input_channel = param.x->dims()[1];
  const int output_channel = param.filter->dims()[0];
  const int groups = param.groups;

  const int kernel_h = param.filter->dims()[2];
  const int kernel_w = param.filter->dims()[3];

  const int stride_h = param.strides[0];
  const int stride_w = param.strides[1];

  if (input_channel == groups && output_channel == groups &&
      (groups & 3) == 0) {
    if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) {
      impl_ = new DepthwiseConv<float>;
      VLOG(3) << "invoking conv_depthwise_3x3s1";
    } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 2 &&
               stride_w == 2) {
      impl_ = new DepthwiseConv<float>;
      VLOG(3) << "invoking conv_depthwise_3x3s2";
    }
  }

  if (impl_) {
    impl_->SetContext(std::move(this->ctx_));
    impl_->SetParam(param);
    impl_->PrepareForRun();
    is_first_epoch_ = false;
  }
#endif
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(conv2d,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::Conv2dCompute<float>,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("SecondInput", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindPaddleOpVersion("conv2d", 1)
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::Conv2dCompute<float>,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindPaddleOpVersion("depthwise_conv2d", 1)
    .Finalize();
