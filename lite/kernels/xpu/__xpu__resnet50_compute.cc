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

#include "lite/kernels/xpu/__xpu__resnet50_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUResNet50Compute::PrepareForRun() {
  auto& param = this->template Param<param_t>();

  for (auto* filter : param.filter) {
    arg_filter_.push_back(
        reinterpret_cast<const int16_t*>(filter->data<float>()));
  }
  for (auto* bias : param.bias) {
    arg_bias_.push_back(bias->data<float>());
  }
  for (auto* max_filter : param.max_filter) {
    arg_max_filter_.push_back(max_filter->data<float>());
  }
}

void XPUResNet50Compute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int n = param.input->dims()[0];
  int c = param.input->dims()[1];
  int h = param.input->dims()[2];
  int w = param.input->dims()[3];

  int r = xdnn::resnet50<float, int16_t, int16_t, float16>(
      ctx.GetRawContext(),                             /* context */
      param.input->data<float>(),                      /* bottom */
      arg_filter_,                                     /* weight_list */
      param.output->mutable_data<float>(TARGET(kXPU)), /* top */
      n,
      c,
      h,
      w,
      arg_max_x_,
      arg_max_filter_,
      arg_max_y_,
      arg_bias_,
      {nullptr, nullptr, nullptr, nullptr},
      true);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__resnet50,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUResNet50Compute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("MaxFilter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
