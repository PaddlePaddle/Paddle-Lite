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
  auto& param = this->Param<param_t>();

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

void XPUResNet50DtypeCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

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
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int batch_size = param.input->dims()[0];
  int r = xdnn::conv2d_int16_resnet<float, int16_t>(
      ctx.GetRawContext(),                             /* context */
      batch_size,                                      /* num */
      param.input->data<float>(),                      /* bottom */
      &arg_filter_[0],                                 /* weight_list */
      param.output->mutable_data<float>(TARGET(kXPU)), /* top */
      &arg_bias_[0],                                   /* bias_list */
      &arg_max_filter_[0] /* max_filter_list */);
  CHECK_EQ(r, 0);
}

void XPUResNet50DtypeCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int batch_size = param.input->dims()[0];
  int r = xdnn::conv2d_int16_resnet_d<float, int16_t>(
      ctx.GetRawContext(),                             /* context */
      batch_size,                                      /* num */
      param.input->data<float>(),                      /* bottom */
      &arg_filter_[0],                                 /* weight_list */
      param.output->mutable_data<float>(TARGET(kXPU)), /* top */
      &arg_bias_[0],                                   /* bias_list */
      &arg_max_filter_[0] /* max_filter_list */);
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

REGISTER_LITE_KERNEL(__xpu__resnet50_d,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUResNet50DtypeCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("MaxFilter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
