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

#include "lite/kernels/xpu/__xpu__resnet_cbam_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUResNetCbamCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  for (auto* filter : param.filter) {
    arg_filter_.push_back(
        reinterpret_cast<const int16_t*>(filter->data<float>()));
  }
  for (auto* bias : param.bias) {
    if (bias == nullptr) {
      arg_bias_.push_back(nullptr);
    } else {
      arg_bias_.push_back(bias->data<float>());
    }
  }
  for (auto* max_filter : param.max_filter) {
    arg_max_filter_.push_back(max_filter->data<float>());
  }
}

void XPUResNetCbamCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto input_dims = param.input->dims();
  int batch_size = input_dims[0];
  int height = input_dims[2];
  int width = input_dims[3];

  int r = xdnn::conv2d_int16_resnet_cbam<float, int16_t>(
      ctx.GetRawContext(),                             /* context */
      batch_size,                                      /* num */
      height,                                          /* height */
      width,                                           /* width */
      param.input->data<float>(),                      /* bottom */
      &arg_filter_[0],                                 /* weight_list */
      param.output->mutable_data<float>(TARGET(kXPU)), /* top */
      &arg_bias_[0],                                   /* bias_list */
      &arg_max_filter_[0],                             /* max_filter_list */
      param.pool_p,                                    /* pool_p */
      true,                                            /* midtype_fp16 */
      false /* dynamic_shape */);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__resnet_cbam,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUResNetCbamCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("MaxFilter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
