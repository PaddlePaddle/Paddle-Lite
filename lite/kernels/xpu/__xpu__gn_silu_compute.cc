// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/__xpu__gn_silu_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename InType, PrecisionType PType>
void XPUGnSiluCompute<InType, PType>::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  // prepare gn_scale
  for (auto* gn_scale : param.gn_scale) {
    arg_gn_scale_.push_back(gn_scale->template data<float>());
  }
  // prepare gn_bias
  for (auto* gn_bias : param.gn_bias) {
    arg_gn_bias_.push_back(gn_bias->template data<float>());
  }
}

template <typename InType, PrecisionType PType>
void XPUGnSiluCompute<InType, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  const InType* in = param.input->template data<InType>();
  InType* out = param.output->template mutable_data<InType>(TARGET(kXPU));
  int n = static_cast<int>(param.input->dims()[0]);
  int c = static_cast<int>(param.input->dims()[1]);
  int h = static_cast<int>(param.input->dims()[2]);
  int w = static_cast<int>(param.input->dims()[3]);
  int groups = param.groups;
  float eps = param.epsilon;
  int r = xdnn::group_norm_silu_fusion<InType>(ctx.GetRawContext(),
                                               in,
                                               out,
                                               n,
                                               c,
                                               h,
                                               w,
                                               groups,
                                               eps,
                                               arg_gn_scale_[0],
                                               arg_gn_bias_[0],
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

using GroupNormSilu_FP32 = xpu::XPUGnSiluCompute<float, PRECISION(kFloat)>;
using GroupNormSilu_FP16 = xpu::XPUGnSiluCompute<float16, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(
    __xpu__gn_silu, kXPU, kFloat, kNCHW, GroupNormSilu_FP32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("GNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("GNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(
    __xpu__gn_silu, kXPU, kFP16, kNCHW, GroupNormSilu_FP16, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("GNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("GNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
