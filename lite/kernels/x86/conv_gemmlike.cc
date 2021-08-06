// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/x86/conv_gemmlike.h"
#include "lite/backends/x86/math/conv_impl.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <>
void GemmLikeConv<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  ReInitWhenNeeded();
}

PROFILE_INFO(kFloat, kFloat);

template <>
void GemmLikeConv<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<X86Context>();
  auto weights = param.filter->data<float>();
  if (flag_trans_weights_) {
    weights = weights_.data<float>();
  }
  const float* bias = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    bias = bias_.data<float>();
  }
  auto din = param.x->data<float>();
  auto dout = param.output->mutable_data<float>();

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int iw = x_dims[3];  // nchw
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int oc = o_dims[1];
  if (flag_1x1gemm_) {
    lite::x86::math::conv1x1s1_gemm(
        din, dout, bs, oc, oh, ow, ic, ih, iw, weights, bias, param, &ctx);
    KERNEL_FUNC_NAME("conv1x1s1_gemm_fp32")
  } else {
    lite::x86::math::conv_im2col_gemm(
        din, dout, bs, oc, oh, ow, ic, ih, iw, weights, bias, param, &ctx);
    KERNEL_FUNC_NAME("conv_im2col_gemm_fp32")
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
