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

#include "lite/kernels/arm/conv_gemmlike.h"
#include <vector>
#include "lite/backends/arm/math/gemm_prepacked_int8.h"
#include "lite/backends/arm/math/packed_sgemm.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <>
void GemmLikeConv<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  ReInitWhenNeeded();
}

template <>
void GemmLikeConv<PRECISION(kInt8), PRECISION(kFloat)>::PrepareForRun() {
  ReInitWhenNeeded();
  auto& param = this->Param<param_t>();
  /// update scale
  w_scale_ = param.weight_scale;
  if (w_scale_.size() != 1 && w_scale_.size() != param.filter->dims()[0]) {
    LOG(FATAL) << "weights scale size must equal to filter size";
    return;
  }
  if (w_scale_.size() == 1) {
    for (int i = 0; i < param.filter->dims()[0] - 1; ++i) {
      w_scale_.push_back(w_scale_[0]);
    }
  }
  float input_scale = param.input_scale;
  for (auto& ws : w_scale_) {
    ws *= input_scale;
  }
}

template <>
void GemmLikeConv<PRECISION(kInt8), PRECISION(kInt8)>::PrepareForRun() {
  ReInitWhenNeeded();
  auto& param = this->Param<param_t>();
  /// update scale
  /// update scale
  w_scale_ = param.weight_scale;
  if (w_scale_.size() != 1 && w_scale_.size() != param.filter->dims()[0]) {
    LOG(FATAL) << "weights scale size must equal to filter size";
    return;
  }
  if (w_scale_.size() == 1) {
    for (int i = 0; i < param.filter->dims()[0] - 1; ++i) {
      w_scale_.push_back(w_scale_[0]);
    }
  }
  float input_scale = param.input_scale;
  float output_scale = param.output_scale;
  for (auto& ws : w_scale_) {
    ws = ws * input_scale / output_scale;
  }
  //!  update bias
  if (param.bias) {
    bias_.Resize(param.bias->dims());
    auto ptr = bias_.mutable_data<float>();
    auto ptr_in = param.bias->data<float>();
    for (int i = 0; i < bias_.numel(); ++i) {
      ptr[i] = ptr_in[i] / param.output_scale;
    }
    flag_trans_bias_ = true;
  }
}

#ifdef LITE_WITH_PROFILE
template <>
void GemmLikeConv<PRECISION(kFloat), PRECISION(kFloat)>::
    SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
  ch->kernel_func_name = kernel_func_name_;
}
#endif

template <>
void GemmLikeConv<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  ctx.ExtendWorkspace(workspace_size_);
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
    lite::arm::math::conv1x1s1_gemm(
        din, dout, bs, oc, oh, ow, ic, ih, iw, weights, bias, param, &ctx);
#ifdef LITE_WITH_PROFILE
    kernel_func_name_ = "conv1x1s1_gemm";
#endif
  } else {
    lite::arm::math::conv_im2col_gemm(
        din, dout, bs, oc, oh, ow, ic, ih, iw, weights, bias, param, &ctx);
#ifdef LITE_WITH_PROFILE
    kernel_func_name_ = "conv_im2col_gemm";
#endif
  }
}

#ifdef LITE_WITH_PROFILE
template <>
void GemmLikeConv<PRECISION(kInt8), PRECISION(kFloat)>::
    SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
  ch->kernel_func_name = kernel_func_name_;
}
#endif

template <>
void GemmLikeConv<PRECISION(kInt8), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  ctx.ExtendWorkspace(workspace_size_);
  auto weights = param.filter->data<int8_t>();
  if (flag_trans_weights_) {
    weights = weights_.data<int8_t>();
  }
  auto bias = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    bias = bias_.data<float>();
  }
  auto din = param.x->data<int8_t>();
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
    lite::arm::math::conv1x1s1_gemm_int8(din,
                                         dout,
                                         bs,
                                         oc,
                                         oh,
                                         ow,
                                         ic,
                                         ih,
                                         iw,
                                         weights,
                                         bias,
                                         param,
                                         &ctx,
                                         w_scale_.data());
#ifdef LITE_WITH_PROFILE
    kernel_func_name_ = "conv1x1s1_gemm_int8";
#endif
  } else {
    lite::arm::math::conv_im2col_gemm_int8(din,
                                           dout,
                                           bs,
                                           oc,
                                           oh,
                                           ow,
                                           ic,
                                           ih,
                                           iw,
                                           weights,
                                           bias,
                                           param,
                                           &ctx,
                                           w_scale_.data());
#ifdef LITE_WITH_PROFILE
    kernel_func_name_ = "conv_im2col_gemm_int8";
#endif
  }
}

#ifdef LITE_WITH_PROFILE
template <>
void GemmLikeConv<PRECISION(kInt8), PRECISION(kInt8)>::
    SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
  ch->kernel_func_name = kernel_func_name_;
}
#endif

template <>
void GemmLikeConv<PRECISION(kInt8), PRECISION(kInt8)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  ctx.ExtendWorkspace(workspace_size_);
  auto weights = param.filter->data<int8_t>();
  if (flag_trans_weights_) {
    weights = weights_.data<int8_t>();
  }
  auto bias = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    bias = bias_.data<float>();
  }
  auto din = param.x->data<int8_t>();
  auto dout = param.output->mutable_data<int8_t>();

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
    lite::arm::math::conv1x1s1_gemm_int8(din,
                                         dout,
                                         bs,
                                         oc,
                                         oh,
                                         ow,
                                         ic,
                                         ih,
                                         iw,
                                         weights,
                                         bias,
                                         param,
                                         &ctx,
                                         w_scale_.data());
#ifdef LITE_WITH_PROFILE
    kernel_func_name_ = "conv1x1s1_gemm_int8";
#endif
  } else {
    lite::arm::math::conv_im2col_gemm_int8(din,
                                           dout,
                                           bs,
                                           oc,
                                           oh,
                                           ow,
                                           ic,
                                           ih,
                                           iw,
                                           weights,
                                           bias,
                                           param,
                                           &ctx,
                                           w_scale_.data());
#ifdef LITE_WITH_PROFILE
    kernel_func_name_ = "conv_im2col_gemm_int8";
#endif
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
