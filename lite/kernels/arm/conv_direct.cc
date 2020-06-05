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

#include "lite/kernels/arm/conv_direct.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

#ifdef LITE_WITH_PROFILE
template <>
void DirectConv<PRECISION(kFloat), PRECISION(kFloat)>::
    SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
  ch->kernel_func_name = kernel_func_name_;
}
#endif

template <>
void DirectConv<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  // extend workspace
  if (param.strides[0] == 2) {
    ctx.ExtendWorkspace(
        lite::arm::math::conv3x3s2_direct_workspace_size(param, &ctx));
  } else {
    ctx.ExtendWorkspace(
        lite::arm::math::conv3x3s1_direct_workspace_size(param, &ctx));
  }

  const auto* i_data = param.x->data<float>();
  const auto* w_data = weights_.data<float>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  auto* o_data = param.output->mutable_data<float>();

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
  if (param.strides[0] == 1) {
    lite::arm::math::conv_3x3s1_direct_fp32(i_data,
                                            o_data,
                                            bs,
                                            oc,
                                            oh,
                                            ow,
                                            ic,
                                            ih,
                                            iw,
                                            w_data,
                                            b_data,
                                            param,
                                            &ctx);
#ifdef LITE_WITH_PROFILE
    kernel_func_name_ = "conv_3x3s1_direct_fp32";
#endif
  } else {
    lite::arm::math::conv_3x3s2_direct_fp32(i_data,
                                            o_data,
                                            bs,
                                            oc,
                                            oh,
                                            ow,
                                            ic,
                                            ih,
                                            iw,
                                            w_data,
                                            b_data,
                                            param,
                                            &ctx);
#ifdef LITE_WITH_PROFILE
    kernel_func_name_ = "conv_3x3s2_direct_fp32";
#endif
  }
}

#ifdef LITE_WITH_PROFILE
template <>
void DirectConv<PRECISION(kInt8), PRECISION(kFloat)>::
    SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
  ch->kernel_func_name = kernel_func_name_;
}
#endif

template <>
void DirectConv<PRECISION(kInt8), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  const auto* i_data = param.x->data<int8_t>();
  const auto* w_data = weights_.data<int8_t>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    b_data = bias_.data<float>();
  }
  auto* o_data = param.output->mutable_data<float>();

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
  if (param.strides[0] == 1) {
    lite::arm::math::conv_3x3s1_direct_int8(i_data,
                                            o_data,
                                            bs,
                                            oc,
                                            oh,
                                            ow,
                                            ic,
                                            ih,
                                            iw,
                                            w_data,
                                            b_data,
                                            param,
                                            &ctx,
                                            w_scale_.data());
#ifdef LITE_WITH_PROFILE
    kernel_func_name_ = "conv_3x3s1_direct_int8";
#endif
  } else {
    lite::arm::math::conv_3x3s2_direct_int8(i_data,
                                            o_data,
                                            bs,
                                            oc,
                                            oh,
                                            ow,
                                            ic,
                                            ih,
                                            iw,
                                            w_data,
                                            b_data,
                                            param,
                                            &ctx,
                                            w_scale_.data());
#ifdef LITE_WITH_PROFILE
    kernel_func_name_ = "conv_3x3s2_direct_int8";
#endif
  }
}

#ifdef LITE_WITH_PROFILE
template <>
void DirectConv<PRECISION(kInt8), PRECISION(kInt8)>::
    SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
  ch->kernel_func_name = kernel_func_name_;
}
#endif

template <>
void DirectConv<PRECISION(kInt8), PRECISION(kInt8)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  const auto* i_data = param.x->data<int8_t>();
  const auto* w_data = weights_.data<int8_t>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    b_data = bias_.data<float>();
  }
  auto* o_data = param.output->mutable_data<int8_t>();

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
  if (param.strides[0] == 1) {
    lite::arm::math::conv_3x3s1_direct_int8(i_data,
                                            o_data,
                                            bs,
                                            oc,
                                            oh,
                                            ow,
                                            ic,
                                            ih,
                                            iw,
                                            w_data,
                                            b_data,
                                            param,
                                            &ctx,
                                            w_scale_.data());
#ifdef LITE_WITH_PROFILE
    kernel_func_name_ = "conv_3x3s1_direct_int8";
#endif
  } else {
    lite::arm::math::conv_3x3s2_direct_int8(i_data,
                                            o_data,
                                            bs,
                                            oc,
                                            oh,
                                            ow,
                                            ic,
                                            ih,
                                            iw,
                                            w_data,
                                            b_data,
                                            param,
                                            &ctx,
                                            w_scale_.data());
#ifdef LITE_WITH_PROFILE
    kernel_func_name_ = "conv_3x3s2_direct_int8";
#endif
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
