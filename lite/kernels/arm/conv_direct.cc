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

PROFILE_INFO(kFloat, kFloat)

template <>
void DirectConv<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param();
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
    KERNEL_FUNC_NAME("conv_3x3s1_direct_fp32")
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
    KERNEL_FUNC_NAME("conv_3x3s2_direct_fp32")
  }
}

PROFILE_INFO(kInt8, kFloat)

template <>
void DirectConv<PRECISION(kInt8), PRECISION(kFloat)>::Run() {
  auto& param = this->Param();
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
    KERNEL_FUNC_NAME("conv_3x3s1_direct_int8")
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
    KERNEL_FUNC_NAME("conv_3x3s2_direct_int8")
  }
}

PROFILE_INFO(kInt8, kInt8)

template <>
void DirectConv<PRECISION(kInt8), PRECISION(kInt8)>::Run() {
  auto& param = this->Param();
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
    KERNEL_FUNC_NAME("conv_3x3s1_direct_int8")
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
    KERNEL_FUNC_NAME("conv_3x3s2_direct_int8")
  }
}

#ifdef ENABLE_ARM_FP16
template <>
void DirectConv<PRECISION(kFP16), PRECISION(kFP16)>::Run() {
  auto& param = this->Param();
  auto& ctx = this->ctx_->template As<ARMContext>();
  // extend workspace
  if (param.strides[0] == 2) {
    ctx.ExtendWorkspace(
        lite::arm::math::fp16::conv3x3s2_direct_workspace_size(param, &ctx));
  } else {
    ctx.ExtendWorkspace(
        lite::arm::math::fp16::conv3x3s1_direct_workspace_size(param, &ctx));
  }
  const auto* i_data = param.x->data<float16_t>();
  const auto* w_data = weights_.data<float16_t>();
  const auto* b_data = param.bias ? param.bias->data<float16_t>() : nullptr;
  auto* o_data = param.output->mutable_data<float16_t>();

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
  if (param.strides[0] == 2) {
    lite::arm::math::fp16::conv_3x3s2_direct_fp16(i_data,
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
    KERNEL_FUNC_NAME("conv_3x3s2_direct_fp16")
  } else {
    lite::arm::math::fp16::conv_3x3s1_direct_fp16(i_data,
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
    KERNEL_FUNC_NAME("conv_3x3s1_direct_fp16")
  }
}
PROFILE_INFO(kFP16, kFP16)
#endif
}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
