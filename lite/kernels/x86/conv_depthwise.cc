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

#include "lite/kernels/x86/conv_depthwise.h"
#include "lite/backends/x86/math/avx/conv_depthwise_pack4.h"
#include "lite/backends/x86/math/avx/conv_depthwise_pack8.h"
#include "lite/backends/x86/math/avx/conv_utils.h"
#include "lite/backends/x86/math/conv_depthwise_impl.h"
#include "lite/backends/x86/math/conv_depthwise_int8.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {
#define CONV_DW_PARAM                                                         \
  i_data, o_data, bs, oc, oh, ow, ic, ih, iw, w_data, b_data, pad, flag_bias, \
      act_param

template <>
void DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {}

template <>
void DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);

  auto input_dims = param.x->dims();
  CHECK_EQ(input_dims.size(), 4UL);

  const auto* i_data = param.x->data<float>();
  const auto* w_data = param.filter->data<float>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  auto act_param = param.activation_param;
  const auto stride = param.strides[1];
  auto pad = (*param.paddings)[2];
  bool flag_bias = param.bias != nullptr;
  auto* o_data = param.output->mutable_data<float>();
  auto dilations = *param.dilations;
  bool pad_less = pad < 2;

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int iw = x_dims[3];
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int oc = o_dims[1];
  int kh = w_dims[2];

  if (kh == 3) {
    if ((dilations[0] == 1) && (dilations[1] == 1) && pad_less) {
      if (stride == 1) {
        lite::x86::math::conv_depthwise_3x3s1_p01_direct(CONV_DW_PARAM);
      } else if (stride == 2) {
        lite::x86::math::conv_depthwise_3x3s2_p01_direct(CONV_DW_PARAM);
      }
    } else {
      lite::x86::math::conv_depthwise_3x3_pack(
          param, &input_padding_, &input_pack_, &filter_pack_, &output_pack_);
    }
  } else if (kh == 5) {
    if (stride == 1) {
      lite::x86::math::conv_depthwise_5x5s1(CONV_DW_PARAM);
    } else if (stride == 2) {
      lite::x86::math::conv_depthwise_5x5s2(CONV_DW_PARAM);
    }
  } else {
    LOG(FATAL) << "kw and kh only support 3 or 5";
  }
  KERNEL_FUNC_NAME("conv_depthwise_direct")
}

PROFILE_INFO(kFloat, kFloat)

template <>
void DepthwiseConv<PRECISION(kInt8), PRECISION(kFloat)>::PrepareForRun() {
  auto& param = this->Param<param_t>();

  //! update scale
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

#define CONV_DW_INT8_PARAM                                              \
  o_data, i_data, w_data, b_data, bs, ic, ih, iw, oh, ow, pad_h, pad_w, \
      flag_act, alpha, w_scale_.data(), &ctx
template <>
void DepthwiseConv<PRECISION(kInt8), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);
  auto& ctx = this->ctx_->template As<X86Context>();
  const auto* i_data = param.x->data<int8_t>();
  const auto* w_data = param.filter->data<int8_t>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    b_data = bias_.data<float>();
  }
  auto* o_data = param.output->mutable_data<float>();

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int iw = x_dims[3];
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oh = o_dims[2];
  int ow = o_dims[3];
  auto padding = *param.paddings;
  int pad_h = padding[0];
  int pad_w = padding[2];

  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  float alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      alpha = act_param.Relu_clipped_coef;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      alpha = act_param.Leaky_relu_alpha;
    }
  }

  if (w_dims[2] == 3 && param.strides[0] == 1) {
    lite::x86::math::conv_3x3s1_dw_int8(CONV_DW_INT8_PARAM);
  } else if (w_dims[2] == 3 && param.strides[0] == 2) {
    if (padding[0] == 0) {
      lite::x86::math::conv_3x3s2p0_dw_int8(CONV_DW_INT8_PARAM);
    } else if (padding[0] == 1) {
      lite::x86::math::conv_3x3s2p1_dw_int8(CONV_DW_INT8_PARAM);
    } else {
      LOG(FATAL) << "X86 doesn't support paddings >= 2, now padding: "
                 << padding[0];
    }
  } else {
    LOG(FATAL) << "X86 doesn't support other depthwise, now kernel: "
               << w_dims[2] << ", and "
               << "strides: " << param.strides[0];
  }
}

PROFILE_INFO(kInt8, kFloat)

template <>
void DepthwiseConv<PRECISION(kInt8), PRECISION(kInt8)>::PrepareForRun() {
  auto& param = this->Param<param_t>();

  //! update scale
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
  //! update relu6 parameter
  if (param.activation_param.active_type == lite_api::ActivationType::kRelu6) {
    param.activation_param.Relu_clipped_coef =
        param.activation_param.Relu_clipped_coef / param.output_scale;
  }
  //! update leakyRelu parameter
  if (param.activation_param.active_type ==
      lite_api::ActivationType::kLeakyRelu) {
    param.activation_param.Leaky_relu_alpha =
        param.activation_param.Leaky_relu_alpha / param.output_scale;
  }
}

template <>
void DepthwiseConv<PRECISION(kInt8), PRECISION(kInt8)>::Run() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);
  auto& ctx = this->ctx_->template As<X86Context>();
  const auto* i_data = param.x->data<int8_t>();
  const auto* w_data = param.filter->data<int8_t>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    b_data = bias_.data<float>();
  }
  auto* o_data = param.output->mutable_data<int8_t>();

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int iw = x_dims[3];
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oh = o_dims[2];
  int ow = o_dims[3];
  auto padding = *param.paddings;
  int pad_h = padding[0];
  int pad_w = padding[2];

  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  float alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      alpha = act_param.Relu_clipped_coef;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      alpha = act_param.Leaky_relu_alpha;
    }
  }

  if (w_dims[2] == 3 && param.strides[0] == 1) {
    lite::x86::math::conv_3x3s1_dw_int8(CONV_DW_INT8_PARAM);
  } else if (w_dims[2] == 3 && param.strides[0] == 2) {
    if (padding[0] == 0) {
      lite::x86::math::conv_3x3s2p0_dw_int8(CONV_DW_INT8_PARAM);
    } else if (padding[0] == 1) {
      lite::x86::math::conv_3x3s2p1_dw_int8(CONV_DW_INT8_PARAM);
    } else {
      LOG(FATAL) << "X86 doesn't support paddings >= 2, now padding: "
                 << padding[0];
    }
  } else {
    LOG(FATAL) << "X86 doesn't support other depthwise, now kernel: "
               << w_dims[2] << ", and "
               << "strides: " << param.strides[0];
  }
}

PROFILE_INFO(kInt8, kInt8)
#undef CONV_DW_INT8_PARAM
}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
