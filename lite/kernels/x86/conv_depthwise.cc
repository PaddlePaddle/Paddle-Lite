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
#include "lite/backends/x86/math/conv_depthwise_int8.h"
#include "lite/backends/x86/math/conv_depthwise_pack4.h"
#include "lite/backends/x86/math/conv_depthwise_pack8.h"
#include "lite/backends/x86/math/conv_utils.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <>
void DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {}

template <>
void DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);

  auto input_dims = param.x->dims();
  CHECK_EQ(input_dims.size(), 4UL);
  int batch_size = param.x->dims()[0];
  int input_channel = param.x->dims()[1];

  const int pack_size =
      input_channel % 8 == 0 ? 8 : input_channel % 4 == 0 ? 4 : 1;
  const int pack_num = input_channel / pack_size;

  if (pack_size == 8) {
    // lite::x86::math::pack8_m256(param.x, &input_pack_, pack_num, false);
    // lite::x86::math::padding8_m256(
    //    &input_pack_, &input_padding_, *(param.paddings));
    lite::x86::math::pack_padding8_m256(
        param.x, &input_padding_, pack_num, *(param.paddings));
  } else if (pack_size == 4) {
    lite::x86::math::pack4_m128(param.x, &input_pack_, pack_num, false);
    lite::x86::math::padding4_m128(
        &input_pack_, &input_padding_, *(param.paddings));
  } else {
    lite::x86::math::padding1_float(
        param.x, &input_padding_, *(param.paddings));
  }

  // filter [oc, ic/groups=1, kh, kw]
  auto filter_dims = param.filter->dims();
  CHECK_EQ(filter_dims.size(), 4UL);
  int kernel_h = param.filter->dims()[2];
  int kernel_w = param.filter->dims()[3];

  // filter [oc, 1, ih, iw] & pack_size=8 => [oc/8, ih, iw, 8]
  // filter [oc, 1, ih, iw] & pack_size=4 => [ic/4, ih, iw, 4]
  if (pack_size == 8) {
    lite::x86::math::pack8_m256(param.filter, &filter_pack_, pack_num, true);
  } else if (pack_size == 4) {
    lite::x86::math::pack4_m128(param.filter, &filter_pack_, pack_num, true);
  }

  // attributes
  const int stride_h = param.strides[0];
  const int stride_w = param.strides[1];
  const int dilation_h = (*param.dilations)[0];
  const int dilation_w = (*param.dilations)[1];

  // act type
  auto act_param = param.activation_param;
  bool has_act = act_param.has_active;
  auto act_type = act_param.active_type;

  // output [bs, oc, oh, ow]
  CHECK_EQ(param.output->dims().size(), 4UL);
  const int in_h = input_padding_.dims()[2], in_w = input_padding_.dims()[3];
  const int kernel_extend_h = dilation_h * (kernel_h - 1) + 1;
  const int kernel_extend_w = dilation_w * (kernel_w - 1) + 1;
  int output_height = (in_h - kernel_extend_h) / stride_h + 1;
  int output_width = (in_w - kernel_extend_w) / stride_w + 1;
  // output_trans [bs, oc/8, oh, ow, 8]
  // output_trans [bs, oc/4, oh, ow, 4]
  output_pack_.Resize(
      {batch_size, pack_num, output_height, output_width, pack_size});

  if (pack_size == 8) {
    if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 &&
        dilation_h == 1 && dilation_w == 1) {
      lite::x86::math::conv_depthwise_3x3s1_m256(&input_padding_,
                                                 &output_pack_,
                                                 &filter_pack_,
                                                 param.bias,
                                                 has_act,
                                                 act_type);
      KERNEL_FUNC_NAME("conv_depthwise_3x3s1_m256")
    } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 2 &&
               stride_w == 2 && dilation_h == 1 && dilation_w == 1) {
      lite::x86::math::conv_depthwise_3x3s2_m256(&input_padding_,
                                                 &output_pack_,
                                                 &filter_pack_,
                                                 param.bias,
                                                 has_act,
                                                 act_type);
      KERNEL_FUNC_NAME("conv_depthwise_3x3s2_m256")
    } else {
      lite::x86::math::conv_depthwise_m256(&input_padding_,
                                           &output_pack_,
                                           &filter_pack_,
                                           param.bias,
                                           stride_h,
                                           stride_w,
                                           dilation_h,
                                           dilation_w,
                                           has_act,
                                           act_type);
      KERNEL_FUNC_NAME("conv_depthwise_m256")
    }
  } else if (pack_size == 4) {
    lite::x86::math::conv_depthwise_m128(&input_padding_,
                                         &output_pack_,
                                         &filter_pack_,
                                         param.bias,
                                         stride_h,
                                         stride_w,
                                         dilation_h,
                                         dilation_w,
                                         has_act,
                                         act_type);
    KERNEL_FUNC_NAME("conv_depthwise_m128")
  }

  // [bs, oh, ow, oc] => [bs, oc, oh, ow]
  if (pack_size == 8) {
    lite::x86::math::unpack8_m256(&output_pack_, param.output);
  } else if (pack_size == 4) {
    lite::x86::math::unpack4_m128(&output_pack_, param.output);
  }
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
  o_data, i_data, w_data, b_data, bs, ic, iw, ih, oh, ow, pad_h, pad_w, \
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
