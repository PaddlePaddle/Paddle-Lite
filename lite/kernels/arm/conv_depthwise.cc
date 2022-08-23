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

#include "lite/kernels/arm/conv_depthwise.h"
#include "lite/backends/arm/math/conv_block_utils.h"
#include "lite/backends/arm/math/conv_impl.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/conv_impl_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <>
void DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>::ReInitWhenNeeded() {
  auto& param = this->template Param<param_t>();
  auto x_dims = param.x->dims();
  if (last_shape_ == x_dims) {
    return;
  }
  auto w_dims = param.filter->dims();
  auto kw = w_dims[3];
  auto win = param.x->dims()[3];
  auto paddings = *param.paddings;
  // select dw conv kernel
  if (kw == 3) {
    bool pads_less = ((paddings[1] < 2) && (paddings[3] < 2));
    if (pads_less && paddings[0] == paddings[2] &&
        (paddings[0] == 0 || paddings[0] == 1)) {
      flag_trans_weights_ = false;
    } else {
      // trans weights
      constexpr int cblock = 4;
      auto oc = w_dims[0];
      auto kh = w_dims[2];
      auto cround = ROUNDUP(oc, cblock);
      weights_.Resize({cround, 1, kh, kw});
      auto w_data = weights_.mutable_data<float>();
      auto w_data_in = param.filter->data<float>();
      lite::arm::math::conv_trans_weights_numc(
          w_data_in, w_data, oc, 1, cblock, kh * kw);
      flag_trans_weights_ = true;
    }
    impl_ = lite::arm::math::conv_depthwise_3x3_fp32;
    KERNEL_FUNC_NAME("conv_depthwise_3x3_fp32")
  } else if (kw == 5) {
    auto strides = param.strides;
    bool pads_equal = (paddings[0] == paddings[2]) && (paddings[0] == 2);
    // todo s1 profile is not great than c4
    bool s1_equal =
        0 &&
        (strides[0] == 1 && strides[1] == 1 && pads_equal &&
         static_cast<int>(param.activation_param.active_type) < 4 && win > 8);
    bool s2_equal =
        (strides[0] == 2 && strides[1] == 2 && pads_equal &&
         static_cast<int>(param.activation_param.active_type) < 4 && win > 16);
    if (s1_equal || s2_equal) {
      flag_trans_weights_ = false;
    } else {
      // trans weights
      constexpr int cblock = 4;
      auto oc = w_dims[0];
      auto kh = w_dims[2];
      auto cround = ROUNDUP(oc, cblock);
      weights_.Resize({cround, 1, kh, kw});
      auto w_data = weights_.mutable_data<float>();
      auto w_data_in = param.filter->data<float>();
      lite::arm::math::conv_trans_weights_numc(
          w_data_in, w_data, oc, 1, cblock, kh * kw);
      flag_trans_weights_ = true;
    }
    impl_ = lite::arm::math::conv_depthwise_5x5_fp32;
    KERNEL_FUNC_NAME("conv_depthwise_5x5_fp32")
  } else {
    LOG(FATAL) << "this type dw conv not impl: " << kw;
  }
  last_shape_ = x_dims;
}

template <>
void DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);
  auto& ctx = this->ctx_->template As<ARMContext>();
  // select dw conv kernel
  ReInitWhenNeeded();
  last_shape_ = param.x->dims();
}

template <>
void DepthwiseConv<PRECISION(kInt8), PRECISION(kFloat)>::ReInitWhenNeeded() {
  auto& param = this->template Param<param_t>();
  auto x_dims = param.x->dims();
  if (last_shape_ == x_dims) {
    return;
  }

  auto paddings = *param.paddings;
  auto strides = param.strides;
  int iw = x_dims[3];
  int ih = x_dims[2];
  auto w_dims = param.filter->dims();
  auto kw = w_dims[3];
  auto act_param = param.activation_param;
  bool has_act = act_param.has_active;
  lite_api::ActivationType act_type = act_param.active_type;
  // s1: no activation and relu activation is supported now
  // s2: only support pad=1
  bool support_act_type_s1 =
      (has_act == false) ||
      (has_act == true && (act_type == lite_api::ActivationType::kRelu ||
                           act_type == lite_api::ActivationType::kRelu6));
  bool pads_equal = (paddings[0] == paddings[2]) && (paddings[0] < 2);
  bool support_pad_type_s2 = pads_equal && (paddings[0] == 1);
  bool support_stride_type_s1 = (strides[0] == 1 && strides[1] == 1);
  bool support_stride_type_s2 = (strides[0] == 2 && strides[1] == 2);
  bool support_width_type_s1 = iw > 9 ? true : false;
  bool support_width_type_s2 = iw > 18 ? true : false;
  bool s1_trans =
      (!support_act_type_s1 || !pads_equal || !support_width_type_s1);
  bool s2_trans = (!support_pad_type_s2 || !support_width_type_s2);
  /// select dw conv kernel
  if (kw == 3) {
    // trans weights
    if ((support_stride_type_s1 && s1_trans) ||
        (support_stride_type_s2 && s2_trans)) {
      if (flag_trans_weights_) return;
      int cround = ROUNDUP(w_dims[0], 8);
      auto kh = w_dims[2];
      auto kw = w_dims[3];
      auto oc = w_dims[0];
      weights_.Resize({cround / 8, 1, kh * kw, 8});
      auto wptr = param.filter->data<int8_t>();
      auto wptr_new = weights_.mutable_data<int8_t>();
      lite::arm::math::conv_trans_weights_numc(wptr, wptr_new, oc, 1, 8, 9);
      flag_trans_weights_ = true;
    } else {
      flag_trans_weights_ = false;
    }
  }
  last_shape_ = x_dims;
}

template <>
void DepthwiseConv<PRECISION(kInt8), PRECISION(kFloat)>::PrepareForRun() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto w_dims = param.filter->dims();
  int kh = w_dims[2];
  int kw = w_dims[3];
  int oc = w_dims[0];
  /// update scale
  float in_scale = param.input_scale;
  auto& scale = param.weight_scale;
  CHECK(scale.size() == 1 || scale.size() == oc)
      << "weights scale size must = filter size or = 1";
  w_scale_.resize(oc);
  for (int i = 0; i < oc; ++i) {
    if (scale.size() == 1) {
      w_scale_[i] = scale[0] * in_scale;
    } else {
      w_scale_[i] = scale[i] * in_scale;
    }
  }

  if (kw == 3) {
    ReInitWhenNeeded();
    impl_ = lite::arm::math::conv_depthwise_3x3_int8_fp32;
    KERNEL_FUNC_NAME("conv_depthwise_3x3_int8_fp32")

  } else if (kw == 5) {
    // trans weights
    impl_ = lite::arm::math::conv_depthwise_5x5_int8_fp32;
    KERNEL_FUNC_NAME("conv_depthwise_5x5_int8_fp32")
    int cround = ROUNDUP(w_dims[0], 8);
    weights_.Resize({cround / 8, 1, kh * kw, 8});
    auto wptr = param.filter->data<int8_t>();
    auto wptr_new = weights_.mutable_data<int8_t>();
    lite::arm::math::conv_trans_weights_numc(wptr, wptr_new, oc, 1, 8, 25);
    flag_trans_weights_ = true;
  } else {
    LOG(FATAL) << "this type dw conv not impl";
  }
  last_shape_ = param.x->dims();
}

template <>
void DepthwiseConv<PRECISION(kInt8), PRECISION(kInt8)>::ReInitWhenNeeded() {
  auto& param = this->template Param<param_t>();
  auto x_dims = param.x->dims();
  if (last_shape_ == x_dims) {
    return;
  }

  auto paddings = *param.paddings;
  auto strides = param.strides;
  int iw = x_dims[3];
  int ih = x_dims[2];
  auto w_dims = param.filter->dims();
  auto kw = w_dims[3];
  auto act_param = param.activation_param;
  bool has_act = act_param.has_active;
  lite_api::ActivationType act_type = act_param.active_type;
  // s1: no activation and relu activation is supported now
  // s2: only support pad=1
  bool support_act_type_s1 =
      (has_act == false) ||
      (has_act == true && (act_type == lite_api::ActivationType::kRelu ||
                           act_type == lite_api::ActivationType::kRelu6));
  bool pads_equal = (paddings[0] == paddings[2]) && (paddings[0] < 2);
  bool support_pad_type_s2 = pads_equal && (paddings[0] == 1);
  bool support_stride_type_s1 = (strides[0] == 1 && strides[1] == 1);
  bool support_stride_type_s2 = (strides[0] == 2 && strides[1] == 2);
  bool support_width_type_s1 = iw > 9 ? true : false;
  bool support_width_type_s2 = iw > 18 ? true : false;
  bool s1_trans =
      (!support_act_type_s1 || !pads_equal || !support_width_type_s1);
  bool s2_trans = (!support_pad_type_s2 || !support_width_type_s2);

  /// select dw conv kernel
  if (kw == 3) {
    // trans weights
    if ((support_stride_type_s1 && s1_trans) ||
        (support_stride_type_s2 && s2_trans)) {
      if (flag_trans_weights_) return;
      int cround = ROUNDUP(w_dims[0], 8);
      auto kh = w_dims[2];
      auto kw = w_dims[3];
      auto oc = w_dims[0];
      weights_.Resize({cround / 8, 1, kh * kw, 8});
      auto wptr = param.filter->data<int8_t>();
      auto wptr_new = weights_.mutable_data<int8_t>();
      lite::arm::math::conv_trans_weights_numc(wptr, wptr_new, oc, 1, 8, 9);
      flag_trans_weights_ = true;
    } else {
      flag_trans_weights_ = false;
    }
  }
  last_shape_ = x_dims;
}

template <>
void DepthwiseConv<PRECISION(kInt8), PRECISION(kInt8)>::PrepareForRun() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto w_dims = param.filter->dims();
  int kw = w_dims[3];
  int kh = w_dims[2];
  int oc = w_dims[0];
  /// update scale
  float in_scale = param.input_scale;
  float out_scale = param.output_scale;
  auto& scale = param.weight_scale;
  CHECK(scale.size() == 1 || scale.size() == oc)
      << "weights scale size must = filter size or = 1";
  w_scale_.resize(oc);
  for (int i = 0; i < oc; ++i) {
    if (scale.size() == 1) {
      w_scale_[i] = scale[0] * in_scale / out_scale;
    } else {
      w_scale_[i] = scale[i] * in_scale / out_scale;
    }
  }
  /// update bias
  if (param.bias) {
    bias_.Resize(param.bias->dims());
    auto ptr = bias_.mutable_data<float>();
    auto ptr_in = param.bias->data<float>();
    for (int i = 0; i < bias_.numel(); ++i) {
      ptr[i] = ptr_in[i] / out_scale;
    }
    flag_trans_bias_ = true;
  }
  //! update relu6 parameter
  if (param.activation_param.active_type == lite_api::ActivationType::kRelu6) {
    param.activation_param.Relu_clipped_coef =
        param.activation_param.Relu_clipped_coef / param.output_scale;
  }
  //! update hardswish parameter
  if (param.activation_param.active_type ==
      lite_api::ActivationType::kHardSwish) {
    param.activation_param.hard_swish_offset =
        param.activation_param.hard_swish_offset / param.output_scale;
    param.activation_param.hard_swish_threshold =
        param.activation_param.hard_swish_threshold / param.output_scale;
  }

  if (kw == 3) {
    ReInitWhenNeeded();

    impl_ = lite::arm::math::conv_depthwise_3x3_int8_int8;
    KERNEL_FUNC_NAME("conv_depthwise_3x3_int8_int8")
  } else if (kw == 5) {
    // trans weights
    impl_ = lite::arm::math::conv_depthwise_5x5_int8_int8;
    KERNEL_FUNC_NAME("conv_depthwise_5x5_int8_int8")
    int cround = ROUNDUP(w_dims[0], 8);
    weights_.Resize({cround / 8, 1, kh * kw, 8});
    auto wptr = param.filter->data<int8_t>();
    auto wptr_new = weights_.mutable_data<int8_t>();
    lite::arm::math::conv_trans_weights_numc(wptr, wptr_new, oc, 1, 8, 25);
    flag_trans_weights_ = true;
  } else {
    LOG(FATAL) << "this type dw conv not impl";
  }
  last_shape_ = param.x->dims();
}

PROFILE_INFO(kFloat, kFloat)

#define CONV_DW_PARAM \
  i_data, o_data, bs, oc, oh, ow, ic, ih, iw, w_data, b_data, param, &ctx
template <>
void DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);
  auto& ctx = this->ctx_->template As<ARMContext>();
  const auto* i_data = param.x->data<float>();
  const auto* w_data = flag_trans_weights_ ? weights_.data<float>()
                                           : param.filter->data<float>();
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
  impl_(CONV_DW_PARAM, w_scale_.data());
}

PROFILE_INFO(kInt8, kFloat)

template <>
void DepthwiseConv<PRECISION(kInt8), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);
  auto& ctx = this->ctx_->template As<ARMContext>();
  const auto* i_data = param.x->data<int8_t>();
  const auto* w_data = flag_trans_weights_ ? weights_.data<int8_t>()
                                           : param.filter->data<int8_t>();
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
  int oc = o_dims[1];

  impl_(CONV_DW_PARAM, w_scale_.data());
}

PROFILE_INFO(kInt8, kInt8)

template <>
void DepthwiseConv<PRECISION(kInt8), PRECISION(kInt8)>::Run() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);
  auto& ctx = this->ctx_->template As<ARMContext>();
  const auto* i_data = param.x->data<int8_t>();
  const auto* w_data = flag_trans_weights_ ? weights_.data<int8_t>()
                                           : param.filter->data<int8_t>();
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
  int oc = o_dims[1];

  impl_(CONV_DW_PARAM, w_scale_.data());
}

#ifdef ENABLE_ARM_FP16
template <>
void DepthwiseConv<PRECISION(kFP16), PRECISION(kFP16)>::ReInitWhenNeeded() {}

template <>
void DepthwiseConv<PRECISION(kFP16), PRECISION(kFP16)>::PrepareForRun() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto w_dims = param.filter->dims();
  auto kw = w_dims[3];
  auto channel = w_dims[0];
  auto hin = param.x->dims()[2];
  auto win = param.x->dims()[3];
  auto paddings = *param.paddings;
  if (last_shape_ == param.x->dims()) {
    return;
  }
  if (kw == 3) {
    flag_trans_weights_ = false;
    KERNEL_FUNC_NAME("conv_depthwise_3x3_fp16")
  } else if (kw == 5) {
    auto strides = param.strides;
    if ((strides[0] == 1 && strides[1] == 1) ||
        (strides[0] == 2 && strides[1] == 2)) {
      // trans weights
      constexpr int cblock = 8;
      auto oc = w_dims[0];
      auto kh = w_dims[2];
      auto cround = ROUNDUP(oc, cblock);
      weights_.Resize({cround, 1, kh, kw});
      auto w_data = weights_.mutable_data<float16_t>();
      auto w_data_in = param.filter->data<float16_t>();
      lite::arm::math::conv_trans_weights_numc(
          w_data_in, w_data, oc, 1, cblock, kh * kw);
      flag_trans_weights_ = true;
      KERNEL_FUNC_NAME("conv_depthwise_5x5_fp16")
    } else {
      LOG(FATAL)
          << "5x5 depthwise conv only support stride == 1 or stride == 2";
    }
  } else {
    LOG(FATAL) << "DepthwiseConv FP16 Only Support 3x3 or 5x5!";
  }
}
PROFILE_INFO(kFP16, kFP16)

template <>
void DepthwiseConv<PRECISION(kFP16), PRECISION(kFP16)>::Run() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);
  auto& ctx = this->ctx_->template As<ARMContext>();
  const auto* i_data = param.x->data<float16_t>();
  const auto* w_data = flag_trans_weights_ ? weights_.data<float16_t>()
                                           : param.filter->data<float16_t>();
  const auto* b_data = param.bias ? param.bias->data<float16_t>() : nullptr;
  if (flag_trans_bias_) {
    b_data = bias_.data<float16_t>();
  }
  auto* o_data = param.output->mutable_data<float16_t>();

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();
  int kw = w_dims[3];
  int sw = param.strides[0];

  int iw = x_dims[3];
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int oc = o_dims[1];

  if (kw == 3) {
    lite::arm::math::fp16::conv_depthwise_3x3_fp16(CONV_DW_PARAM);
  } else if (kw == 5) {
    if (sw == 1) {
      lite::arm::math::fp16::conv_depthwise_5x5s1_fp16(CONV_DW_PARAM);
    } else {
      lite::arm::math::fp16::conv_depthwise_5x5s2_fp16(CONV_DW_PARAM);
    }
  }
}

#endif
#undef CONV_DW_PARAM
}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
