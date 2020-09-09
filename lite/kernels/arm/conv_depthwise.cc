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

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <>
void DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto w_dims = param.filter->dims();
  auto kw = w_dims[3];
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
#ifdef LITE_WITH_PROFILE
    kernel_func_name_ = "conv_depthwise_3x3_fp32";
#endif
  } else if (kw == 5) {
    auto strides = param.strides;
    if ((strides[0] == 1 && strides[1] == 1) ||
        (strides[0] == 2 && strides[1] == 2)) {
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
      impl_ = lite::arm::math::conv_depthwise_5x5_fp32;
#ifdef LITE_WITH_PROFILE
      kernel_func_name_ = "conv_depthwise_5x5_fp32";
#endif
    } else {
      LOG(FATAL)
          << "5x5 depthwise conv only support stride == 1 or stride == 2";
    }
  } else {
    LOG(FATAL) << "this type dw conv not impl";
  }
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

  auto paddings = *param.paddings;
  auto strides = param.strides;
  auto x_dims = param.x->dims();
  int iw = x_dims[3];
  int ih = x_dims[2];
  auto act_param = param.activation_param;
  bool has_act = act_param.has_active;
  lite_api::ActivationType act_type = act_param.active_type;
  // no activation and relu activation is supported now
  bool support_act_type =
      (has_act == false) ||
      (has_act == true && act_type == lite_api::ActivationType::kRelu);
  bool support_pad_type =
      (paddings[0] == paddings[1]) && (paddings[2] == paddings[3]) &&
      (paddings[0] == paddings[2]) && (paddings[0] == 0 || paddings[0] == 1);
  bool support_stride_type = (strides[0] == 1 && strides[1] == 1);
  bool support_width_type = iw > 9 ? true : false;
  /// select dw conv kernel
  if (kw == 3) {
    // trans weights
    impl_ = lite::arm::math::conv_depthwise_3x3_int8_fp32;
#ifdef LITE_WITH_PROFILE
    kernel_func_name_ = "conv_depthwise_3x3_int8_fp32";
#endif
    if (!support_act_type || !support_pad_type || !support_stride_type ||
        !support_width_type) {
      int cround = ROUNDUP(w_dims[0], 8);
      weights_.Resize({cround / 8, 1, kh * kw, 8});
      auto wptr = param.filter->data<int8_t>();
      auto wptr_new = weights_.mutable_data<int8_t>();
      lite::arm::math::conv_trans_weights_numc(wptr, wptr_new, oc, 1, 8, 9);
      flag_trans_weights_ = true;
    } else {
      flag_trans_weights_ = false;
    }
  } else if (kw == 5) {
    // trans weights
    impl_ = lite::arm::math::conv_depthwise_5x5_int8_fp32;
#ifdef LITE_WITH_PROFILE
    kernel_func_name_ = "conv_depthwise_5x5_int8_fp32";
#endif
    int cround = ROUNDUP(w_dims[0], 8);
    weights_.Resize({cround / 8, 1, kh * kw, 8});
    auto wptr = param.filter->data<int8_t>();
    auto wptr_new = weights_.mutable_data<int8_t>();
    lite::arm::math::conv_trans_weights_numc(wptr, wptr_new, oc, 1, 8, 25);
    flag_trans_weights_ = true;
  } else {
    LOG(FATAL) << "this type dw conv not impl";
  }
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
  if (param.activation_param.has_active &&
      param.activation_param.active_type == lite_api::ActivationType::kRelu6) {
    param.activation_param.Relu_clipped_coef =
        param.activation_param.Relu_clipped_coef / param.output_scale;
  }

  auto paddings = *param.paddings;
  auto strides = param.strides;
  auto x_dims = param.x->dims();
  int iw = x_dims[3];
  int ih = x_dims[2];
  auto act_param = param.activation_param;

  bool has_act = act_param.has_active;
  lite_api::ActivationType act_type = act_param.active_type;
  // no activation and relu activation is supported now
  bool support_act_type =
      (has_act == false) ||
      (has_act == true && act_type == lite_api::ActivationType::kRelu);
  bool support_pad_type =
      (paddings[0] == paddings[1]) && (paddings[2] == paddings[3]) &&
      (paddings[0] == paddings[2]) && (paddings[0] == 0 || paddings[0] == 1);
  bool support_stride_type = (strides[0] == 1 && strides[1] == 1);
  bool support_width_type = iw > 9 ? true : false;
  /// select dw conv kernel
  if (kw == 3) {
    // trans weights
    impl_ = lite::arm::math::conv_depthwise_3x3_int8_int8;
#ifdef LITE_WITH_PROFILE
    kernel_func_name_ = "conv_depthwise_3x3_int8_int8";
#endif
    if (!support_act_type || !support_pad_type || !support_stride_type ||
        !support_width_type) {
      int cround = ROUNDUP(w_dims[0], 8);
      weights_.Resize({cround / 8, 1, kh * kw, 8});
      auto wptr = param.filter->data<int8_t>();
      auto wptr_new = weights_.mutable_data<int8_t>();
      lite::arm::math::conv_trans_weights_numc(wptr, wptr_new, oc, 1, 8, 9);
      flag_trans_weights_ = true;
    } else {
      flag_trans_weights_ = false;
    }
  } else if (kw == 5) {
    // trans weights
    impl_ = lite::arm::math::conv_depthwise_5x5_int8_int8;
#ifdef LITE_WITH_PROFILE
    kernel_func_name_ = "conv_depthwise_5x5_int8_int8";
#endif
    int cround = ROUNDUP(w_dims[0], 8);
    weights_.Resize({cround / 8, 1, kh * kw, 8});
    auto wptr = param.filter->data<int8_t>();
    auto wptr_new = weights_.mutable_data<int8_t>();
    lite::arm::math::conv_trans_weights_numc(wptr, wptr_new, oc, 1, 8, 25);
    flag_trans_weights_ = true;
  } else {
    LOG(FATAL) << "this type dw conv not impl";
  }
}

#ifdef LITE_WITH_PROFILE
template <>
void DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>::
    SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
  ch->kernel_func_name = kernel_func_name_;
}
#endif

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

  impl_(i_data,
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
}

#ifdef LITE_WITH_PROFILE
template <>
void DepthwiseConv<PRECISION(kInt8), PRECISION(kFloat)>::
    SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
  ch->kernel_func_name = kernel_func_name_;
}
#endif

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

  impl_(i_data,
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
}

#ifdef LITE_WITH_PROFILE
template <>
void DepthwiseConv<PRECISION(kInt8), PRECISION(kInt8)>::
    SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
  ch->kernel_func_name = kernel_func_name_;
}
#endif

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

  impl_(i_data,
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
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
