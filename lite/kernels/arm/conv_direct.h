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

#pragma once

#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/context.h"
#include "lite/core/kernel.h"
#include "lite/core/target_wrapper.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

#define ROUNDUP(a, b) ((((a) + (b)-1) / (b)) * (b))

template <PrecisionType Ptype, PrecisionType OutType>
inline bool direct_conv_trans_weights(
    const Tensor* win,
    Tensor* wout,
    const Tensor* bin,
    Tensor* bout,
    int stride,
    const std::vector<float>& w_scale,
    float in_scale,
    float out_scale,
    std::vector<float>& merge_scale,          // NOLINT
    operators::ActivationParam& act_param) {  // NOLINT
  constexpr int cblock = 4;
  int oc = win->dims()[0];
  int ic = win->dims()[1];
  int kh = win->dims()[2];
  int kw = win->dims()[3];
  int cround = ROUNDUP(oc, cblock);
  wout->Resize({cround, ic, kh, kw});
  auto w_in_data = win->data<float>();
  auto transed_w_data = wout->mutable_data<float>();
  if (ic == 3 && stride == 2 && (oc % 4 == 0)) {
    // [chout, 3, kh, kw] -> [chout / cblock, kh, kw, 3, cblock]
    lite::arm::math::conv_trans_weights_c4toc12(
        w_in_data, transed_w_data, oc, ic, cblock, kh * kw);
  } else {
    // [chout, chin, kh, kw] -> [chout / n, chin, kh, kw, n]
    lite::arm::math::conv_trans_weights_numc(
        w_in_data, transed_w_data, oc, ic, cblock, kh * kw);
  }
  return false;
}

template <>
inline bool direct_conv_trans_weights<PRECISION(kInt8), PRECISION(kFloat)>(
    const Tensor* win,
    Tensor* wout,
    const Tensor* bin,
    Tensor* bout,
    int stride,
    const std::vector<float>& w_scale,
    float in_scale,
    float out_scale,
    std::vector<float>& merge_scale,          // NOLINT
    operators::ActivationParam& act_param) {  // NOLINT
  CHECK_EQ(stride, 2);
#ifdef __aarch64__
  int cblock = 8;
#else
  int cblock = 4;
#endif
  int oc = win->dims()[0];
  int ic = win->dims()[1];
  int kh = win->dims()[2];
  int kw = win->dims()[3];
  int cround = ROUNDUP(oc, cblock);
  wout->Resize({cround, ic, kh, kw});
  auto w_in_data = win->data<int8_t>();
  auto transed_w_data = wout->mutable_data<int8_t>();
  lite::arm::math::conv_trans_weights_numc(
      w_in_data, transed_w_data, oc, ic, cblock, kh * kw);
  /// update scale
  CHECK(w_scale.size() == 1 || w_scale.size() == oc)
      << "weights scale size must = filter size or = 1";
  merge_scale.resize(oc);
  for (int i = 0; i < oc; ++i) {
    if (w_scale.size() == 1) {
      merge_scale[i] = w_scale[0] * in_scale;
    } else {
      merge_scale[i] = w_scale[i] * in_scale;
    }
  }
  return false;
}

template <>
inline bool direct_conv_trans_weights<PRECISION(kInt8), PRECISION(kInt8)>(
    const Tensor* win,
    Tensor* wout,
    const Tensor* bin,
    Tensor* bout,
    int stride,
    const std::vector<float>& w_scale,
    float in_scale,
    float out_scale,
    std::vector<float>& merge_scale,          // NOLINT
    operators::ActivationParam& act_param) {  // NOLINT
  CHECK_EQ(stride, 2);
#ifdef __aarch64__
  int cblock = 8;
#else
  int cblock = 4;
#endif
  int oc = win->dims()[0];
  int ic = win->dims()[1];
  int kh = win->dims()[2];
  int kw = win->dims()[3];
  int cround = ROUNDUP(oc, cblock);
  wout->Resize({cround, ic, kh, kw});
  auto w_in_data = win->data<int8_t>();
  auto transed_w_data = wout->mutable_data<int8_t>();
  lite::arm::math::conv_trans_weights_numc(
      w_in_data, transed_w_data, oc, ic, cblock, kh * kw);
  /// update scale
  CHECK(w_scale.size() == 1 || w_scale.size() == oc)
      << "weights scale size must = filter size or = 1";
  merge_scale.resize(oc);
  float scale = in_scale / out_scale;
  for (int i = 0; i < oc; ++i) {
    if (w_scale.size() == 1) {
      merge_scale[i] = w_scale[0] * scale;
    } else {
      merge_scale[i] = w_scale[i] * scale;
    }
  }
  //! update relu6 parameter
  if (act_param.active_type == lite_api::ActivationType::kRelu6) {
    act_param.Relu_clipped_coef = act_param.Relu_clipped_coef / out_scale;
  }
  //! update hardswish parameter
  if (act_param.active_type == lite_api::ActivationType::kHardSwish) {
    act_param.hard_swish_scale = act_param.hard_swish_scale / out_scale;
    act_param.hard_swish_offset = act_param.hard_swish_offset / out_scale;
    act_param.hard_swish_threshold = act_param.hard_swish_threshold / out_scale;
  }
  /// update bias
  if (bin) {
    bout->Resize(bin->dims());
    auto ptr = bout->mutable_data<float>();
    auto ptr_in = bin->data<float>();
    for (int i = 0; i < bin->numel(); ++i) {
      ptr[i] = ptr_in[i] / out_scale;
    }
    return true;
  }
  return false;
}

template <PrecisionType Ptype, PrecisionType OutType>
inline std::pair<uint32_t, uint32_t> direct_conv_ptype() {
  return std::make_pair(4, 4);
}
template <>
inline std::pair<uint32_t, uint32_t>
direct_conv_ptype<PRECISION(kInt8), PRECISION(kFloat)>() {
#ifdef __aarch64__
  return std::make_pair(8, 4);
#else
  return std::make_pair(4, 4);
#endif
}
template <>
inline std::pair<uint32_t, uint32_t>
direct_conv_ptype<PRECISION(kInt8), PRECISION(kInt8)>() {
#ifdef __aarch64__
  return std::make_pair(8, 4);
#else
  return std::make_pair(4, 4);
#endif
}
#ifdef ENABLE_ARM_FP16
template <>
inline bool direct_conv_trans_weights<PRECISION(kFP16), PRECISION(kFP16)>(
    const Tensor* win,
    Tensor* wout,
    const Tensor* bin,
    Tensor* bout,
    int stride,
    const std::vector<float>& w_scale,
    float in_scale,
    float out_scale,
    std::vector<float>& merge_scale,          // NOLINT
    operators::ActivationParam& act_param) {  // NOLINT
  constexpr int cblock = 8;
  int oc = win->dims()[0];
  int ic = win->dims()[1];
  int kh = win->dims()[2];
  int kw = win->dims()[3];
  int cround = ROUNDUP(oc, cblock);
  wout->Resize({cround, ic, kh, kw});
  auto w_in_data = win->data<float16_t>();
  auto transed_w_data = wout->mutable_data<float16_t>();
  if (ic == 3 && stride == 2 && (oc % 8 == 0)) {
    // [chout, 3, kh, kw] -> [chout / cblock, kh, kw, 3, cblock]
    lite::arm::math::conv_trans_weights_c4toc12(
        w_in_data, transed_w_data, oc, ic, cblock, kh * kw);
  } else {
    // [chout, chin, kh, kw] -> [chout / n, chin, kh, kw, n]
    lite::arm::math::conv_trans_weights_numc(
        w_in_data, transed_w_data, oc, ic, cblock, kh * kw);
  }
  return false;
}
template <>
inline std::pair<uint32_t, uint32_t>
direct_conv_ptype<PRECISION(kFP16), PRECISION(kFP16)>() {
#ifdef __aarch64__
  return std::make_pair(8, 8);
#else
  return std::make_pair(8, 4);
#endif
}
#endif

/// only support 3x3s1 and 3x3s2
template <PrecisionType Ptype, PrecisionType OutType>
class DirectConv : public KernelLite<TARGET(kARM), Ptype> {
 public:
  DirectConv() = default;
  ~DirectConv() {}

  virtual void PrepareForRun() {
    auto& param = this->template Param<param_t>();
    auto& ctx = this->ctx_->template As<ARMContext>();

    auto x_dims = param.x->dims();
    auto w_dims = param.filter->dims();
    auto o_dims = param.output->dims();

    int ic = x_dims[1];
    int oc = o_dims[1];
    int sw = param.strides[1];
    int kw = w_dims[3];
    int kh = w_dims[2];
    CHECK(sw == 1 || sw == 2)
        << "direct conv only support conv3x3s1 and conv3x3s2";
    CHECK(kw == 3 && kh == 3)
        << "direct conv only support conv3x3s1 and conv3x3s2";

    flag_trans_bias_ =
        direct_conv_trans_weights<Ptype, OutType>(param.filter,
                                                  &weights_,
                                                  param.bias,
                                                  &bias_,
                                                  sw,
                                                  param.weight_scale,
                                                  param.input_scale,
                                                  param.output_scale,
                                                  w_scale_,
                                                  param.activation_param);
  }

  virtual void ReInitWhenNeeded() {
    auto& param = this->template Param<param_t>();
    auto& ctx = this->ctx_->template As<ARMContext>();
    auto dim_in = param.x->dims();
    if (last_shape_ == dim_in) {
      return;
    }

    auto w_dims = param.filter->dims();
    auto dim_out = param.output->dims();
    auto paddings = *param.paddings;
    const int threads = ctx.threads();
    int llc_size = ctx.llc_size() / sizeof(float);
    const int pad_w = paddings[2];
    const int pad_h = paddings[0];
    const int kernel_w = w_dims[3];
    const int stride_w = param.strides[1];
    int ow = dim_out[3];
    int oh = dim_out[2];
    int ic = dim_in[1];
    if (ic == 3) {
      ic = 4;
    }
    auto&& res = direct_conv_ptype<Ptype, OutType>();
    const int OUT_C_BLOCK = res.first;
    const int OUT_W_BLOCK = res.second;
    const int OUT_H_BLOCK = 2;
    const int wout_round = ROUNDUP(ow, OUT_W_BLOCK);
    const int win_round = (wout_round - 1) * stride_w + kernel_w;
    /* get h block */
    /* win_round * ic * hin_r_block + wout_round * OUT_C_BLOCK * hout_r_block */
    /* * threads = llc_size */
    /* win_round = (wout_round - 1) * stride_w + kernel_w */
    /* hin_r_block = (hout_r_block - 1) * stride_w + kernel_w*/
    int a = kernel_w * stride_w;
    int b = kernel_w * kernel_w;
    int c = stride_w * stride_w;
    int hout_r_block =
        (llc_size - ic * (a * (wout_round - 2) + b - c * (wout_round - 1))) /
        ((ic * ((wout_round - 1) * c + a)) +
         wout_round * OUT_C_BLOCK * threads);
    hout_r_block = hout_r_block > oh ? oh : hout_r_block;
    hout_r_block = (hout_r_block / OUT_H_BLOCK) * OUT_H_BLOCK;
    hout_r_block = hout_r_block < OUT_H_BLOCK ? OUT_H_BLOCK : hout_r_block;
    const int hin_r_block = (hout_r_block - 1) * stride_w + kernel_w;
    int in_len = win_round * ic;
    int pre_in_size = hin_r_block * in_len;
    int pre_out_size = OUT_C_BLOCK * hout_r_block * wout_round;

    workspace_size_ = sizeof(float) * (pre_in_size + threads * pre_out_size);
    last_shape_ = dim_in;
  }

  virtual void Run();

#ifdef LITE_WITH_PROFILE
  virtual void SetProfileRuntimeKernelInfo(
      paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
  }
  std::string kernel_func_name_{"NotImplForConvDirect"};
#define PROFILE_INFO(dtype1, dtype2)                                        \
  template <>                                                               \
  void DirectConv<PRECISION(dtype1), PRECISION(dtype2)>::                   \
      SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) { \
    ch->kernel_func_name = kernel_func_name_;                               \
  }

#define KERNEL_FUNC_NAME(kernel_func_name) kernel_func_name_ = kernel_func_name;

#else
#define PROFILE_INFO(dtype1, dtype2)
#define KERNEL_FUNC_NAME(kernel_func_name)
#endif

  /// todo, support inplace weights transform
 protected:
  Tensor weights_;
  Tensor bias_;
  bool flag_trans_weights_{false};
  bool flag_trans_bias_{false};
  std::vector<float> w_scale_;
  DDim last_shape_;
  int workspace_size_{0};

 private:
  using param_t = operators::ConvParam;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
