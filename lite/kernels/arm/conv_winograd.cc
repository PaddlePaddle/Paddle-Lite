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

#include "lite/kernels/arm/conv_winograd.h"
#include "lite/backends/arm/math/conv_impl.h"
#include "lite/backends/arm/math/packed_sgemm.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

#define WINOGRAD_INIT(cblock)                        \
  auto& param = this->Param<param_t>();              \
  auto& ctx = this->ctx_->template As<ARMContext>(); \
  int threads = ctx.threads();                       \
  auto x_dims = param.x->dims();                     \
  auto w_dims = param.filter->dims();                \
  auto o_dims = param.output->dims();                \
  if (last_shape_ == x_dims) {                       \
    return;                                          \
  }                                                  \
  last_shape_ = x_dims;                              \
  /* update workspace size */                        \
  int ic = x_dims[1];                                \
  int ih = x_dims[2];                                \
  int iw = x_dims[3];                                \
  int oc = o_dims[1];                                \
  int oh = o_dims[2];                                \
  int ow = o_dims[3];                                \
  int tile_block = 8;                                \
  auto pad = *(param.paddings);                      \
  int pad_h0 = pad[0];                               \
  int pad_h1 = pad[1];                               \
  int pad_w0 = pad[2];                               \
  int pad_w1 = pad[3];                               \
  int oc_pad = (oc + cblock - 1) / cblock * cblock;  \
  int ic_pad = (ic + cblock - 1) / cblock * cblock;

#define FUNCS_PARAM i_data, o_data, bs, oc, oh, ow, ic, ih, iw, w_data, b_data
template <>
void WinogradConv<PRECISION(kFloat), PRECISION(kFloat)>::ReInitWhenNeeded() {
  WINOGRAD_INIT(4)

  // select best wino_unit
  int wino_unit = ow * oh / (tile_block * threads);
  if (wino_unit < 16) {
    wino_iw = 4;
    if (last_function_ == 0) {
      return;
    }
    last_function_ = 0;
  } else if (wino_unit < 36) {
    wino_iw = 6;
    if (last_function_ == 1) {
      return;
    }
    last_function_ = 1;
  } else {
    wino_iw = 8;
    if (last_function_ == 2) {
      return;
    }
    last_function_ = 2;
  }
  last_function_ = -1;

  const int new_input_size =
      ic_pad * (ih + pad_h0 + pad_h1) * (iw + pad_w0 + pad_w1);
  const int temp_size = (tile_block * (oc_pad + ic_pad) * wino_iw * wino_iw +
                         8 * wino_iw * wino_iw) *
                        threads;

  workspace_size_ = (temp_size + new_input_size) * sizeof(float);

  //! update trans weights impl
  weights_.Resize({1, 1, 1, wino_iw * wino_iw * oc_pad * ic_pad});
  void* trans_tmp_ptr = malloc(sizeof(float) * wino_iw * wino_iw * oc * ic);
  auto weights_data_ = weights_.mutable_data<float>();
  memset(reinterpret_cast<char*>(weights_data_),
         0,
         weights_.numel() * sizeof(float));
  switch (wino_iw) {
    case 8:
      lite::arm::math::weight_trans_c4_8x8(
          weights_data_, param.filter->data<float>(), ic, oc, trans_tmp_ptr);
      break;
    case 6:
      lite::arm::math::weight_trans_c4_6x6(
          weights_data_, param.filter->data<float>(), ic, oc, trans_tmp_ptr);
      break;
    case 4:
      lite::arm::math::weight_trans_c4_4x4(
          weights_data_, param.filter->data<float>(), ic, oc, trans_tmp_ptr);
      break;
    default:
      lite::arm::math::weight_trans_c4_8x8(
          weights_data_, param.filter->data<float>(), ic, oc, trans_tmp_ptr);
  }

  free(trans_tmp_ptr);
}

template <>
void WinogradConv<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  ReInitWhenNeeded();
}

PROFILE_INFO(kFloat, kFloat)

template <>
void WinogradConv<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  ctx.ExtendWorkspace(workspace_size_);
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

  if (wino_iw == 8) {
    lite::arm::math::conv_compute_6x6_3x3(FUNCS_PARAM, param, &ctx);
    KERNEL_FUNC_NAME("conv_compute_6x6_3x3")
  } else if (wino_iw == 6) {
    lite::arm::math::conv_compute_4x4_3x3(FUNCS_PARAM, param, &ctx);
    KERNEL_FUNC_NAME("conv_compute_4x4_3x3")
  } else {
    int tile_block = 8;
    int block_count =
        (((ow + 1) / 2) * ((oh + 1) / 2) + tile_block - 1) / tile_block;
    if (block_count != 1) {
      lite::arm::math::conv_compute_2x2_3x3(FUNCS_PARAM, param, &ctx);
      KERNEL_FUNC_NAME("conv_compute_2x2_3x3")
    } else {
      lite::arm::math::conv_compute_2x2_3x3_small(FUNCS_PARAM, param, &ctx);
      KERNEL_FUNC_NAME("conv_compute_2x2_3x3_small")
    }
  }
}

template <PrecisionType OutType>
void WinogradConv<PRECISION(kInt8), OutType>::ReInitWhenNeeded() {
  WINOGRAD_INIT(8)
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
  if (param.bias) {
    bias_.Resize(param.bias->dims());
    auto ptr = bias_.mutable_data<float>();
    auto ptr_in = param.bias->template data<float>();
    for (int i = 0; i < bias_.numel(); ++i) {
      ptr[i] = ptr_in[i];
    }
  }
  if (OutType == PRECISION(kInt8)) {
    float output_scale = param.output_scale;
    //! update relu6 parameter
    if (param.activation_param.active_type ==
        lite_api::ActivationType::kRelu6) {
      param.activation_param.Relu_clipped_coef =
          param.activation_param.Relu_clipped_coef / output_scale;
    }
    //! update hardswish parameter
    if (param.activation_param.active_type ==
        lite_api::ActivationType::kHardSwish) {
      param.activation_param.hard_swish_scale =
          param.activation_param.hard_swish_scale / param.output_scale;
      param.activation_param.hard_swish_offset =
          param.activation_param.hard_swish_offset / output_scale;
      param.activation_param.hard_swish_threshold =
          param.activation_param.hard_swish_threshold / output_scale;
    }

    for (auto& ws : w_scale_) {
      ws /= output_scale;
    }
    if (param.bias) {
      auto ptr = bias_.mutable_data<float>();
      for (int i = 0; i < bias_.numel(); ++i) {
        ptr[i] /= output_scale;
      }
    }
  }

  //! update trans weights impl
  // choose_small_ = ow * oh / (tile_block * threads) < 36 ? true : false;
  // select best wino_unit
  int wino_unit = ow * oh / (tile_block * threads);
  if (wino_unit < 16) {
    wino_iw = 4;
    if (last_function_ == 0) {
      return;
    }
    last_function_ = 0;
    for (auto& ws : w_scale_) {
      ws *= 0.25f;
    }
  } else {
    wino_iw = 6;
    if (last_function_ == 1) {
      return;
    }
    last_function_ = 1;
    for (auto& ws : w_scale_) {
      ws /= 576;
    }
  }
  last_function_ = -1;

  const int new_input_size =
      ic_pad * (ih + pad_h0 + pad_h1) * (iw + pad_w0 + pad_w1) +
      oc_pad * oh * ow * sizeof(int32_t);
  int tmp_input_thread_size_byte =
      tile_block * ic_pad * wino_iw * wino_iw * sizeof(int16_t);
  int tmp_output_thread_size_byte =
      tile_block * oc_pad * wino_iw * wino_iw * sizeof(int32_t);
  int tmp_trans_size_byte = wino_iw * wino_iw * sizeof(int16_t) * 8;
  int tmp_remain_trans_size_byte = wino_iw * wino_iw * sizeof(int8_t) * 8;
  int tmp_trans_out_size_byte = wino_iw * (wino_iw - 2) * sizeof(int32_t) * 8;
  int tmp_remain_trans_out_size_byte =
      (wino_iw - 2) * (wino_iw - 2) * sizeof(int32_t) * 8;
  const int temp_size = tmp_input_thread_size_byte +
                        tmp_output_thread_size_byte + tmp_trans_size_byte +
                        tmp_remain_trans_size_byte + tmp_trans_out_size_byte +
                        tmp_remain_trans_out_size_byte;
  workspace_size_ = (temp_size + new_input_size) * 2;

  weights_.Resize({1, 1, 1, wino_iw * wino_iw * oc_pad * ic_pad});
  void* trans_tmp_ptr = malloc(sizeof(int32_t) * wino_iw * wino_iw * oc * ic);
  auto weights_data_ = weights_.mutable_data<int16_t>();
  memset(reinterpret_cast<char*>(weights_data_),
         0,
         weights_.numel() * sizeof(int16_t));
  switch (wino_iw) {
    case 4:
      lite::arm::math::weight_trans_c8_4x4_int8(
          weights_data_,
          param.filter->template data<int8_t>(),
          ic,
          oc,
          trans_tmp_ptr);
      break;
    case 6:
      lite::arm::math::weight_trans_c8_6x6_int8(
          weights_data_,
          param.filter->template data<int8_t>(),
          ic,
          oc,
          trans_tmp_ptr);
      break;
    default:
      lite::arm::math::weight_trans_c8_6x6_int8(
          weights_data_,
          param.filter->template data<int8_t>(),
          ic,
          oc,
          trans_tmp_ptr);
  }
  free(trans_tmp_ptr);
}

template <PrecisionType OutType>
void WinogradConv<PRECISION(kInt8), OutType>::PrepareForRun() {
  ReInitWhenNeeded();
}

template <PrecisionType OutType>
void WinogradConv<PRECISION(kInt8), OutType>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  ctx.ExtendWorkspace(workspace_size_);
  const auto* i_data = param.x->template data<int8_t>();
  const auto* w_data = weights_.data<int16_t>();
  const auto* b_data = param.bias ? bias_.data<float>() : nullptr;
  // const float* i_data;
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

  // now  always choose small
  if (OutType == PRECISION(kInt8)) {
    auto* o_data = param.output->template mutable_data<int8_t>();
    if (wino_iw == 6) {
      lite::arm::math::conv_compute_4x4_3x3_int8<int8_t>(
          FUNCS_PARAM, w_scale_.data(), param, &ctx);
      KERNEL_FUNC_NAME("conv_compute_4x4_3x3_int8_int8")
    } else {
      lite::arm::math::conv_compute_2x2_3x3_int8<int8_t>(
          FUNCS_PARAM, w_scale_.data(), param, &ctx);
      KERNEL_FUNC_NAME("conv_compute_2x2_3x3_int8_int8")
    }
  } else {
    auto* o_data = param.output->template mutable_data<float>();
    if (wino_iw == 6) {
      lite::arm::math::conv_compute_4x4_3x3_int8<float>(
          FUNCS_PARAM, w_scale_.data(), param, &ctx);
      KERNEL_FUNC_NAME("conv_compute_4x4_3x3_int8_float")
    } else {
      lite::arm::math::conv_compute_2x2_3x3_int8<float>(
          FUNCS_PARAM, w_scale_.data(), param, &ctx);
      KERNEL_FUNC_NAME("conv_compute_2x2_3x3_int8_float")
    }
  }
}

PROFILE_INFO(kInt8, kInt8)
PROFILE_INFO(kInt8, kFloat)
template class WinogradConv<PRECISION(kInt8), PRECISION(kInt8)>;
template class WinogradConv<PRECISION(kInt8), PRECISION(kFloat)>;

#ifdef ENABLE_ARM_FP16
template <>
void WinogradConv<PRECISION(kFP16), PRECISION(kFP16)>::ReInitWhenNeeded() {
  WINOGRAD_INIT(8)

  // select best wino_unit
  int wino_unit = ow * oh / (tile_block * threads);
  if (wino_unit < 16) {
    wino_iw = 4;
    if (last_function_ == 0) {
      return;
    }
    last_function_ = 0;
  } else {
    wino_iw = 6;
    if (last_function_ == 1) {
      return;
    }
    last_function_ = 1;
  }
  last_function_ = -1;

  const int new_input_size =
      ic_pad * (ih + pad_h0 + pad_h1) * (iw + pad_w0 + pad_w1);
  const int temp_size = (tile_block * (ic_pad + oc_pad) * wino_iw * wino_iw +
                         8 * wino_iw * wino_iw) *
                        threads;
  workspace_size_ = (temp_size + new_input_size) * sizeof(float16_t);

  weights_.Resize({1, 1, 1, wino_iw * wino_iw * oc_pad * ic_pad});
  void* trans_tmp_ptr = malloc(sizeof(float16_t) * wino_iw * wino_iw * oc * ic);
  auto weights_data_ = weights_.mutable_data<float16_t>();
  memset(reinterpret_cast<char*>(weights_data_),
         0,
         weights_.numel() * sizeof(int16_t));
  switch (wino_iw) {
    case 4:
      lite::arm::math::fp16::weight_trans_c8_4x4_fp16(
          weights_data_,
          param.filter->template data<float16_t>(),
          ic,
          oc,
          trans_tmp_ptr);
      break;
    case 6:
      lite::arm::math::fp16::weight_trans_c8_6x6_fp16(
          weights_data_,
          param.filter->template data<float16_t>(),
          ic,
          oc,
          trans_tmp_ptr);
      break;
    default:
      lite::arm::math::fp16::weight_trans_c8_6x6_fp16(
          weights_data_,
          param.filter->template data<float16_t>(),
          ic,
          oc,
          trans_tmp_ptr);
  }
  free(trans_tmp_ptr);
}

template <>
void WinogradConv<PRECISION(kFP16), PRECISION(kFP16)>::PrepareForRun() {
  ReInitWhenNeeded();
}

template <>
void WinogradConv<PRECISION(kFP16), PRECISION(kFP16)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  ctx.ExtendWorkspace(workspace_size_);
  const auto* i_data = param.x->template data<float16_t>();
  const auto* w_data = weights_.data<float16_t>();
  const auto* b_data =
      param.bias ? param.bias->template data<float16_t>() : nullptr;
  auto* o_data = param.output->template mutable_data<float16_t>();
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

  if (wino_iw == 6) {
    lite::arm::math::fp16::conv_compute_4x4_3x3_fp16(FUNCS_PARAM, param, &ctx);
    KERNEL_FUNC_NAME("conv_compute_4x4_3x3_fp16")
  } else {
    lite::arm::math::fp16::conv_compute_2x2_3x3_fp16(FUNCS_PARAM, param, &ctx);
    KERNEL_FUNC_NAME("conv_compute_2x2_3x3_fp16")
  }
}

#endif
#undef FUNCS_PARAM
#undef WINOGRAD_INIT
}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
