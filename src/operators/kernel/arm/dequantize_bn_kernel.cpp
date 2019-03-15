/* Copyright (c) 201f8 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cmath>
#include "framework/context.h"
#include "operators/kernel/dequant_bn_kernel.h"
#include "operators/math/activation.h"
#include "operators/math/quantize.h"
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {

#if defined(FUSION_DEQUANT_BN_OP) || defined(FUSION_DEQUANT_ADD_BN_OP) || \
    defined(FUSION_DEQUANT_BN_RELU_OP) ||                                 \
    defined(FUSION_DEQUANT_ADD_BN_RELU_OP) ||                             \
    defined(FUSION_DEQUANT_ADD_BN_QUANT_OP) ||                            \
    defined(FUSION_DEQUANT_ADD_BN_RELU_QUANT_OP)
void PublicFusionDequantBNInitParam(FusionDequantBNParam<CPU> *param,
                                    const framework::Tensor *bias) {
  // batch norm params
  const Tensor *bn_mean = param->bn_mean_;
  const Tensor *bn_variance = param->bn_variance_;
  Tensor *bn_scale = param->bn_scale_;
  Tensor *bn_bias = param->bn_bias_;
  const float epsilon = param->epsilon_;

  const float *mean_ptr = bn_mean->data<float>();
  const float *var_ptr = bn_variance->data<float>();
  float *bn_scale_ptr = bn_scale->mutable_data<float>();
  float *bn_bias_ptr = bn_bias->mutable_data<float>();
  for (int c = 0; c < bn_scale->numel(); ++c) {
    float inv_scale = 1.f / (std::sqrt(var_ptr[c] + epsilon));
    float val = bias ? bias->data<float>()[c] : 0;
    bn_bias_ptr[c] =
        inv_scale * bn_scale_ptr[c] * (val - mean_ptr[c]) + bn_bias_ptr[c];
    bn_scale_ptr[c] = inv_scale * bn_scale_ptr[c];
  }
}
#endif

#if defined(FUSION_DEQUANT_BN_OP) || defined(FUSION_DEQUANT_ADD_BN_OP) || \
    defined(FUSION_DEQUANT_BN_RELU_OP) ||                                 \
    defined(FUSION_DEQUANT_ADD_BN_RELU_OP)
template <ActivationType Act>
void DequantBNCompute(const FusionDequantBNParam<CPU> *param) {
  const int32_t *input = param->input_->data<int32_t>();
  const float *bn_scale = param->bn_scale_->data<float>();
  const float *bn_bias = param->bn_bias_->data<float>();
  // dequantize params
  const float activation_scale = param->activation_scale_->data<float>()[0];
  const float weight_scale = param->weight_scale_;
  const float dequant_scale = activation_scale / weight_scale;

  float *output = param->output_->mutable_data<float>();
  int batch_size = param->input_->dims()[0];
  int channels = param->input_->dims()[1];
  size_t spatial_size = param->input_->dims()[2] * param->input_->dims()[3];

  #pragma omp parallel for collapse(2)
  // num_threads(framework::threads())
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int c = 0; c < channels; ++c) {
      // not fuse bn and dequant scale to minimize precision difference
      // float scale = bn_scale[c] * dequant_scale;
      float scale = bn_scale[c];
      float bias = bn_bias[c];
      size_t offset = (batch * channels + c) * spatial_size;
      const int32_t *x = input + offset;
      float *y = output + offset;
      size_t remain = spatial_size;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      int loop = spatial_size >> 4;
      remain = spatial_size & 0xF;
      float32x4_t __dequant_scale = vdupq_n_f32(dequant_scale);
      float32x4_t __scale = vdupq_n_f32(scale);
      float32x4_t __bias = vdupq_n_f32(bias);
      for (int k = 0; k < loop; ++k, x += 16, y += 16) {
        int32x4_t r0 = vld1q_s32(x);
        int32x4_t r1 = vld1q_s32(x + 4);
        int32x4_t r2 = vld1q_s32(x + 8);
        int32x4_t r3 = vld1q_s32(x + 12);
        float32x4_t f0 = vcvtq_f32_s32(r0);
        float32x4_t f1 = vcvtq_f32_s32(r1);
        float32x4_t f2 = vcvtq_f32_s32(r2);
        float32x4_t f3 = vcvtq_f32_s32(r3);
        f0 = vmulq_f32(__dequant_scale, f0);
        f1 = vmulq_f32(__dequant_scale, f1);
        f2 = vmulq_f32(__dequant_scale, f2);
        f3 = vmulq_f32(__dequant_scale, f3);
        f0 = vmlaq_f32(__bias, __scale, f0);
        f1 = vmlaq_f32(__bias, __scale, f1);
        f2 = vmlaq_f32(__bias, __scale, f2);
        f3 = vmlaq_f32(__bias, __scale, f3);
        f0 = math::vActiveq_f32<Act>(f0);
        f1 = math::vActiveq_f32<Act>(f1);
        f2 = math::vActiveq_f32<Act>(f2);
        f3 = math::vActiveq_f32<Act>(f3);
        vst1q_f32(y, f0);
        vst1q_f32(y + 4, f1);
        vst1q_f32(y + 8, f2);
        vst1q_f32(y + 12, f3);
      }
#endif  // __ARM_NEON__
      for (int k = 0; k < remain; ++k) {
        y[k] = math::Active<Act>(scale * (dequant_scale * x[k]) + bias);
      }
    }
  }
}
#endif

#ifdef FUSION_DEQUANT_BN_OP
template <>
bool FusionDequantBNKernel<CPU, float>::Init(FusionDequantBNParam<CPU> *param) {
  PublicFusionDequantBNInitParam(param, nullptr);
  return true;
}

template <>
void FusionDequantBNKernel<CPU, float>::Compute(
    const FusionDequantBNParam<CPU> &param) {
  DequantBNCompute<IDENTITY>(&param);
}
#endif  // FUSION_DEQUANT_BN_OP

#ifdef FUSION_DEQUANT_BN_RELU_OP
template <>
bool FusionDequantBNReluKernel<CPU, float>::Init(
    FusionDequantBNParam<CPU> *param) {
  PublicFusionDequantBNInitParam(param, nullptr);
  return true;
}

template <>
void FusionDequantBNReluKernel<CPU, float>::Compute(
    const FusionDequantBNParam<CPU> &param) {
  DequantBNCompute<RELU>(&param);
}
#endif  // FUSION_DEQUANT_BN_RELU_OP

#ifdef FUSION_DEQUANT_ADD_BN_OP
template <>
bool FusionDequantAddBNKernel<CPU, float>::Init(
    FusionDequantAddBNParam<CPU> *param) {
  const framework::Tensor *bias = param->bias_;
  PublicFusionDequantBNInitParam(param, bias);
  return true;
}

template <>
void FusionDequantAddBNKernel<CPU, float>::Compute(
    const FusionDequantAddBNParam<CPU> &param) {
  DequantBNCompute<IDENTITY>(&param);
}
#endif  // FUSION_DEQUANT_ADD_BN_OP

#ifdef FUSION_DEQUANT_ADD_BN_RELU_OP
template <>
bool FusionDequantAddBNReluKernel<CPU, float>::Init(
    FusionDequantAddBNParam<CPU> *param) {
  const framework::Tensor *bias = param->bias_;
  PublicFusionDequantBNInitParam(param, bias);
  return true;
}

template <>
void FusionDequantAddBNReluKernel<CPU, float>::Compute(
    const FusionDequantAddBNParam<CPU> &param) {
  DequantBNCompute<RELU>(&param);
}
#endif  // FUSION_DEQUANT_ADD_BN_RELU_OP

#if defined(FUSION_DEQUANT_ADD_BN_QUANT_OP) || \
    defined(FUSION_DEQUANT_ADD_BN_RELU_QUANT_OP)
template <Activation Act, RoundType R>
void DequantBNQuantCompute(const FusionDequantAddBNQuantParam<CPU> *param) {
  const int32_t *input = param->input_->data<int32_t>();
  const float *bn_scale = param->bn_scale_->data<float>();
  const float *bn_bias = param->bn_bias_->data<float>();
  // dequantize params
  const float activation_scale = param->activation_scale_->data<float>()[0];
  const float weight_scale = param->weight_scale_;
  const float dequant_scale = activation_scale / weight_scale;
  // quantize params
  Tensor *output_scale = param->online_scale_;
  float max_abs = 0.f;

  int8_t *output = param->output_->mutable_data<int8_t>();
  int batch_size = param->input_->dims()[0];
  int channels = param->input_->dims()[1];
  size_t spatial_size = param->input_->dims()[2] * param->input_->dims()[3];

  //  if (param->is_static_) {
  if (true) {
    max_abs = param->static_scale_;
    float quant_scale = 127.f / max_abs;
    #pragma omp parallel for collapse(2)
    // num_threads(framework::threads())
    for (int batch = 0; batch < batch_size; ++batch) {
      for (int c = 0; c < channels; ++c) {
        // not fuse bn and dequant scale to minimize precision difference
        // float scale = bn_scale[c] * dequant_scale;
        float scale = bn_scale[c];
        float bias = bn_bias[c];
        size_t offset = (batch * channels + c) * spatial_size;
        const int32_t *x = input + offset;
        int8_t *y = output + offset;
        size_t remain = spatial_size;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
        int loop = spatial_size >> 4;
        remain = spatial_size & 0xF;
        float32x4_t __dequant_scale = vdupq_n_f32(dequant_scale);
        float32x4_t __scale = vdupq_n_f32(scale);
        float32x4_t __bias = vdupq_n_f32(bias);
        float32x4_t __quant_scale = vdupq_n_f32(quant_scale);
        for (int k = 0; k < loop; ++k, x += 16, y += 16) {
          int32x4_t r0 = vld1q_s32(x);
          int32x4_t r1 = vld1q_s32(x + 4);
          int32x4_t r2 = vld1q_s32(x + 8);
          int32x4_t r3 = vld1q_s32(x + 12);
          float32x4_t f0 = vcvtq_f32_s32(r0);
          float32x4_t f1 = vcvtq_f32_s32(r1);
          float32x4_t f2 = vcvtq_f32_s32(r2);
          float32x4_t f3 = vcvtq_f32_s32(r3);
          f0 = vmulq_f32(__dequant_scale, f0);
          f1 = vmulq_f32(__dequant_scale, f1);
          f2 = vmulq_f32(__dequant_scale, f2);
          f3 = vmulq_f32(__dequant_scale, f3);
          f0 = vmlaq_f32(__bias, __scale, f0);
          f1 = vmlaq_f32(__bias, __scale, f1);
          f2 = vmlaq_f32(__bias, __scale, f2);
          f3 = vmlaq_f32(__bias, __scale, f3);
          f0 = math::vActiveq_f32<Act>(f0);
          f1 = math::vActiveq_f32<Act>(f1);
          f2 = math::vActiveq_f32<Act>(f2);
          f3 = math::vActiveq_f32<Act>(f3);
          f0 = vmulq_f32(__quant_scale, f0);
          f1 = vmulq_f32(__quant_scale, f1);
          f2 = vmulq_f32(__quant_scale, f2);
          f3 = vmulq_f32(__quant_scale, f3);
          int32x4_t q0 = math::vRoundq_f32<R>(f0);
          int32x4_t q1 = math::vRoundq_f32<R>(f1);
          int32x4_t q2 = math::vRoundq_f32<R>(f2);
          int32x4_t q3 = math::vRoundq_f32<R>(f3);
          int16x4_t d0 = vmovn_s32(q0);
          int16x4_t d1 = vmovn_s32(q1);
          int16x4_t d2 = vmovn_s32(q2);
          int16x4_t d3 = vmovn_s32(q3);
          int16x8_t q5 = vcombine_s16(d0, d1);
          int16x8_t q6 = vcombine_s16(d2, d3);
          int8x8_t d5 = vmovn_s16(q5);
          int8x8_t d6 = vmovn_s16(q6);
          vst1_s8(y, d5);
          vst1_s8(y + 8, d6);
        }
#endif  // __ARM_NEON__
        for (int k = 0; k < remain; ++k) {
          float x_temp =
              math::Active<Act>(scale * (dequant_scale * x[k]) + bias);
          y[k] = math::Round<R>(x_temp * quant_scale);
        }
      }
    }
  } else {
    // TODO(hjchen2)
    max_abs = std::max(max_abs, 1e-6f);
  }
  param->online_scale_->mutable_data<float>()[0] = max_abs;
}

template <>
bool FusionDequantAddBNQuantKernel<CPU, float>::Init(
    FusionDequantAddBNQuantParam<CPU> *param) {
  const framework::Tensor *bias = param->bias_;
  PublicFusionDequantBNInitParam(param, bias);
  return true;
}

template <>
void FusionDequantAddBNQuantKernel<CPU, float>::Compute(
    const FusionDequantAddBNQuantParam<CPU> &param) {
  switch (param.round_type_) {
    case ROUND_NEAREST_TO_EVEN:
      DequantBNQuantCompute<IDENTITY, ROUND_NEAREST_TO_EVEN>(&param);
      break;
    case ROUND_NEAREST_TOWARDS_ZERO:
      DequantBNQuantCompute<IDENTITY, ROUND_NEAREST_TOWARDS_ZERO>(&param);
      break;
    case ROUND_NEAREST_AWAY_ZERO:
      DequantBNQuantCompute<IDENTITY, ROUND_NEAREST_AWAY_ZERO>(&param);
      break;
    default:
      LOG(kLOG_ERROR) << "round type is not supported.";
      break;
  }
}
#endif  // FUSION_DEQUANT_ADD_BN_QUANT_OP

#ifdef FUSION_DEQUANT_ADD_BN_RELU_QUANT_OP
template <>
bool FusionDequantAddBNReluQuantKernel<CPU, float>::Init(
    FusionDequantAddBNQuantParam<CPU> *param) {
  const framework::Tensor *bias = param->bias_;
  PublicFusionDequantBNInitParam(param, bias);
  return true;
}

template <>
void FusionDequantAddBNReluQuantKernel<CPU, float>::Compute(
    const FusionDequantAddBNQuantParam<CPU> &param) {
  switch (param.round_type_) {
    case ROUND_NEAREST_TO_EVEN:
      DequantBNQuantCompute<RELU, ROUND_NEAREST_TO_EVEN>(&param);
      break;
    case ROUND_NEAREST_TOWARDS_ZERO:
      DequantBNQuantCompute<RELU, ROUND_NEAREST_TOWARDS_ZERO>(&param);
      break;
    case ROUND_NEAREST_AWAY_ZERO:
      DequantBNQuantCompute<RELU, ROUND_NEAREST_AWAY_ZERO>(&param);
      break;
    default:
      LOG(kLOG_ERROR) << "round type is not supported.";
      break;
  }
}
#endif  // FUSION_DEQUANT_ADD_BN_RELU_QUANT_OP

}  // namespace operators
}  // namespace paddle_mobile
