/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef QUANT_OP

#include "operators/kernel/quantize_kernel.h"
#include <cmath>
#include "framework/context.h"
#include "operators/math/quantize.h"

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#ifndef __aarch64__
inline float32_t vmaxvq_f32(float32x4_t r) {
  float32x2_t v = vmax_f32(vget_high_f32(r), vget_low_f32(r));
  return vget_lane_f32(vpmax_f32(v, v), 0);
}
#endif

template <RoundType R>
inline void QuantizeOffline(const Tensor *input, const float scale,
                            const float max_abs, Tensor *output) {
  const float *x = input->data<const float>();
  int8_t *y = output->mutable_data<int8_t>();
  size_t remain = input->numel();
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  size_t loop = remain >> 4;
  remain = remain & 0xF;
  float32x4_t __scale = vdupq_n_f32(scale);
  float32x4_t __postive_max = vdupq_n_f32(max_abs);
  float32x4_t __negtive_max = vdupq_n_f32(-max_abs);

#pragma omp parallel for num_threads(framework::threads())
  for (size_t i = 0; i < loop; ++i) {
    const float *local_x = x + (i << 4);
    int8_t *local_y = y + (i << 4);
    float32x4_t r0 = vld1q_f32(local_x);
    float32x4_t r1 = vld1q_f32(local_x + 4);
    float32x4_t r2 = vld1q_f32(local_x + 8);
    float32x4_t r3 = vld1q_f32(local_x + 12);
    r0 = vmaxq_f32(vminq_f32(r0, __postive_max), __negtive_max);
    r1 = vmaxq_f32(vminq_f32(r1, __postive_max), __negtive_max);
    r2 = vmaxq_f32(vminq_f32(r2, __postive_max), __negtive_max);
    r3 = vmaxq_f32(vminq_f32(r3, __postive_max), __negtive_max);
    r0 = vmulq_f32(r0, __scale);
    r1 = vmulq_f32(r1, __scale);
    r2 = vmulq_f32(r2, __scale);
    r3 = vmulq_f32(r3, __scale);
    int32x4_t q0 = math::vRoundq_f32<R>(r0);
    int32x4_t q1 = math::vRoundq_f32<R>(r1);
    int32x4_t q2 = math::vRoundq_f32<R>(r2);
    int32x4_t q3 = math::vRoundq_f32<R>(r3);
    int16x4_t d0 = vmovn_s32(q0);
    int16x4_t d1 = vmovn_s32(q1);
    int16x4_t d2 = vmovn_s32(q2);
    int16x4_t d3 = vmovn_s32(q3);
    int16x8_t q5 = vcombine_s16(d0, d1);
    int16x8_t q6 = vcombine_s16(d2, d3);
    int8x8_t d5 = vmovn_s16(q5);
    int8x8_t d6 = vmovn_s16(q6);
    vst1_s8(local_y, d5);
    vst1_s8(local_y + 8, d6);
  }
  x += (loop << 4);
  y += (loop << 4);
#endif
  for (size_t i = 0; i < remain; ++i) {
    float x_temp = std::max(std::min(x[i], max_abs), -max_abs);
    y[i] = math::Round<R>(x_temp * scale);
  }
}

template <RoundType R>
inline void QuantizeOnline(const Tensor *input, const float scale,
                           Tensor *output) {
  const float *x = input->data<const float>();
  int8_t *y = output->mutable_data<int8_t>();
  size_t remain = input->numel();
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  size_t loop = remain >> 4;
  remain = remain & 0xF;
  float32x4_t __scale = vdupq_n_f32(scale);

#pragma omp parallel for num_threads(framework::threads())
  for (size_t i = 0; i < loop; ++i) {
    const float *local_x = x + (i << 4);
    int8_t *local_y = y + (i << 4);
    float32x4_t r0 = vld1q_f32(local_x);
    float32x4_t r1 = vld1q_f32(local_x + 4);
    float32x4_t r2 = vld1q_f32(local_x + 8);
    float32x4_t r3 = vld1q_f32(local_x + 12);
    r0 = vmulq_f32(r0, __scale);
    r1 = vmulq_f32(r1, __scale);
    r2 = vmulq_f32(r2, __scale);
    r3 = vmulq_f32(r3, __scale);
    int32x4_t q0 = math::vRoundq_f32<R>(r0);
    int32x4_t q1 = math::vRoundq_f32<R>(r1);
    int32x4_t q2 = math::vRoundq_f32<R>(r2);
    int32x4_t q3 = math::vRoundq_f32<R>(r3);
    int16x4_t d0 = vmovn_s32(q0);
    int16x4_t d1 = vmovn_s32(q1);
    int16x4_t d2 = vmovn_s32(q2);
    int16x4_t d3 = vmovn_s32(q3);
    int16x8_t q5 = vcombine_s16(d0, d1);
    int16x8_t q6 = vcombine_s16(d2, d3);
    int8x8_t d5 = vmovn_s16(q5);
    int8x8_t d6 = vmovn_s16(q6);
    vst1_s8(local_y, d5);
    vst1_s8(local_y + 8, d6);
  }
  x += (loop << 4);
  y += (loop << 4);
#endif
  for (size_t i = 0; i < remain; ++i) {
    y[i] = math::Round<R>(x[i] * scale);
  }
}

template <RoundType R>
static void Quantize(const Tensor *input, const float max_abs,
                     const bool offline, Tensor *output) {
  float scale = 127.f / max_abs;
  if (offline) {
    QuantizeOffline<R>(input, scale, max_abs, output);
  } else {
    QuantizeOnline<R>(input, scale, output);
  }
}

float find_abs_max(const Tensor *input) {
  float max_abs = 0.f;
  const float *x = input->data<const float>();
  size_t remain = input->numel();
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  size_t loop = remain >> 4;
  remain = remain & 0xF;
  float32x4_t __max = {0.f, 0.f, 0.f, 0.f};

  for (size_t i = 0; i < loop; ++i, x += 16) {
    float32x4_t r0 = vld1q_f32(x);
    float32x4_t r1 = vld1q_f32(x + 4);
    float32x4_t r2 = vld1q_f32(x + 8);
    float32x4_t r3 = vld1q_f32(x + 12);
    r0 = vabsq_f32(r0);
    r1 = vabsq_f32(r1);
    r2 = vabsq_f32(r2);
    r3 = vabsq_f32(r3);
    r0 = vmaxq_f32(r0, r1);
    r1 = vmaxq_f32(r2, r3);
    r0 = vmaxq_f32(r0, r1);
    __max = vmaxq_f32(r0, __max);
  }
  max_abs = vmaxvq_f32(__max);
#endif
  for (size_t i = 0; i < remain; ++i) {
    max_abs = std::max(max_abs, static_cast<float>(fabs(x[i])));
  }
  return max_abs;
}

}  // namespace operators
}  // namespace paddle_mobile
#endif  // __ARM_NEON__

namespace paddle_mobile {
namespace operators {

template <>
bool QuantizeKernel<CPU, float>::Init(QuantizeParam<CPU> *param) {
  return true;
}

template <>
void QuantizeKernel<CPU, float>::Compute(const QuantizeParam<CPU> &param) {
  const LoDTensor *input = param.input_;
  LoDTensor *output = param.output_;
  Tensor *output_scale = param.online_scale_;
  float max_abs = 0.f;
  if (param.offline_) {
    max_abs = param.offline_scale_->data<float>()[0];
  } else {
    max_abs = find_abs_max(input);
  }
  max_abs = std::max(max_abs, 1e-6f);
  param.online_scale_->mutable_data<float>()[0] = max_abs;
  switch (param.round_type_) {
    case ROUND_NEAREST_TO_EVEN:
      Quantize<ROUND_NEAREST_TO_EVEN>(input, max_abs, param.offline_, output);
      break;
    case ROUND_NEAREST_TOWARDS_ZERO:
      Quantize<ROUND_NEAREST_TOWARDS_ZERO>(input, max_abs, param.offline_,
                                           output);
      break;
    case ROUND_NEAREST_AWAY_ZERO:
      Quantize<ROUND_NEAREST_AWAY_ZERO>(input, max_abs, param.offline_, output);
      break;
    default:
      LOG(kLOG_ERROR) << "round type is not supported.";
      break;
  }
  output->set_lod(input->lod());
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // QUANT_OP
