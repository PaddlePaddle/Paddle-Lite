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

#ifdef SOFTMAX_OP

#include "operators/math/softmax.h"
#include <math.h>
#include <algorithm>
#include <limits>
#include "common/types.h"
#include "operators/math/math.h"

namespace paddle_mobile {
namespace operators {
namespace math {

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#ifndef __aarch64__
inline float32_t vmaxvq_f32(const float32x4_t &r) {
  float32x2_t v = vmax_f32(vget_high_f32(r), vget_low_f32(r));
  return vget_lane_f32(vpmax_f32(v, v), 0);
}

inline float32_t vaddvq_f32(const float32x4_t &r) {
  float32x2_t v = vadd_f32(vget_high_f32(r), vget_low_f32(r));
  return vget_lane_f32(vpadd_f32(v, v), 0);
}
#endif  // __aarch64__
#endif  // __ARM_NEON__

float find_max(const float *input, const int num_classes) {
  int remain = num_classes;
  float max = -std::numeric_limits<float>::max();
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  int loop = num_classes >> 3;
  remain = num_classes & 0x7;
  float32x4_t __max = vdupq_n_f32(max);
  for (int i = 0; i < loop; ++i, input += 8) {
    float32x4_t x0 = vld1q_f32(input);
    float32x4_t x1 = vld1q_f32(input + 4);
    __max = vmaxq_f32(x0, __max);
    __max = vmaxq_f32(x1, __max);
  }
  max = vmaxvq_f32(__max);
#endif
  for (int i = 0; i < remain; ++i) {
    max = std::max(max, input[i]);
  }
  return max;
}

void SoftmaxBasic(const float *input, int num_classes, float *y) {
  float *output = y;
  // find max
  float max = find_max(input, num_classes);

  // exp(x - max) and sum(exp(x - max))
  int remain = num_classes;
  float sum = 0.f;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  int loop = num_classes >> 3;
  remain = num_classes & 0x7;
  float32x4_t __max = vdupq_n_f32(max);
  float32x4_t __sum = vdupq_n_f32(0.f);
  for (int i = 0; i < loop; ++i, input += 8, output += 8) {
    float32x4_t x0 = vld1q_f32(input);
    float32x4_t x1 = vld1q_f32(input + 4);
    x0 = vsubq_f32(x0, __max);
    x1 = vsubq_f32(x1, __max);
    x0 = exp_ps(x0);
    x1 = exp_ps(x1);
    __sum = vaddq_f32(x0, __sum);
    __sum = vaddq_f32(x1, __sum);
    vst1q_f32(output, x0);
    vst1q_f32(output + 4, x1);
  }
  sum += vaddvq_f32(__sum);
#endif  // __ARM_NEON__
  for (int i = 0; i < remain; ++i) {
    float out = expf(input[i] - max);
    sum += out;
    output[i] = out;
  }

  // exp(x - max) / sum
  float inv_sum = 1.f / sum;
  output = y;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  float32x4_t __inv_sum = vdupq_n_f32(inv_sum);
  for (int i = 0; i < loop; ++i, output += 8) {
    float32x4_t x0 = vld1q_f32(output);
    float32x4_t x1 = vld1q_f32(output + 4);
    x0 = vmulq_f32(x0, __inv_sum);
    x1 = vmulq_f32(x1, __inv_sum);
    vst1q_f32(output, x0);
    vst1q_f32(output + 4, x1);
  }
#endif
  for (int i = 0; i < remain; ++i) {
    output[i] *= inv_sum;
  }
}

template <>
void SoftmaxFuntor<CPU, float>::operator()(const framework::Tensor *X,
                                           framework::Tensor *Y) {
  const framework::DDim &dims = X->dims();
  int batch_size = dims[0];
  int num_classes = dims[dims.size() - 1];
  int channels = X->numel() / batch_size / num_classes;
  const float *x = X->data<float>();
  float *y = Y->mutable_data<float>();

#pragma omp parallel for collapse(2) num_threads(framework::threads())
  for (int batch = 0; batch < X->dims()[0]; ++batch) {
    for (int channel = 0; channel < channels; ++channel) {
      size_t offset = (batch * channels + channel) * num_classes;
      const float *input = x + offset;
      float *output = y + offset;
      SoftmaxBasic(input, num_classes, output);
    }
  }
}

template <>
void SequenceSoftmaxFuntor<CPU, float>::operator()(
    const framework::LoDTensor *X, framework::LoDTensor *Y) {
  const float *x = X->data<float>();
  const auto &lod = X->lod().back();
  float *y = Y->mutable_data<float>();

#pragma omp parallel for num_threads(framework::threads())
  for (int batch = 0; batch < lod.size() - 1; ++batch) {
    int num_classes = lod[batch + 1] - lod[batch];
    size_t offset = lod[batch];
    const float *input = x + offset;
    float *output = y + offset;
    SoftmaxBasic(input, num_classes, output);
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif  // SOFTMAX_OP
