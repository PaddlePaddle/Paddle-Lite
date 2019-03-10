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

#pragma once

#include "framework/tensor.h"
#include "operators/math/activation.h"
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {
namespace math {

template <ActivationType Act>
void AddChannelWise(const framework::Tensor *input,
                    const framework::Tensor *bias, framework::Tensor *output) {
  const float *input_ptr = input->data<float>();
  const float *bias_ptr = bias->data<float>();
  float *output_ptr = output->mutable_data<float>();
  // maybe check shape
  int batch_size = input->dims()[0];
  int channels = input->dims()[1];
  size_t spatial_size = input->dims()[2] * input->dims()[3];

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int channel = 0; channel < channels; ++channel) {
      size_t offset = (batch * channels + channel) * spatial_size;
      const float *x = input_ptr + offset;
      float *y = output_ptr + offset;
      float beta = bias_ptr[channel];
      int j = 0;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      float32x4_t __bias = vdupq_n_f32(beta);
      for (; j < spatial_size - 15; j += 16, x += 16, y += 16) {
        float32x4_t in0 = vld1q_f32(x);
        float32x4_t in1 = vld1q_f32(x + 4);
        float32x4_t in2 = vld1q_f32(x + 8);
        float32x4_t in3 = vld1q_f32(x + 12);
        in0 = vaddq_f32(__bias, in0);
        in1 = vaddq_f32(__bias, in1);
        in2 = vaddq_f32(__bias, in2);
        in3 = vaddq_f32(__bias, in3);
        in0 = math::vActiveq_f32<Act>(in0);
        in1 = math::vActiveq_f32<Act>(in1);
        in2 = math::vActiveq_f32<Act>(in2);
        in3 = math::vActiveq_f32<Act>(in3);
        vst1q_f32(y, in0);
        vst1q_f32(y + 4, in1);
        vst1q_f32(y + 8, in2);
        vst1q_f32(y + 12, in3);
      }
      for (; j < spatial_size - 3; j += 4, x += 4, y += 4) {
        float32x4_t in0 = vld1q_f32(x);
        in0 = vaddq_f32(__bias, in0);
        in0 = math::vActiveq_f32<Act>(in0);
        vst1q_f32(y, in0);
      }
#endif
      for (; j < spatial_size; ++j, ++x, ++y) {
        *y = math::Active<Act>((*x) + beta);
      }
    }
  }
}

template <ActivationType Act>
void ScaleAddChannelWise(const framework::Tensor *input,
                         const framework::Tensor *scale,
                         const framework::Tensor *bias,
                         framework::Tensor *output) {
  const float *input_ptr = input->data<float>();
  const float *scale_ptr = scale->data<float>();
  const float *bias_ptr = bias->data<float>();
  float *output_ptr = output->mutable_data<float>();
  // maybe check shape
  int batch_size = input->dims()[0];
  int channels = input->dims()[1];
  size_t spatial_size = input->dims()[2] * input->dims()[3];

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int channel = 0; channel < channels; ++channel) {
      size_t offset = (batch * channels + channel) * spatial_size;
      const float *x = input_ptr + offset;
      float *y = output_ptr + offset;
      float alpha = scale_ptr[channel];
      float beta = bias_ptr[channel];
      int j = 0;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      float32x4_t __scale = vdupq_n_f32(alpha);
      float32x4_t __bias = vdupq_n_f32(beta);
      for (; j < spatial_size - 15; j += 16, x += 16, y += 16) {
        float32x4_t in0 = vld1q_f32(x);
        float32x4_t in1 = vld1q_f32(x + 4);
        float32x4_t in2 = vld1q_f32(x + 8);
        float32x4_t in3 = vld1q_f32(x + 12);
        in0 = vmlaq_f32(__bias, __scale, in0);
        in1 = vmlaq_f32(__bias, __scale, in1);
        in2 = vmlaq_f32(__bias, __scale, in2);
        in3 = vmlaq_f32(__bias, __scale, in3);
        in0 = math::vActiveq_f32<Act>(in0);
        in1 = math::vActiveq_f32<Act>(in1);
        in2 = math::vActiveq_f32<Act>(in2);
        in3 = math::vActiveq_f32<Act>(in3);
        vst1q_f32(y, in0);
        vst1q_f32(y + 4, in1);
        vst1q_f32(y + 8, in2);
        vst1q_f32(y + 12, in3);
      }
      for (; j < spatial_size - 3; j += 4, x += 4, y += 4) {
        float32x4_t in0 = vld1q_f32(x);
        in0 = vmlaq_f32(__bias, __scale, in0);
        in0 = math::vActiveq_f32<Act>(in0);
        vst1q_f32(y, in0);
      }
#endif
      for (; j < spatial_size; ++j, ++x, ++y) {
        *y = math::Active<Act>(alpha * (*x) + beta);
      }
    }
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
