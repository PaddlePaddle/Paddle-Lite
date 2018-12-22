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

#ifdef BATCHNORM_OP

#pragma once

#include <cmath>
#include "operators/op_param.h"
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif  // __ARM_NEON__

namespace paddle_mobile {
namespace operators {

template <typename P>
void BatchnormCompute(const BatchNormParam<CPU> &param) {
  const float epsilon = param.Epsilon();
  const float *mean_ptr = param.InputMean()->data<float>();
  const float *variance_ptr = param.InputVariance()->data<float>();
  const float *scale_ptr = param.InputScale()->data<float>();
  const float *bias_ptr = param.InputBias()->data<float>();

  const framework::Tensor *input = param.InputX();
  const float *input_ptr = input->data<float>();
  framework::Tensor *output = param.OutputY();
  float *output_ptr = output->mutable_data<float>();
  size_t spatial_size = output->dims()[2] * output->dims()[3];
  int channels = output->dims()[1];

  #pragma omp parallel for collapse(2)
  for (int batch = 0; batch < output->dims()[0]; ++batch) {
    for (int c = 0; c < channels; ++c) {
      float inv_scale = 1.f / (std::sqrt(variance_ptr[c] + epsilon));
      float bias = bias_ptr[c] - inv_scale * scale_ptr[c] * mean_ptr[c];
      float scale = inv_scale * scale_ptr[c];
      size_t offset = (batch * channels + c) * spatial_size;
      const float *x = input_ptr + offset;
      float *y = output_ptr + offset;
      size_t remain = spatial_size;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      int loop = spatial_size >> 4;
      remain = spatial_size & 0xF;
      float32x4_t __scale = vdupq_n_f32(scale);
      float32x4_t __bias = vdupq_n_f32(bias);
      for (int k = 0; k < loop; ++k, x += 16, y += 16) {
        float32x4_t r0 = vld1q_f32(x);
        float32x4_t r1 = vld1q_f32(x + 4);
        float32x4_t r2 = vld1q_f32(x + 8);
        float32x4_t r3 = vld1q_f32(x + 12);
        r0 = vmlaq_f32(__bias, __scale, r0);
        r1 = vmlaq_f32(__bias, __scale, r1);
        r2 = vmlaq_f32(__bias, __scale, r2);
        r3 = vmlaq_f32(__bias, __scale, r3);
        vst1q_f32(y, r0);
        vst1q_f32(y + 4, r1);
        vst1q_f32(y + 8, r2);
        vst1q_f32(y + 12, r3);
      }
#endif  // __ARM_NEON__
      for (int k = 0; k < remain; ++k) {
        y[k] = scale * x[k] + bias;
      }
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
