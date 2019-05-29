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

#include "operators/math/activation.h"
#include "operators/op_param.h"
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif  // __ARM_NEON__

namespace paddle_mobile {
namespace operators {

template <typename Dtype, ActivationType Act>
struct ActivationCompute {
  void operator()(const Tensor *input, Tensor *output) {}
  void operator()(const Tensor *input, Tensor *output, float alpha) {}
};

template <ActivationType Act>
struct ActivationCompute<float, Act> {
  void operator()(const Tensor *input, Tensor *output) {
    const float *x = input->data<float>();
    float *y = output->mutable_data<float>();
    size_t remain = input->numel();
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
    size_t loop = remain >> 4;
    remain = remain & 0xF;

#pragma omp parallel for
    for (size_t i = 0; i < loop; ++i) {
      const float *local_x = x + (i << 4);
      float *local_y = y + (i << 4);
      float32x4_t r0 = vld1q_f32(local_x);
      float32x4_t r1 = vld1q_f32(local_x + 4);
      float32x4_t r2 = vld1q_f32(local_x + 8);
      float32x4_t r3 = vld1q_f32(local_x + 12);
      r0 = math::vActiveq_f32<Act>(r0);
      r1 = math::vActiveq_f32<Act>(r1);
      r2 = math::vActiveq_f32<Act>(r2);
      r3 = math::vActiveq_f32<Act>(r3);
      vst1q_f32(local_y, r0);
      vst1q_f32(local_y + 4, r1);
      vst1q_f32(local_y + 8, r2);
      vst1q_f32(local_y + 12, r3);
    }
    x += (loop << 4);
    y += (loop << 4);
#endif
    for (size_t i = 0; i < remain; ++i) {
      y[i] = math::Active<Act>(x[i]);
    }
  }

  void operator()(const Tensor *input, Tensor *output, float falpha) {
    const float *x = input->data<float>();
    float *y = output->mutable_data<float>();
    size_t remain = input->numel();
    float alphas[4] = {falpha, falpha, falpha, falpha};
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
    size_t loop = remain >> 4;
    remain = remain & 0xF;

#pragma omp parallel for
    for (size_t i = 0; i < loop; ++i) {
      const float *local_x = x + (i << 4);
      float *local_y = y + (i << 4);
      float32x4_t r0 = vld1q_f32(local_x);
      float32x4_t r1 = vld1q_f32(local_x + 4);
      float32x4_t r2 = vld1q_f32(local_x + 8);
      float32x4_t r3 = vld1q_f32(local_x + 12);
      float32x4_t a_r0 = vld1q_f32(alphas);
      float32x4_t a_r1 = vld1q_f32(alphas);
      float32x4_t a_r2 = vld1q_f32(alphas);
      float32x4_t a_r3 = vld1q_f32(alphas);
      r0 = math::vActiveq_f32<Act>(r0, a_r0);
      r1 = math::vActiveq_f32<Act>(r1, a_r1);
      r2 = math::vActiveq_f32<Act>(r2, a_r2);
      r3 = math::vActiveq_f32<Act>(r3, a_r3);
      vst1q_f32(local_y, r0);
      vst1q_f32(local_y + 4, r1);
      vst1q_f32(local_y + 8, r2);
      vst1q_f32(local_y + 12, r3);
    }
    x += (loop << 4);
    y += (loop << 4);
#endif
    for (size_t i = 0; i < remain; ++i) {
      y[i] = math::Active<Act>(x[i], falpha);
    }
  }
};

}  // namespace operators
}  // namespace paddle_mobile
