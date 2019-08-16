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

#ifdef LRN_OP

#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#include "operators/math/math.h"
#endif
#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename T>
struct LRNFunctor {
  void operator()(const framework::Tensor &input, framework::Tensor *out, int N,
                  int C, int H, int W, int n, float k, float alpha,
                  float beta) {
    const float *input_ptr = input.data<float>();
    const int start = -(n - 1) / 2;
    const int end = start + n;
    auto out_ptr = out->data<T>();

    const int stride0 = C * H * W;
    const int stride1 = H * W;
    const int stride2 = W;
    framework::Tensor sqr_buffer;
    auto sqr_buffer_ptr = sqr_buffer.mutable_data<float>(input.dims());
    std::fill(sqr_buffer_ptr, sqr_buffer_ptr + sqr_buffer.numel(), 0.0);

    for (int a = 0; a < N; a++) {
#pragma parallel for
      for (int b = 0; b < C; b++) {
        for (int index = start; index < end; index++) {
          int channel = b + index;
          if (channel >= 0 && channel < C) {
            int tmp_s = a * stride0 + b * stride1;
            int tmp_c = a * stride0 + channel * stride1;
#ifdef __ARM_NEON
            int n4 = stride1 / 4;
            int m4 = stride1 % 4;
            float32x4_t sqr0;
            float32x4_t in0;
            float32x4_t res0;
            for (int i = 0; i < n4; i++) {
              sqr0 = vld1q_f32(sqr_buffer_ptr + tmp_s);
              in0 = vld1q_f32(input_ptr + tmp_c);

              res0 = vmlaq_f32(sqr0, in0, in0);
              vst1q_f32(sqr_buffer_ptr + tmp_s, res0);

              tmp_s += 4;
              tmp_c += 4;
            }

            for (int i = 0; i < m4; i++) {
              int s_i = tmp_s + i;
              int c_i = tmp_c + i;
              sqr_buffer_ptr[s_i] += input_ptr[c_i] * input_ptr[c_i];
            }

#else
            for (int tmp = 0; tmp < stride1; tmp++) {
              int s_i = tmp_s + tmp;
              int c_i = tmp_c + tmp;
              sqr_buffer_ptr[s_i] += input_ptr[c_i] * input_ptr[c_i];
            }
#endif
          }
        }
      }
    }

#ifdef __ARM_NEON

    float32x4_t sqr1, sqr2, sqr3, sqr4;
    float32x4_t alpha4;
    float32x4_t k4;
    float32x4_t beta4;
    float32x4_t res1, res2, res3, res4;
    float32x4_t in1, in2, in3, in4;

    beta4 = vdupq_n_f32(beta);
    alpha4 = vdupq_n_f32(alpha);
    k4 = vdupq_n_f32(k);
    auto out_tmp_ptr = out_ptr;

    int n16 = input.numel() / 16;
    int m16 = input.numel() % 16;
    int m16n4 = m16 / 4;
    int m16m4 = m16 % 4;

    for (int i = 0; i < n16; i++) {
      sqr1 = vld1q_f32(sqr_buffer_ptr);
      sqr2 = vld1q_f32(sqr_buffer_ptr + 4);
      sqr3 = vld1q_f32(sqr_buffer_ptr + 8);
      sqr4 = vld1q_f32(sqr_buffer_ptr + 12);

      in1 = vld1q_f32(input_ptr);
      in2 = vld1q_f32(input_ptr + 4);
      in3 = vld1q_f32(input_ptr + 8);
      in4 = vld1q_f32(input_ptr + 12);

      sqr1 = vmlaq_f32(k4, sqr1, alpha4);
      sqr2 = vmlaq_f32(k4, sqr2, alpha4);
      sqr3 = vmlaq_f32(k4, sqr3, alpha4);
      sqr4 = vmlaq_f32(k4, sqr4, alpha4);

      sqr1 = pow_ps(sqr1, -beta4);
      sqr2 = pow_ps(sqr2, -beta4);
      sqr3 = pow_ps(sqr3, -beta4);
      sqr4 = pow_ps(sqr4, -beta4);

      sqr1 = vmulq_f32(sqr1, in1);
      sqr2 = vmulq_f32(sqr2, in2);
      sqr3 = vmulq_f32(sqr3, in3);
      sqr4 = vmulq_f32(sqr4, in4);

      vst1q_f32(out_tmp_ptr, sqr1);
      vst1q_f32(out_tmp_ptr + 4, sqr2);
      vst1q_f32(out_tmp_ptr + 8, sqr3);
      vst1q_f32(out_tmp_ptr + 12, sqr4);

      sqr_buffer_ptr += 4 * 4;
      input_ptr += 4 * 4;
      out_tmp_ptr += 4 * 4;
    }
    for (int i = 0; i < m16n4; i++) {
      sqr4 = vld1q_f32(sqr_buffer_ptr);
      in4 = vld1q_f32(input_ptr);
      sqr4 = vmlaq_f32(k4, sqr4, alpha4);
      sqr4 = pow_ps(sqr4, -beta4);
      sqr4 = vmulq_f32(sqr4, in4);
      vst1q_f32(out_tmp_ptr, sqr4);
      sqr_buffer_ptr += 4;
      input_ptr += 4;
      out_tmp_ptr += 4;
    }

    for (int i = 0; i < m16m4; i++) {
      out_tmp_ptr[i] = input_ptr[i] / pow(k + alpha * sqr_buffer_ptr[i], beta);
    }

#else
    for (int i = 0; i < input.numel(); i++) {
      out_ptr[i] = input_ptr[i] / pow(k + alpha * sqr_buffer_ptr[i], beta);
    }
#endif
  }
};

template <typename DeviceType, typename T>
class LrnKernel
    : public framework::OpKernelBase<DeviceType, LrnParam<DeviceType>> {
 public:
  void Compute(const LrnParam<DeviceType> &param);
  bool Init(LrnParam<DeviceType> *param);
};
}  // namespace operators
}  // namespace paddle_mobile

#endif
