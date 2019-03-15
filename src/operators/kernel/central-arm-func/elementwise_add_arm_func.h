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

#ifdef ELEMENTWISEADD_OP

#pragma once

#include "framework/context.h"
#include "operators/math/elementwise_op_function.h"
#include "operators/op_param.h"
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {

template <typename T>
inline void ElementwiseAddCompute(const ElementwiseAddParam<CPU> &param) {
  const framework::Tensor *input_x = param.InputX();
  const framework::Tensor *input_y = param.InputY();
  framework::Tensor *Out = param.Out();
  int axis = param.Axis();

  const auto &x_dims = input_x->dims();
  const auto &y_dims = input_y->dims();
  /// axis = -1 represent the last dimensions.
  axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);
  size_t batch = 1;
  size_t channels = 1;
  size_t elementwise_num = 1;
  for (int i = 0; i < axis; ++i) {
    batch *= x_dims[i];
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    channels *= y_dims[i];
  }
  for (int i = y_dims.size() + axis; i < x_dims.size(); ++i) {
    elementwise_num *= x_dims[i];
  }
  const float *bias_data = input_y->data<float>();
  const float *input_data = input_x->data<float>();
  float *output_data = Out->mutable_data<float>();

  #pragma omp parallel for collapse(2)
  // num_threads(framework::threads())
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      size_t offset = (i * channels + j) * elementwise_num;
      const float *input = input_data + offset;
      const float bias = bias_data[j];
      float *output = output_data + offset;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      int loop = elementwise_num >> 0x4;
      int remain = elementwise_num & 0xF;
      float32x4_t rb = vdupq_n_f32(bias);
      for (int k = 0; k < loop; ++k) {
        float32x4_t r0 = vld1q_f32(input);
        float32x4_t r1 = vld1q_f32(input + 4);
        float32x4_t r2 = vld1q_f32(input + 8);
        float32x4_t r3 = vld1q_f32(input + 12);
        r0 = vaddq_f32(r0, rb);
        r1 = vaddq_f32(r1, rb);
        r2 = vaddq_f32(r2, rb);
        r3 = vaddq_f32(r3, rb);
        vst1q_f32(output, r0);
        vst1q_f32(output + 4, r1);
        vst1q_f32(output + 8, r2);
        vst1q_f32(output + 12, r3);
        input += 16;
        output += 16;
      }
      if (remain >= 8) {
        float32x4_t r0 = vld1q_f32(input);
        float32x4_t r1 = vld1q_f32(input + 4);
        r0 = vaddq_f32(r0, rb);
        r1 = vaddq_f32(r1, rb);
        vst1q_f32(output, r0);
        vst1q_f32(output + 4, r1);
        input += 8;
        output += 8;
        remain -= 8;
      }
      if (remain >= 4) {
        float32x4_t r0 = vld1q_f32(input);
        r0 = vaddq_f32(r0, rb);
        vst1q_f32(output, r0);
        input += 4;
        output += 4;
        remain -= 4;
      }
      if (remain > 0) {
        float32x4_t r0 = vld1q_f32(input);
        r0 = vaddq_f32(r0, rb);
        switch (remain) {
          case 1:
            vst1q_lane_f32(output, r0, 0);
            break;
          case 2:
            vst1_f32(output, vget_low_f32(r0));
            break;
          case 3:
            vst1_f32(output, vget_low_f32(r0));
            vst1q_lane_f32(output, r0, 2);
            break;
        }
      }
#else
      for (int k = 0; k < elementwise_num; ++k) {
        output[k] = input[k] + bias;
      }
#endif  // __ARM_NEON__
    }
  }
}

template class ElementwiseAddKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
