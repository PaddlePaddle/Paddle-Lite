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

#include "operators/math/elementwise_op_function.h"
#include "operators/op_param.h"
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {

template <typename T>
struct AddFunctor {
  inline T operator()(T a, T b) const { return a + b; }
};

template <typename P>
void ElementwiseAddCompute(const ElementwiseAddParam<CPU> &param) {
  const Tensor *input_x = param.InputX();
  const Tensor *input_y = param.InputY();
  Tensor *Out = param.Out();
  Out->mutable_data<float>();
  int axis = param.Axis();
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  size_t batch = 1;
  size_t elementwise_num = 1;
  for (int i = 0; i < axis; ++i) {
    batch *= input_x->dims()[i];
  }
  for (int i = axis + 1; i < input_x->dims().size(); ++i) {
    elementwise_num *= input_x->dims()[i];
  }
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < input_x->dims()[axis]; ++j) {
      size_t offset = (i * input_x->dims()[axis] + j) * elementwise_num;
      const float *input = input_x->data<float>() + offset;
      const float *bias = input_y->data<float>() + j;
      float *output = Out->mutable_data<float>() + offset;

      int loop = elementwise_num >> 0x4;
      int remain = elementwise_num & 0xF;
      for (int k = 0; k < loop; ++k) {
        float32x4_t rb = vdupq_n_f32(*bias);
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
      for (int k = 0; k < remain; ++k) {
        output[k] = input[k] + *bias;
      }
    }
  }
#else
  ElementwiseComputeEx<AddFunctor<float>, float>(input_x, input_y, axis,
                                                 AddFunctor<float>(), Out);
#endif
}

template class ElementwiseAddKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
