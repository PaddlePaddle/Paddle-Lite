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

#ifdef RELU_OP
#pragma once

#include <operators/math/transform.h>
#include "operators/op_param.h"
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {

template <typename T>
struct ReluFunctor {
  inline T operator()(T in) const { return in > 0 ? in : 0; }
};

/*
 * @b 特化到具体平台的实现, param 从 op 层传入
 * */
template <typename P>
void ReluCompute(const ReluParam<CPU> &param) {
  const auto *input_x = param.InputX();
  auto *input_x_ptr = input_x->data<float>();
  auto *out = param.Out();
  auto *out_ptr = out->mutable_data<float>();

  int numel = input_x->numel();
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#if __aarch64__
  if (numel > 0) {
    int loop = numel >> 0x4;
    int remain = numel & 0xF;
    float32x4_t zero = vdupq_n_f32(0.f);
    for (int i = 0; i < loop; ++i) {
      float32x4_t r0 = vld1q_f32(input_x_ptr);
      float32x4_t r1 = vld1q_f32(input_x_ptr + 4);
      float32x4_t r2 = vld1q_f32(input_x_ptr + 8);
      float32x4_t r3 = vld1q_f32(input_x_ptr + 12);
      r0 = vmaxq_f32(r0, zero);
      r1 = vmaxq_f32(r1, zero);
      r2 = vmaxq_f32(r2, zero);
      r3 = vmaxq_f32(r3, zero);
      vst1q_f32(out_ptr, r0);
      vst1q_f32(out_ptr + 4, r1);
      vst1q_f32(out_ptr + 8, r2);
      vst1q_f32(out_ptr + 12, r3);
      input_x_ptr += 16;
      out_ptr += 16;
    }
    for (int i = 0; i < remain; ++i) {
      out_ptr[i] = (input_x_ptr[i] > 0) * input_x_ptr[i];
    }
#else
  if (numel > 64) {
    asm volatile(
        "pld        [%[input_x_ptr], #0]        \n\t"
        "vmov.f32   q8,    #0.0                 \n\t"
        "subs %[num], %[num], #32                \n\t"
        "blt        end_num_%=                  \n\t"
        "loop_num_%=:                           \n\t"
        "pld        [%[input_x_ptr], #1024]      \n\t"

        "vld1.32 {q0, q1}, [%[input_x_ptr]]!    \n\t"
        "vld1.32 {q2, q3}, [%[input_x_ptr]]!    \n\t"
        "vld1.32 {q4, q5}, [%[input_x_ptr]]!    \n\t"
        "vld1.32 {q6, q7}, [%[input_x_ptr]]!    \n\t"

        "vmax.f32 q0, q0, q8                   \n\t"
        "vmax.f32 q1, q1, q8                    \n\t"
        "vmax.f32 q2, q2, q8                   \n\t"
        "vmax.f32 q3, q3, q8                   \n\t"
        "vmax.f32 q4, q4, q8                   \n\t"
        "vmax.f32 q5, q5, q8                   \n\t"
        "vmax.f32 q6, q6, q8                   \n\t"
        "vmax.f32 q7, q7, q8                   \n\t"

        "vst1.32 {q0, q1}, [%[out_ptr]]!        \n\t"
        "vst1.32 {q2, q3}, [%[out_ptr]]!       \n\t"
        "vst1.32 {q4, q5}, [%[out_ptr]]!       \n\t"
        "vst1.32 {q6, q7}, [%[out_ptr]]!       \n\t"

        "subs %[num], %[num], #32              \n\t"
        "bge        loop_num_%=                \n\t"
        "end_num_%=:                           \n\t"
        "cmp %[num], #0                         \n\t"
        "bge   end_%=                          \n\t"
        "mov r6, #4                             \n\t"
        "mul r5, %[num], r6                     \n\t"
        "add %[input_x_ptr], %[input_x_ptr], r5     \n\t"
        "vld1.32 {q0, q1}, [%[input_x_ptr]]!    \n\t"
        "vld1.32 {q2, q3}, [%[input_x_ptr]]!    \n\t"
        "vld1.32 {q4, q5}, [%[input_x_ptr]]!    \n\t"
        "vld1.32 {q6, q7}, [%[input_x_ptr]]!    \n\t"
        "vmax.f32 q0, q0, q8                   \n\t"
        "vmax.f32 q1, q1, q8                    \n\t"
        "vmax.f32 q2, q2, q8                   \n\t"
        "vmax.f32 q3, q3, q8                   \n\t"
        "vmax.f32 q4, q4, q8                   \n\t"
        "vmax.f32 q5, q5, q8                   \n\t"
        "vmax.f32 q6, q6, q8                   \n\t"
        "vmax.f32 q7, q7, q8                   \n\t"
        "add %[out_ptr], %[out_ptr], r5       \n\t"
        "vst1.32 {q0, q1}, [%[out_ptr]]!        \n\t"
        "vst1.32 {q2, q3}, [%[out_ptr]]!       \n\t"
        "vst1.32 {q4, q5}, [%[out_ptr]]!       \n\t"
        "vst1.32 {q6, q7}, [%[out_ptr]]!       \n\t"
        "end_%=:                                \n\t"
        :
        :
        [out_ptr] "r"(out_ptr), [input_x_ptr] "r"(input_x_ptr), [num] "r"(numel)
        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "r5",
          "r6");
#endif
  } else {
#endif
    ReluFunctor<float> func_;
    math::Transform trans;
    trans(input_x_ptr, input_x_ptr + numel, out_ptr, func_);
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  }
#endif
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
