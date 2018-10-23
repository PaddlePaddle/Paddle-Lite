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

#ifdef CONV_OP

#include "operators/math/conv_arm_int8.h"

namespace paddle_mobile {
namespace operators {

void conv3x3s1_int8(const framework::Tensor& input,
                    const framework::Tensor& weight,
                    framework::Tensor* output) {
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  const int8_t* in_data = input.data<int8_t>();
  const int8_t* w_data = weight.data<int8_t>();
  int32_t* out_data = output->mutable_data<int32_t>();
  // make sure that batch size is 1
  int input_c = input.dims()[1];
  int input_h = input.dims()[2];
  int input_w = input.dims()[3];
  int output_c = output->dims()[1];
  int output_h = output->dims()[2];
  int output_w = output->dims()[3];
  int image_size = input_h * input_w;
  int out_image_size = output_h * output_w;
  memset(out_data, 0, output_c * out_image_size * sizeof(int32_t));
#if __aarch64__
  // TODO(hjchen2)
#else
  int oc = 0;
  #pragma omp parallel for
  for (; oc < output_c - 1; oc += 2) {
    for (int ic = 0; ic < input_c; ++ic) {
      const int8_t* kernel0 = w_data + (oc * input_c + ic) * 9;
      const int8_t* kernel1 = w_data + ((oc + 1) * input_c + ic) * 9;
      int32_t* output0 = out_data + oc * out_image_size;
      int32_t* output0n = output0 + output_w;
      int32_t* output1 = out_data + (oc + 1) * out_image_size;
      int32_t* output1n = output1 + output_w;

      int oh = 0;
      for (; oh < output_h - 1; oh += 2) {
        const int8_t* r0 = in_data + ic * image_size + oh * input_w;
        const int8_t* r1 = r0 + input_w;
        const int8_t* r2 = r1 + input_w;
        const int8_t* r3 = r2 + input_w;

        int ow = output_w >> 3;
        int remain = output_w & 0x7;
        if (ow > 0) {
          asm volatile(
              "vld1.8     {d0}, [%[kernel0]]        \n"
              "ldr        r5,   [%[kernel0], #8]    \n"
              "vld1.8     {d1}, [%[kernel1]]        \n"
              "ldr        r6,   [%[kernel1], #8]    \n"

              "0:                                   \n"
              "vld1.8     {d2-d3}, [%[r0]]          \n"  // r0
              "add        %[r0], #8                 \n"
              "vext.8     d4, d2, d3, #1            \n"
              "vext.8     d5, d2, d3, #2            \n"
              "vdup.s8    d6, d0[0]                 \n"
              "vdup.s8    d7, d0[1]                 \n"
              "vdup.s8    d8, d0[2]                 \n"
              "vdup.s8    d9, d1[0]                 \n"
              "vdup.s8    d10, d1[1]                \n"
              "vdup.s8    d11, d1[2]                \n"
              "vmull.s8   q6, d2, d6                \n"
              "vmull.s8   q7, d4, d7                \n"
              "vmlal.s8   q6, d5, d8                \n"
              "vaddl.s16  q12, d12, d14             \n"
              "vaddl.s16  q13, d13, d15             \n"
              "vmull.s8   q6, d2, d9                \n"
              "vmull.s8   q7, d4, d10               \n"
              "vmlal.s8   q6, d5, d11               \n"
              "vaddl.s16  q14, d12, d14             \n"
              "vaddl.s16  q15, d13, d15             \n"

              "vld1.8     {d2-d3}, [%[r1]]          \n"  // r1
              "add        %[r1], #8                 \n"
              "vext.8     d4, d2, d3, #1            \n"
              "vext.8     d5, d2, d3, #2            \n"

              "vmull.s8   q6, d2, d6                \n"  // next row
              "vmull.s8   q7, d4, d7                \n"
              "vmlal.s8   q6, d5, d8                \n"
              "vaddl.s16  q8, d12, d14              \n"
              "vaddl.s16  q9, d13, d15              \n"
              "vmull.s8   q6, d2, d9                \n"
              "vmull.s8   q7, d4, d10               \n"
              "vmlal.s8   q6, d5, d11               \n"
              "vaddl.s16  q10, d12, d14             \n"
              "vaddl.s16  q11, d13, d15             \n"

              "vdup.s8    d6, d0[3]                 \n"
              "vdup.s8    d7, d0[4]                 \n"
              "vdup.s8    d8, d0[5]                 \n"
              "vdup.s8    d9, d1[3]                 \n"
              "vdup.s8    d10, d1[4]                \n"
              "vdup.s8    d11, d1[5]                \n"
              "vmull.s8   q6, d2, d6                \n"
              "vmull.s8   q7, d4, d7                \n"
              "vmlal.s8   q6, d5, d8                \n"
              "vaddw.s16  q12, q12, d12             \n"
              "vaddw.s16  q13, q13, d13             \n"
              "vaddw.s16  q12, q12, d14             \n"
              "vaddw.s16  q13, q13, d15             \n"
              "vmull.s8   q6, d2, d9                \n"
              "vmull.s8   q7, d4, d10               \n"
              "vmlal.s8   q6, d5, d11               \n"
              "vaddw.s16  q14, q14, d12             \n"
              "vaddw.s16  q15, q15, d13             \n"
              "vaddw.s16  q14, q14, d14             \n"
              "vaddw.s16  q15, q15, d15             \n"

              "vld1.8     {d2-d3}, [%[r2]]          \n"  // r2
              "add        %[r2], #8                 \n"
              "vext.8     d4, d2, d3, #1            \n"
              "vext.8     d5, d2, d3, #2            \n"

              "vmull.s8   q6, d2, d6                \n"  // next row
              "vmull.s8   q7, d4, d7                \n"
              "vmlal.s8   q6, d5, d8                \n"
              "vaddw.s16  q8, q8, d12               \n"
              "vaddw.s16  q8, q8, d14               \n"
              "vaddw.s16  q9, q9, d13               \n"
              "vaddw.s16  q9, q9, d15               \n"
              "vmull.s8   q6, d2, d9                \n"
              "vmull.s8   q7, d4, d10               \n"
              "vmlal.s8   q6, d5, d11               \n"
              "vaddw.s16  q10, q10, d12             \n"
              "vaddw.s16  q11, q11, d13             \n"
              "vaddw.s16  q10, q10, d14             \n"
              "vaddw.s16  q11, q11, d15             \n"

              "vdup.s8    d6, d0[6]                 \n"
              "vdup.s8    d7, d0[7]                 \n"
              "vdup.s8    d8, r5                    \n"
              "vdup.s8    d9, d1[6]                 \n"
              "vdup.s8    d10, d1[7]                \n"
              "vdup.s8    d11, r6                   \n"
              "vmull.s8   q6, d2, d6                \n"
              "vmull.s8   q7, d4, d7                \n"
              "vmlal.s8   q6, d5, d8                \n"
              "vaddw.s16  q12, q12, d12             \n"
              "vaddw.s16  q13, q13, d13             \n"
              "vaddw.s16  q12, q12, d14             \n"
              "vaddw.s16  q13, q13, d15             \n"

              "vld1.32    {d12-d15}, [%[output0]]   \n"
              "vadd.s32   q6, q6, q12               \n"
              "vadd.s32   q7, q7, q13               \n"
              "vst1.32    {d12-d15}, [%[output0]]!  \n"

              "vmull.s8   q6, d2, d9                \n"
              "vmull.s8   q7, d4, d10               \n"
              "vmlal.s8   q6, d5, d11               \n"
              "vaddw.s16  q14, q14, d12             \n"
              "vaddw.s16  q15, q15, d13             \n"
              "vaddw.s16  q14, q14, d14             \n"
              "vaddw.s16  q15, q15, d15             \n"

              "vld1.32    {d12-d15}, [%[output1]]   \n"
              "vadd.s32   q6, q6, q14               \n"
              "vadd.s32   q7, q7, q15               \n"
              "vst1.32    {d12-d15}, [%[output1]]!  \n"

              "vld1.8     {d2-d3}, [%[r3]]          \n"  // r3
              "add        %[r3], #8                 \n"
              "vext.8     d4, d2, d3, #1            \n"
              "vext.8     d5, d2, d3, #2            \n"

              "vmull.s8   q6, d2, d6                \n"  // next row
              "vmull.s8   q7, d4, d7                \n"
              "vmlal.s8   q6, d5, d8                \n"
              "vaddw.s16  q8, q8, d12               \n"
              "vaddw.s16  q9, q9, d15               \n"
              "vaddw.s16  q8, q8, d14               \n"
              "vaddw.s16  q9, q9, d13               \n"

              "vld1.32    {d12-d15}, [%[output0n]]  \n"
              "vadd.s32   q6, q6, q8                \n"
              "vadd.s32   q7, q7, q9                \n"
              "vst1.32    {d12-d15}, [%[output0n]]! \n"

              "vmull.s8   q6, d2, d9                \n"
              "vmull.s8   q7, d4, d10               \n"
              "vmlal.s8   q6, d5, d11               \n"
              "vaddw.s16  q10, q10, d12             \n"
              "vaddw.s16  q11, q11, d15             \n"
              "vaddw.s16  q10, q10, d14             \n"
              "vaddw.s16  q11, q11, d13             \n"

              "vld1.32    {d12-d15}, [%[output1n]]  \n"
              "vadd.s32   q6, q6, q10               \n"
              "vadd.s32   q7, q7, q11               \n"
              "vst1.32    {d12-d15}, [%[output1n]]! \n"

              "subs       %[ow], #1                 \n"
              "bne        0b                        \n"
              : [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), [r3] "+r"(r3),
                [ow] "+r"(ow), [output0] "+r"(output0), [output1] "+r"(output1),
                [output0n] "+r"(output0n), [output1n] "+r"(output1n)
              : [kernel0] "r"(kernel0), [kernel1] "r"(kernel1)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "r5",
                "r6");
        }
        if (remain > 0) {
          asm volatile(
              "vld1.8     {d0}, [%[kernel0]]       \n"
              "ldr        r5,   [%[kernel0], #8]   \n"
              "vld1.8     {d1}, [%[kernel1]]       \n"
              "ldr        r6,   [%[kernel1], #8]   \n"

              "0:                                  \n"
              "vld1.8     d4, [%[r0]]              \n"
              "vld1.8     d5, [%[r1]]              \n"
              "vld1.8     d6, [%[r2]]              \n"
              "vld1.8     d7, [%[r3]]              \n"
              "add        %[r0], #1                \n"
              "add        %[r1], #1                \n"
              "add        %[r2], #1                \n"
              "add        %[r3], #1                \n"
              "vdup.s8     d2, r5                  \n"
              "vdup.s8     d3, r6                  \n"
              "vext.8     d8, d0, d2, #3           \n"
              "vext.8     d9, d0, d2, #6           \n"
              "vext.8     d10, d1, d3, #3          \n"
              "vext.8     d11, d1, d3, #6          \n"

              "vmull.s8   q6, d4, d0               \n"
              "vmull.s8   q7, d5, d8               \n"
              "vmlal.s8   q6, d6, d9               \n"
              "vaddl.s16  q12, d12, d14            \n"
              "vdup.s32   d2, d24[1]               \n"
              "vadd.s32   d24, d24, d2             \n"
              "vadd.s32   d24, d24, d25            \n"
              "vmull.s8   q6, d4, d1               \n"
              "vmull.s8   q7, d5, d10              \n"
              "vmlal.s8   q6, d6, d11              \n"
              "vaddl.s16  q13, d12, d14            \n"
              "vdup.s32   d2, d26[1]               \n"
              "vadd.s32   d26, d26, d2             \n"
              "vadd.s32   d26, d26, d27            \n"

              "ldr        r7, [%[output0]]         \n"
              "vdup.s32   d14, r7                  \n"
              "vadd.s32   d14, d14, d24            \n"
              "vst1.32    d14[0], [%[output0]]!    \n"
              "ldr        r7, [%[output1]]         \n"
              "vdup.s32   d14, r7                  \n"
              "vadd.s32   d14, d14, d26            \n"
              "vst1.32    d14[0], [%[output1]]!    \n"

              "vmull.s8   q6, d5, d0               \n"
              "vmull.s8   q7, d6, d8               \n"
              "vmlal.s8   q6, d7, d9               \n"
              "vaddl.s16  q12, d12, d14            \n"
              "vdup.s32   d2, d24[1]               \n"
              "vadd.s32   d24, d24, d2             \n"
              "vadd.s32   d24, d24, d25            \n"
              "vmull.s8   q6, d5, d1               \n"
              "vmull.s8   q7, d6, d10              \n"
              "vmlal.s8   q6, d7, d11              \n"
              "vaddl.s16  q13, d12, d14            \n"
              "vdup.s32   d2, d26[1]               \n"
              "vadd.s32   d26, d26, d2             \n"
              "vadd.s32   d26, d26, d27            \n"

              "ldr        r7, [%[output0n]]        \n"
              "vdup.s32   d14, r7                  \n"
              "vadd.s32   d14, d14, d24            \n"
              "vst1.32    d14[0], [%[output0n]]!   \n"
              "ldr        r7, [%[output1n]]        \n"
              "vdup.s32   d14, r7                  \n"
              "vadd.s32   d14, d14, d26            \n"
              "vst1.32    d14[0], [%[output1n]]!   \n"

              "subs       %[remain], #1            \n"
              "bne        0b                       \n"
              : [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), [r3] "+r"(r3),
                [remain] "+r"(remain), [output0] "+r"(output0),
                [output1] "+r"(output1), [output0n] "+r"(output0n),
                [output1n] "+r"(output1n)
              : [kernel0] "r"(kernel0), [kernel1] "r"(kernel1)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "r5", "r6", "r7");
        }
        output0 += output_w;
        output1 += output_w;
        output0n += output_w;
        output1n += output_w;
      }
      // remain output height
      for (; oh < output_h; ++oh) {
        const int8_t* r0 = in_data + ic * image_size + oh * input_w;
        const int8_t* r1 = r0 + input_w;
        const int8_t* r2 = r1 + input_w;
        const int8_t* r3 = r2 + input_w;
        const int8_t* r4 = r3 + input_w;

        int ow = output_w >> 3;
        int remain = output_w & 0x7;
        if (ow > 0) {
          asm volatile(
              "vld1.8     {d0}, [%[kernel0]]        \n"
              "ldr        r5,   [%[kernel0], #8]    \n"
              "vld1.8     {d1}, [%[kernel1]]        \n"
              "ldr        r6,   [%[kernel1], #8]    \n"

              "0:                                   \n"
              "vld1.8     {d2-d3}, [%[r0]]          \n"  // r0
              "add        %[r0], #8                 \n"
              "vext.8     d4, d2, d3, #1            \n"
              "vext.8     d5, d2, d3, #2            \n"
              "vdup.s8    d6, d0[0]                 \n"
              "vdup.s8    d7, d0[1]                 \n"
              "vdup.s8    d8, d0[2]                 \n"
              "vdup.s8    d9, d1[0]                 \n"
              "vdup.s8    d10, d1[1]                \n"
              "vdup.s8    d11, d1[2]                \n"
              "vmull.s8   q6, d2, d6                \n"
              "vmull.s8   q7, d4, d7                \n"
              "vmlal.s8   q6, d5, d8                \n"
              "vaddl.s16  q12, d12, d14             \n"
              "vaddl.s16  q13, d13, d15             \n"
              "vmull.s8   q6, d2, d9                \n"
              "vmull.s8   q7, d4, d10               \n"
              "vmlal.s8   q6, d5, d11               \n"
              "vaddl.s16  q14, d12, d14             \n"
              "vaddl.s16  q15, d13, d15             \n"

              "vld1.8     {d2-d3}, [%[r1]]          \n"  // r1
              "add        %[r1], #8                 \n"
              "vext.8     d4, d2, d3, #1            \n"
              "vext.8     d5, d2, d3, #2            \n"
              "vdup.s8    d6, d0[3]                 \n"
              "vdup.s8    d7, d0[4]                 \n"
              "vdup.s8    d8, d0[5]                 \n"
              "vdup.s8    d9, d1[3]                 \n"
              "vdup.s8    d10, d1[4]                \n"
              "vdup.s8    d11, d1[5]                \n"
              "vmull.s8   q6, d2, d6                \n"
              "vmull.s8   q7, d4, d7                \n"
              "vmlal.s8   q6, d5, d8                \n"
              "vaddw.s16  q12, q12, d12             \n"
              "vaddw.s16  q12, q12, d14             \n"
              "vaddw.s16  q13, q13, d13             \n"
              "vaddw.s16  q13, q13, d15             \n"
              "vmull.s8   q6, d2, d9                \n"
              "vmull.s8   q7, d4, d10               \n"
              "vmlal.s8   q6, d5, d11               \n"
              "vaddw.s16  q14, q14, d12             \n"
              "vaddw.s16  q14, q14, d14             \n"
              "vaddw.s16  q15, q15, d13             \n"
              "vaddw.s16  q15, q15, d15             \n"

              "vld1.8     {d2-d3}, [%[r2]]          \n"  // r2
              "add        %[r2], #8                 \n"
              "vext.8     d4, d2, d3, #1            \n"
              "vext.8     d5, d2, d3, #2            \n"
              "vdup.s8    d6, d0[6]                 \n"
              "vdup.s8    d7, d0[7]                 \n"
              "vdup.s8    d8, r5                    \n"
              "vdup.s8    d9, d1[6]                 \n"
              "vdup.s8    d10, d1[7]                \n"
              "vdup.s8    d11, r6                   \n"
              "vmull.s8   q6, d2, d6                \n"
              "vmull.s8   q7, d4, d7                \n"
              "vmlal.s8   q6, d5, d8                \n"
              "vaddw.s16  q12, q12, d12             \n"
              "vaddw.s16  q12, q12, d14             \n"
              "vaddw.s16  q13, q13, d13             \n"
              "vaddw.s16  q13, q13, d15             \n"
              "vmull.s8   q6, d2, d9                \n"
              "vmull.s8   q7, d4, d10               \n"
              "vmlal.s8   q6, d5, d11               \n"
              "vaddw.s16  q14, q14, d12             \n"
              "vaddw.s16  q14, q14, d14             \n"
              "vaddw.s16  q15, q15, d13             \n"
              "vaddw.s16  q15, q15, d15             \n"

              "vld1.32    {d12-d15}, [%[output0]]   \n"
              "vadd.s32   q6, q6, q12               \n"
              "vadd.s32   q7, q7, q13               \n"
              "vst1.32    {d12-d15}, [%[output0]]!  \n"
              "vld1.32    {d12-d15}, [%[output1]]  \n"
              "vadd.s32   q6, q6, q14               \n"
              "vadd.s32   q7, q7, q15               \n"
              "vst1.32    {d12-d15}, [%[output1]]! \n"

              "subs       %[ow], #1                 \n"
              "bne        0b                        \n"
              : [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), [ow] "+r"(ow),
                [output0] "+r"(output0), [output1] "+r"(output1)
              : [kernel0] "r"(kernel0), [kernel1] "r"(kernel1)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "r5",
                "r6");
        }

        if (remain > 0) {
          asm volatile(
              "vld1.8     {d0}, [%[kernel0]]       \n"
              "ldr        r5,   [%[kernel0], #8]   \n"
              "vld1.8     {d1}, [%[kernel1]]       \n"
              "ldr        r6,   [%[kernel1], #8]   \n"

              "0:                                  \n"
              "vld1.8     d4, [%[r0]]              \n"
              "vld1.8     d5, [%[r1]]              \n"
              "vld1.8     d6, [%[r2]]              \n"
              "add        %[r0], #1                \n"
              "add        %[r1], #1                \n"
              "add        %[r2], #1                \n"
              "vdup.s8     d2, r5                  \n"
              "vdup.s8     d3, r6                  \n"
              "vext.8     d8, d0, d2, #3           \n"
              "vext.8     d9, d0, d2, #6           \n"
              "vext.8     d10, d1, d3, #3          \n"
              "vext.8     d11, d1, d3, #6          \n"

              "vmull.s8   q6, d4, d0               \n"
              "vmull.s8   q7, d5, d8               \n"
              "vmlal.s8   q6, d6, d9               \n"
              "vaddl.s16  q12, d12, d14            \n"
              "vdup.s32   d2, d24[1]               \n"
              "vadd.s32   d24, d24, d2             \n"
              "vadd.s32   d24, d24, d25            \n"
              "vmull.s8   q6, d4, d1               \n"
              "vmull.s8   q7, d5, d10              \n"
              "vmlal.s8   q6, d6, d11              \n"
              "vaddl.s16  q13, d12, d14            \n"
              "vdup.s32   d2, d26[1]               \n"
              "vadd.s32   d26, d26, d2             \n"
              "vadd.s32   d26, d26, d27            \n"

              "ldr        r7, [%[output0]]         \n"
              "vdup.s32   d14, r7                  \n"
              "vadd.s32   d14, d14, d24            \n"
              "vst1.32    d14[0], [%[output0]]!    \n"
              "ldr        r7, [%[output1]]         \n"
              "vdup.s32   d14, r7                  \n"
              "vadd.s32   d14, d14, d26            \n"
              "vst1.32    d14[0], [%[output1]]!    \n"

              "subs       %[remain], #1            \n"
              "bne        0b                       \n"
              : [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2),
                [remain] "+r"(remain), [output0] "+r"(output0),
                [output1] "+r"(output1)
              : [kernel0] "r"(kernel0), [kernel1] "r"(kernel1)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "r5", "r6", "r7");
        }
      }
    }
  }

  for (; oc < output_c; ++oc) {
    for (int ic = 0; ic < input_c; ++ic) {
      const int8_t* kernel0 = w_data + (oc * input_c + ic) * 9;
      int32_t* output0 = out_data + oc * out_image_size;
      int32_t* output0n = output0 + output_w;

      int oh = 0;
      for (; oh < output_h - 1; oh += 2) {
        const int8_t* r0 = in_data + ic * image_size + oh * input_w;
        const int8_t* r1 = r0 + input_w;
        const int8_t* r2 = r1 + input_w;
        const int8_t* r3 = r2 + input_w;

        int ow = output_w >> 3;
        int remain = output_w & 0x7;
        if (ow > 0) {
          asm volatile(
              "vld1.8     {d0}, [%[kernel0]]        \n"
              "ldr        r5,   [%[kernel0], #8]    \n"

              "0:                                   \n"
              "vld1.8     {d2-d3}, [%[r0]]          \n"  // r0
              "add        %[r0], #8                 \n"
              "vext.8     d4, d2, d3, #1            \n"
              "vext.8     d5, d2, d3, #2            \n"
              "vdup.s8    d6, d0[0]                 \n"
              "vdup.s8    d7, d0[1]                 \n"
              "vdup.s8    d8, d0[2]                 \n"
              "vmull.s8   q6, d2, d6                \n"
              "vmull.s8   q7, d4, d7                \n"
              "vmlal.s8   q6, d5, d8                \n"
              "vaddl.s16  q12, d12, d14             \n"
              "vaddl.s16  q13, d13, d15             \n"

              "vld1.8     {d2-d3}, [%[r1]]          \n"  // r1
              "add        %[r1], #8                 \n"
              "vext.8     d4, d2, d3, #1            \n"
              "vext.8     d5, d2, d3, #2            \n"

              "vmull.s8   q6, d2, d6                \n"  // next row
              "vmull.s8   q7, d4, d7                \n"
              "vmlal.s8   q6, d5, d8                \n"
              "vaddl.s16  q8, d12, d14              \n"
              "vaddl.s16  q9, d13, d15              \n"

              "vdup.s8    d6, d0[3]                 \n"
              "vdup.s8    d7, d0[4]                 \n"
              "vdup.s8    d8, d0[5]                 \n"
              "vmull.s8   q6, d2, d6                \n"
              "vmull.s8   q7, d4, d7                \n"
              "vmlal.s8   q6, d5, d8                \n"
              "vaddw.s16  q12, q12, d12             \n"
              "vaddw.s16  q12, q12, d14             \n"
              "vaddw.s16  q13, q13, d13             \n"
              "vaddw.s16  q13, q13, d15             \n"

              "vld1.8     {d2-d3}, [%[r2]]          \n"  // r2
              "add        %[r2], #8                 \n"
              "vext.8     d4, d2, d3, #1            \n"
              "vext.8     d5, d2, d3, #2            \n"

              "vmull.s8   q6, d2, d6                \n"  // next row
              "vmull.s8   q7, d4, d7                \n"
              "vmlal.s8   q6, d5, d8                \n"
              "vaddw.s16  q8, q8, d12               \n"
              "vaddw.s16  q8, q8, d14               \n"
              "vaddw.s16  q9, q9, d13               \n"
              "vaddw.s16  q9, q9, d15               \n"

              "vdup.s8    d6, d0[6]                 \n"
              "vdup.s8    d7, d0[7]                 \n"
              "vdup.s8    d8, r5                    \n"
              "vmull.s8   q6, d2, d6                \n"
              "vmull.s8   q7, d4, d7                \n"
              "vmlal.s8   q6, d5, d8                \n"
              "vaddw.s16  q12, q12, d12             \n"
              "vaddw.s16  q12, q12, d14             \n"
              "vaddw.s16  q13, q13, d13             \n"
              "vaddw.s16  q13, q13, d15             \n"

              "vld1.32    {d12-d15}, [%[output0]]   \n"
              "vadd.s32   q6, q6, q12               \n"
              "vadd.s32   q7, q7, q13               \n"
              "vst1.32    {d12-d15}, [%[output0]]!  \n"

              "vld1.8     {d2-d3}, [%[r3]]          \n"  // r3
              "add        %[r3], #8                 \n"
              "vext.8     d4, d2, d3, #1            \n"
              "vext.8     d5, d2, d3, #2            \n"

              "vmull.s8   q6, d2, d6                \n"  // next row
              "vmull.s8   q7, d4, d7                \n"
              "vmlal.s8   q6, d5, d8                \n"
              "vaddw.s16  q8, q8, d12               \n"
              "vaddw.s16  q8, q8, d14               \n"
              "vaddw.s16  q9, q9, d13               \n"
              "vaddw.s16  q9, q9, d15               \n"

              "vld1.32    {d12-d15}, [%[output0n]]  \n"
              "vadd.s32   q6, q6, q8                \n"
              "vadd.s32   q7, q7, q9                \n"
              "vst1.32    {d12-d15}, [%[output0n]]! \n"

              "subs       %[ow], #1                 \n"
              "bne        0b                        \n"
              : [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), [r3] "+r"(r3),
                [ow] "+r"(ow), [output0] "+r"(output0),
                [output0n] "+r"(output0n)
              : [kernel0] "r"(kernel0)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "r5");
        }
        if (remain > 0) {
          asm volatile(
              "vld1.8     {d0}, [%[kernel0]]       \n"
              "ldr        r5,   [%[kernel0], #8]   \n"

              "0:                                  \n"
              "vld1.8     d4, [%[r0]]              \n"
              "vld1.8     d5, [%[r1]]              \n"
              "vld1.8     d6, [%[r2]]              \n"
              "vld1.8     d7, [%[r3]]              \n"
              "add        %[r0], #1                \n"
              "add        %[r1], #1                \n"
              "add        %[r2], #1                \n"
              "add        %[r3], #1                \n"
              "vdup.s8    d2, r5                   \n"
              "vext.8     d8, d0, d2, #3           \n"
              "vext.8     d9, d0, d2, #6           \n"

              "vmull.s8   q6, d4, d0               \n"
              "vmull.s8   q7, d5, d8               \n"
              "vmlal.s8   q6, d6, d9               \n"
              "vaddl.s16  q12, d12, d14            \n"
              "vdup.s32   d2, d24[1]               \n"
              "vadd.s32   d24, d24, d2             \n"
              "vadd.s32   d24, d24, d25            \n"

              "ldr        r7, [%[output0]]         \n"
              "vdup.s32   d14, r7                  \n"
              "vadd.s32   d14, d14, d24            \n"
              "vst1.32    d14[0], [%[output0]]!    \n"

              "vmull.s8   q6, d5, d0               \n"
              "vmull.s8   q7, d6, d8               \n"
              "vmlal.s8   q6, d7, d9               \n"
              "vaddl.s16  q12, d12, d14            \n"
              "vdup.s32   d2, d24[1]               \n"
              "vadd.s32   d24, d24, d2             \n"
              "vadd.s32   d24, d24, d25            \n"

              "ldr        r7, [%[output0n]]        \n"
              "vdup.s32   d14, r7                  \n"
              "vadd.s32   d14, d14, d24            \n"
              "vst1.32    d14[0], [%[output0n]]!   \n"

              "subs       %[remain], #1            \n"
              "bne        0b                       \n"
              : [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), [r3] "+r"(r3),
                [remain] "+r"(remain), [output0] "+r"(output0),
                [output0n] "+r"(output0n)
              : [kernel0] "r"(kernel0)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "r5", "r7");
        }
        output0 += output_w;
        output0n += output_w;
      }
      // remain output height
      for (; oh < output_h; ++oh) {
        const int8_t* r0 = in_data + ic * image_size + oh * input_w;
        const int8_t* r1 = r0 + input_w;
        const int8_t* r2 = r1 + input_w;

        int ow = output_w >> 3;
        int remain = output_w & 0x7;
        if (ow > 0) {
          asm volatile(
              "vld1.8     {d0}, [%[kernel0]]        \n"
              "ldr        r5,   [%[kernel0], #8]    \n"

              "0:                                   \n"
              "vld1.8     {d2-d3}, [%[r0]]          \n"  // r0
              "add        %[r0], #8                 \n"
              "vext.8     d4, d2, d3, #1            \n"
              "vext.8     d5, d2, d3, #2            \n"
              "vdup.s8    d6, d0[0]                 \n"
              "vdup.s8    d7, d0[1]                 \n"
              "vdup.s8    d8, d0[2]                 \n"
              "vmull.s8   q6, d2, d6                \n"
              "vmull.s8   q7, d4, d7                \n"
              "vmlal.s8   q6, d5, d8                \n"
              "vaddl.s16  q12, d12, d14             \n"
              "vaddl.s16  q13, d13, d15             \n"

              "vld1.8     {d2-d3}, [%[r1]]          \n"  // r1
              "add        %[r1], #8                 \n"
              "vext.8     d4, d2, d3, #1            \n"
              "vext.8     d5, d2, d3, #2            \n"
              "vdup.s8    d6, d0[3]                 \n"
              "vdup.s8    d7, d0[4]                 \n"
              "vdup.s8    d8, d0[5]                 \n"
              "vmull.s8   q6, d2, d6                \n"
              "vmull.s8   q7, d4, d7                \n"
              "vmlal.s8   q6, d5, d8                \n"
              "vaddw.s16  q12, q12, d12             \n"
              "vaddw.s16  q12, q12, d14             \n"
              "vaddw.s16  q13, q13, d13             \n"
              "vaddw.s16  q13, q13, d15             \n"

              "vld1.8     {d2-d3}, [%[r2]]          \n"  // r2
              "add        %[r2], #8                 \n"
              "vext.8     d4, d2, d3, #1            \n"
              "vext.8     d5, d2, d3, #2            \n"
              "vdup.s8    d6, d0[6]                 \n"
              "vdup.s8    d7, d0[7]                 \n"
              "vdup.s8    d8, r5                    \n"
              "vmull.s8   q6, d2, d6                \n"
              "vmull.s8   q7, d4, d7                \n"
              "vmlal.s8   q6, d5, d8                \n"
              "vaddw.s16  q12, q12, d12             \n"
              "vaddw.s16  q12, q12, d14             \n"
              "vaddw.s16  q13, q13, d13             \n"
              "vaddw.s16  q13, q13, d15             \n"

              "vld1.32    {d12-d15}, [%[output0]]   \n"
              "vadd.s32   q6, q6, q12               \n"
              "vadd.s32   q7, q7, q13               \n"
              "vst1.32    {d12-d15}, [%[output0]]!  \n"

              "subs       %[ow], #1                 \n"
              "bne        0b                        \n"
              : [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), [ow] "+r"(ow),
                [output0] "+r"(output0)
              : [kernel0] "r"(kernel0)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "r5");
        }

        if (remain > 0) {
          asm volatile(
              "vld1.8     {d0}, [%[kernel0]]       \n"
              "ldr        r5,   [%[kernel0], #8]   \n"

              "0:                                  \n"
              "vld1.8     d4, [%[r0]]              \n"
              "vld1.8     d5, [%[r1]]              \n"
              "vld1.8     d6, [%[r2]]              \n"
              "add        %[r0], #1                \n"
              "add        %[r1], #1                \n"
              "add        %[r2], #1                \n"
              "vdup.s8    d2, r5                   \n"
              "vext.8     d8, d0, d2, #3           \n"
              "vext.8     d9, d0, d2, #6           \n"

              "vmull.s8   q6, d4, d0               \n"
              "vmull.s8   q7, d5, d8               \n"
              "vmlal.s8   q6, d6, d9               \n"
              "vaddl.s16  q12, d12, d14            \n"
              "vdup.s32   d2, d24[1]               \n"
              "vadd.s32   d24, d24, d2             \n"
              "vadd.s32   d24, d24, d25            \n"

              "ldr        r7, [%[output0]]         \n"
              "vdup.s32   d14, r7                  \n"
              "vadd.s32   d14, d14, d24            \n"
              "vst1.32    d14[0], [%[output0]]!    \n"

              "subs       %[remain], #1            \n"
              "bne        0b                       \n"
              : [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2),
                [remain] "+r"(remain), [output0] "+r"(output0)
              : [kernel0] "r"(kernel0)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "r5", "r7");
        }
      }
    }
  }
#endif
#else
// TODO(hjchen2)
#endif
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
