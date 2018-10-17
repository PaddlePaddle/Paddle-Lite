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

#if __ARM_NEON
#include <arm_neon.h>
#endif
#include "operators/kernel/central-arm-func/conv_arm_int8.h"

namespace paddle_mobile {
namespace operators {

void transform_kernel3x3_s1_int8(const framework::Tensor* filter,
                                 framework::Tensor* filter_tm, int inch,
                                 int outch) {
  filter_tm->mutable_data<int8_t>(
      framework::make_ddim({outch / 4 + outch % 4, inch, 4 * 9}));
  const int8_t* filter_data = filter->data<int8_t>();
  int p = 0;
  for (; p + 3 < outch; p += 4) {
    const int8_t* k0 = filter_data + (p + 0) * inch * 9;
    const int8_t* k1 = filter_data + (p + 1) * inch * 9;
    const int8_t* k2 = filter_data + (p + 2) * inch * 9;
    const int8_t* k3 = filter_data + (p + 3) * inch * 9;

    int8_t* filter_tmp = filter_tm->Slice(p / 4, p / 4 + 1).data<int8_t>();

    for (int q = 0; q < inch; q++) {
      asm volatile(
          "vld1.s8    {d0-d1}, [%[k0]] \n"
          "add %[k0], #9\n"
          "vld1.s8    {d2-d3}, [%[k1]] \n"
          "add %[k1], #9\n"
          "vld1.s8    {d4-d5}, [%[k2]] \n"
          "add %[k2], #9\n"
          "vld1.s8    {d6-d7}, [%[k3]] \n"
          "add %[k3], #9\n"
          "vst4.s8 {d0, d2, d4, d6}, [%[filter_tmp]]!\n"
          "vst4.s8 {d1, d3, d5, d7}, [%[filter_tmp]]\n"
          "add %[filter_tmp], #4\n"
          : [k0] "+r"(k0), [k1] "+r"(k1), [k2] "+r"(k2), [k3] "+r"(k3),
            [filter_tmp] "+r"(filter_tmp)
          :
          : "memory", "q0", "q1", "q2", "q3");
    }
  }
  for (; p < outch; p++) {
    const int8_t* k0 = filter_data + (p + 0) * inch * 9;
    int8_t* filter_tmp =
        filter_tm->Slice(p / 4 + p % 4, p / 4 + p % 4 + 1).data<int8_t>();

    for (int q = 0; q < inch; q++) {
      asm volatile(
          "vld1.s8    {d0-d1}, [%[k0]] \n"
          "add %[k0], #9\n"
          "vst1.s8 {d0-d1}, [%[filter_tmp]]\n"
          "add %[filter_tmp], #9\n"
          : [k0] "+r"(k0), [filter_tmp] "+r"(filter_tmp)
          :
          : "memory", "q0");
    }
  }
}

void conv3x3s1_int8(const framework::Tensor& input,
                    const framework::Tensor& weight,
                    framework::Tensor* output) {
  int64_t inch = input.dims()[1];
  int64_t h = input.dims()[2];
  int64_t w = input.dims()[3];

  int64_t outch = output->dims()[1];
  int64_t outh = output->dims()[2];
  int64_t outw = output->dims()[3];

  const int8_t* in_data = input.data<int8_t>();
  const int8_t* w_data = weight.data<int8_t>();
  int32_t* out_data = output->mutable_data<int32_t>();
  memset(out_data, 0, output->numel() * sizeof(int32_t));

  int64_t nn_outch = outch >> 2;
  int64_t remain_outch_start = nn_outch << 2;

  framework::Tensor weight_tm;
  transform_kernel3x3_s1_int8(&weight, &weight_tm, weight.dims()[1],
                              weight.dims()[0]);

  #pragma omp parallel for
  for (int pp = 0; pp < nn_outch; pp++) {
    int p = pp * 4;
    const int8_t* ktmp = weight_tm.Slice(p / 4, p / 4 + 1).data<int8_t>();

    for (int q = 0; q < inch; q++) {
      int32_t* outptr0 = out_data + p * outh * outw;
      int32_t* outptr1 = outptr0 + outh * outw;
      int32_t* outptr2 = outptr0 + outh * outw * 2;
      int32_t* outptr3 = outptr0 + outh * outw * 3;

      const int8_t* img0 = in_data + q * h * w;
      const int8_t* r0 = img0;
      const int8_t* r1 = img0 + w;
      const int8_t* r2 = img0 + w * 2;
      const int8_t* r3 = img0 + w * 3;

      int i = 0;
      for (; i + 1 < outh; i += 2) {  // 每次计算两行的输出
        int nn = outw >> 3;
        int remain = outw & 7;
        if (nn > 0) {
          asm volatile(
              "0:                         \n"
              //                     "pld        [%[ktmp], #256]      \n"
              "vld1.s8    {d0-d3}, [%[ktmp]]!  \n"  // d0=k00
                                                    // k01(四个通道的k00和k01，8个数字，8*8=64bit)
                                                    // d1=k02 k10  d2=k11
              // k12  d3=k20 k21

              "pld        [%[r0], #128]      \n"
              "vld1.s8    {d4-d5}, [%[r0]]   \n"  // d4=r00 d5=r00n
              "add        %[r0], #8          \n"

              "vdup.s8    d8, d0[0]       \n"  // d8中每个8bit元素的值均为d0[0],d8内容为第1个通道的第1个值重复8次
              "vdup.s8    d9, d0[1]       \n"  // d9中每个8bit元素的值均为d0[1],d9内容为第2个通道的第1个值重复8次

              "pld        [%[r1], #128]      \n"
              "vld1.s8    {d6-d7}, [%[r1]]   \n"  // d6=r10 d7=r10n
              "add        %[r1], #8          \n"

              "vdup.s8    d10, d0[2]      \n"  // d10中每个8bit元素的值均为d0[2],d10内容为第3个通道的第1个值重复8次
              "vdup.s8    d11, d0[3]      \n"  // d11中每个8bit元素的值均为d0[3],d11内容为第4个通道的第1个值重复8次

              "vmull.s8   q8, d4, d8      \n"  // 将第1行的前8个元素与第1个通道的第1个值相乘，结果q8
              "vmull.s8   q9, d4, d9      \n"  // 将第1行的前8个元素与第2个通道的第1个值相乘，结果q9

              "vdup.s8    d12, d0[4]      \n"  // d12内容为第1个通道的第2个值重复8次
              "vdup.s8    d13, d0[5]      \n"  // d13内容为第2个通道的第2个值重复8次

              "vmull.s8   q10, d4, d10    \n"  // 将第1行的前8个元素与第3个通道的第1个值相乘，结果q10
              "vmull.s8   q11, d4, d11    \n"  // 将第1行的前8个元素与第4个通道的第1个值相乘，结果q11

              "vdup.s8    d14, d0[6]      \n"  // d14内容为第3个通道的第2个值重复8次
              "vdup.s8    d15, d0[7]      \n"  // d15内容为第4个通道的第2个值重复8次

              "vmull.s8   q12, d6, d8     \n"  // 将第2行的前8个元素与第1个通道的第1个值相乘，结果q12
              "vmull.s8   q13, d6, d9     \n"  // 将第2行的前8个元素与第2个通道的第1个值相乘，结果q13

              "vext.s8    q2, q2, q2, #1  \n"  // d4=r01,循环右移一位（每位为8bit元素），第1行第2个值

              "vmull.s8   q14, d6, d10    \n"  // 将第2行的前8个元素与第3个通道的第1个值相乘，结果q14
              "vmull.s8   q15, d6, d11    \n"  // 将第2行的前8个元素与第4个通道的第1个值相乘，结果q14

              "vext.s8    q3, q3, q3, #1  \n"  // d6=r11，第2行第2个值

              "vmlal.s8   q8, d4, d12     \n"  // 第1行第1个通道：加乘上第2个值
              "vmlal.s8   q9, d4, d13     \n"  // 第1行第2个通道：加乘上第2个值

              "vdup.s8    d8, d1[0]       \n"  // d8内容为第1个通道的第3个值重复8次
              "vdup.s8    d9, d1[1]       \n"  // d9内容为第2个通道的第3个值重复8次

              "vmlal.s8   q10, d4, d14    \n"  // 第1行第3个通道：加乘上第2个值
              "vmlal.s8   q11, d4, d15    \n"  // 第1行第4个通道：加乘上第2个值

              "vdup.s8    d10, d1[2]      \n"  // d10内容为第3个通道的第3个值重复8次
              "vdup.s8    d11, d1[3]      \n"  // d10内容为第4个通道的第3个值重复8次

              "vmlal.s8   q12, d6, d12    \n"  // 第2行第1个通道：加乘上第2个值
              "vmlal.s8   q13, d6, d13    \n"  // 第2行第2个通道：加乘上第2个值

              "vext.s8    q2, q2, q2, #1  \n"  // d4=r02，第1行第3个值

              "vmlal.s8   q14, d6, d14    \n"  // 第2行第3个通道：加乘上第2个值
              "vmlal.s8   q15, d6, d15    \n"  // 第2行第3个通道：加乘上第2个值

              "vext.s8    q3, q3, q3, #1  \n"  // d6=r12，第2行第3个值

              "vmlal.s8   q8, d4, d8      \n"  // 第1行第1个通道：加乘上第3个值
              "vmlal.s8   q9, d4, d9      \n"  // 第1行第2个通道：加乘上第3个值

              "vdup.s8    d12, d1[4]      \n"  // d12内容为第1个通道的第4个值重复8次
              "vdup.s8    d13, d1[5]      \n"  // d13内容为第2个通道的第4个值重复8次

              "vmlal.s8   q10, d4, d10    \n"  // 第1行第3个通道：加乘上第3个值
              "vmlal.s8   q11, d4, d11    \n"  // 第1行第4个通道：加乘上第3个值

              "vdup.s8    d14, d1[6]      \n"  // d14内容为第3个通道的第4个值重复8次
              "vdup.s8    d15, d1[7]      \n"  // d15内容为第4个通道的第4个值重复8次

              "vmlal.s8   q12, d6, d8     \n"  // 第2行第1个通道：加乘上第3个值
              "vmlal.s8   q13, d6, d9     \n"  // 第2行第2个通道：加乘上第3个值

              "pld        [%[r2], #128]      \n"
              "vld1.s8    {d4-d5}, [%[r2]]   \n"  // d4=r20 d5=r20n
              "add        %[r2], #8          \n"

              "vmlal.s8   q14, d6, d10    \n"  // 第2行第3个通道：加乘上第3个值
              "vmlal.s8   q15, d6, d11    \n"  // 第2行第4个通道：加乘上第3个值

              ///
              "vext.s8    q3, q3, q3, #14 \n"  // d6=r10，输入的第4个值

              "vmlal.s8   q8, d6, d12     \n"  // 第1行第1个通道：加乘上第4个值
              "vmlal.s8   q9, d6, d13     \n"  // 第1行第2个通道：加乘上第4个值

              "vdup.s8    d8, d2[0]       \n"  // d8内容为第1个通道的第5个值重复8次
              "vdup.s8    d9, d2[1]       \n"  // d9内容为第2个通道的第5个值重复8次

              "vmlal.s8   q10, d6, d14    \n"  // 第1行第3个通道：加乘上第4个值
              "vmlal.s8   q11, d6, d15    \n"  // 第1行第4个通道：加乘上第4个值

              "vdup.s8    d10, d2[2]      \n"  // d10内容为第3个通道的第5个值重复8次
              "vdup.s8    d11, d2[3]      \n"  // d11内容为第4个通道的第5个值重复8次

              "vmlal.s8   q12, d4, d12    \n"  // 第2行第1个通道：加乘上第4个值
              "vmlal.s8   q13, d4, d13    \n"  // 第2行第2个通道：加乘上第4个值

              "vext.s8    q3, q3, q3, #1  \n"  // d6=r11，输入的第5个值

              "vmlal.s8   q14, d4, d14    \n"  // 第2行第3个通道：加乘上第4个值
              "vmlal.s8   q15, d4, d15    \n"  // 第2行第4个通道：加乘上第4个值

              "vext.s8    q2, q2, q2, #1  \n"  // d4=r21

              "vmlal.s8   q8, d6, d8      \n"
              "vmlal.s8   q9, d6, d9      \n"

              "vdup.s8    d12, d2[4]      \n"  // d12内容为第1个通道的第6个值重复8次
              "vdup.s8    d13, d2[5]      \n"  // d13内容为第2个通道的第6个值重复8次

              "vmlal.s8   q10, d6, d10    \n"
              "vmlal.s8   q11, d6, d11    \n"

              "vdup.s8    d14, d2[6]      \n"  // d14内容为第3个通道的第6个值重复8次
              "vdup.s8    d15, d2[7]      \n"  // d15内容为第4个通道的第6个值重复8次

              "vmlal.s8   q12, d4, d8     \n"
              "vmlal.s8   q13, d4, d9     \n"

              "vext.s8    q3, q3, q3, #1  \n"  // d6=r12

              "vmlal.s8   q14, d4, d10    \n"
              "vmlal.s8   q15, d4, d11    \n"

              "vext.s8    q2, q2, q2, #1  \n"  // d4=r22

              "vmlal.s8   q8, d6, d12     \n"
              "vmlal.s8   q9, d6, d13     \n"

              "vdup.s8    d8, d3[0]       \n"
              "vdup.s8    d9, d3[1]       \n"

              "vmlal.s8   q10, d6, d14    \n"
              "vmlal.s8   q11, d6, d15    \n"

              "vdup.s8    d10, d3[2]      \n"
              "vdup.s8    d11, d3[3]      \n"

              "vmlal.s8   q12, d4, d12    \n"
              "vmlal.s8   q13, d4, d13    \n"

              "pld        [%[r3], #128]      \n"
              "vld1.s8    {d6-d7}, [%[r3]]   \n"  // d6=r30 d6=r30n
              "add        %[r3], #8          \n"

              "vmlal.s8   q14, d4, d14    \n"
              "vmlal.s8   q15, d4, d15    \n"

              ///
              "vext.s8    q2, q2, q2, #14 \n"  // d4=r20

              "vmlal.s8   q8, d4, d8      \n"
              "vmlal.s8   q9, d4, d9      \n"

              "vdup.s8    d12, d3[4]      \n"
              "vdup.s8    d13, d3[5]      \n"

              "vmlal.s8   q10, d4, d10    \n"
              "vmlal.s8   q11, d4, d11    \n"

              "vdup.s8    d14, d3[6]      \n"
              "vdup.s8    d15, d3[7]      \n"

              "vmlal.s8   q12, d6, d8     \n"
              "vmlal.s8   q13, d6, d9     \n"

              "vext.s8    q2, q2, q2, #1  \n"  // d4=r21

              "vmlal.s8   q14, d6, d10    \n"
              "vmlal.s8   q15, d6, d11    \n"

              "vext.s8    q3, q3, q3, #1  \n"  // d6=r31

              //                     "pld        [%[ktmp], #128]      \n"
              "vld1.s8    {d0}, [%[ktmp]]      \n"
              "add        %[ktmp], #4          \n"

              "vmlal.s8   q8, d4, d12     \n"
              "vmlal.s8   q9, d4, d13     \n"

              "vdup.s8    d8, d0[0]       \n"
              "vdup.s8    d9, d0[1]       \n"

              "vmlal.s8   q10, d4, d14    \n"
              "vmlal.s8   q11, d4, d15    \n"

              "vdup.s8    d10, d0[2]      \n"
              "vdup.s8    d11, d0[3]      \n"

              "vmlal.s8   q12, d6, d12    \n"
              "vmlal.s8   q13, d6, d13    \n"

              "vext.s8    q2, q2, q2, #1  \n"  // d4=r22

              "vmlal.s8   q14, d6, d14    \n"
              "vmlal.s8   q15, d6, d15    \n"

              "vext.s8    q3, q3, q3, #1  \n"  // d6=r32

              "vmlal.s8   q8, d4, d8      \n"
              "vmlal.s8   q9, d4, d9      \n"

              "pld        [%[outptr0], #256]      \n"
              "vld1.s32   {d12-d15}, [%[outptr0]] \n"

              "vmlal.s8   q10, d4, d10    \n"
              "vmlal.s8   q11, d4, d11    \n"

              "pld        [%[outptr1], #256]      \n"
              "vld1.s32   {d0-d3}, [%[outptr1]]   \n"

              "vaddw.s16  q6, q6, d16     \n"
              "vaddw.s16  q7, q7, d17     \n"
              "vaddw.s16  q0, q0, d18     \n"
              "vaddw.s16  q1, q1, d19     \n"

              "pld        [%[outptr2], #256]      \n"
              "vld1.s32   {d16-d19}, [%[outptr2]]  \n"

              "vmlal.s8   q12, d6, d8     \n"
              "vmlal.s8   q13, d6, d9     \n"

              "vst1.s32   {d12-d15}, [%[outptr0]] \n"
              "add        %[outptr0], %[outptr0], %[outw], lsl #2 \n"

              "vmlal.s8   q14, d6, d10    \n"
              "vmlal.s8   q15, d6, d11    \n"

              "pld        [%[outptr3], #256]      \n"
              "vld1.s32   {d4-d7}, [%[outptr3]]   \n"

              "vst1.s32   {d0-d3}, [%[outptr1]]   \n"
              "add        %[outptr1], %[outptr1], %[outw], lsl #2 \n"

              "vaddw.s16  q8, q8, d20     \n"
              "vaddw.s16  q9, q9, d21     \n"

              "pld        [%[outptr0], #256]      \n"
              "vld1.s32   {d12-d15}, [%[outptr0]] \n"

              "vaddw.s16  q2, q2, d22     \n"
              "vaddw.s16  q3, q3, d23     \n"

              ///
              "pld        [%[outptr1], #256]      \n"
              "vld1.s32   {d0-d3}, [%[outptr1]]   \n"

              "vaddw.s16  q6, q6, d24     \n"

              "vst1.s32   {d16-d19}, [%[outptr2]] \n"
              "add        %[outptr2], %[outptr2], %[outw], lsl #2 \n"

              "vaddw.s16  q7, q7, d25     \n"

              "pld        [%[outptr2], #256]      \n"
              "vld1.s32   {d8-d11}, [%[outptr2]] \n"

              "vaddw.s16  q0, q0, d26     \n"

              "vst1.s32   {d4-d7}, [%[outptr3]]   \n"
              "add        %[outptr3], %[outptr3], %[outw], lsl #2 \n"

              ///
              "vaddw.s16  q1, q1, d27     \n"

              "pld        [%[outptr3], #256]      \n"
              "vld1.s32   {d4-d7}, [%[outptr3]]   \n"

              "vaddw.s16  q4, q4, d28     \n"

              "vst1.s32   {d12-d15}, [%[outptr0]]! \n"

              "vaddw.s16  q5, q5, d29     \n"

              "vst1.s32   {d0-d3}, [%[outptr1]]!  \n"

              "vaddw.s16  q2, q2, d30     \n"

              "vst1.s32   {d8-d11}, [%[outptr2]]! \n"

              "vaddw.s16  q3, q3, d31     \n"

              "sub        %[ktmp], #36         \n"
              "subs       %[nn], #1          \n"

              "sub        %[outptr0], %[outptr0], %[outw], lsl #2 \n"
              "sub        %[outptr1], %[outptr1], %[outw], lsl #2 \n"
              "sub        %[outptr2], %[outptr2], %[outw], lsl #2 \n"

              "vst1.s32   {d4-d7}, [%[outptr3]]!  \n"

              "sub        %[outptr3], %[outptr3], %[outw], lsl #2 \n"

              "bne        0b              \n"

              : [nn] "+r"(nn), [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1),
                [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3), [r0] "+r"(r0),
                [r1] "+r"(r1), [r2] "+r"(r2), [r3] "+r"(r3), [ktmp] "+r"(ktmp)
              : [outw] "r"(outw)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
        }

        for (; remain > 0; remain--) {
          asm volatile(
              "vld1.s8    {d0[]}, [%[r0]]!       \n"  // d0 = 00 00
              "vld1.s8    {d1[]}, [%[r0]]!       \n"  // d1 = 01 01
              "vld1.s8    {d2[]}, [%[r0]]        \n"  // d2 = 02 02
              "sub        %[r0], %[r0], #2          \n"

              "vld1.s8    {d3[]}, [%[r1]]!       \n"  // d3 = 10 10
              "vld1.s8    {d4[]}, [%[r1]]!       \n"  // d4 = 11 11
              "vld1.s8    {d5[]}, [%[r1]]        \n"  // d5 = 12 12
              "sub        %[r1], %[r1], #2          \n"

              "vld1.s8    {d6[]}, [%[r2]]!       \n"  // d6 = 20 20
              "vld1.s8    {d7[]}, [%[r2]]!       \n"  // d7 = 21 21
              "vld1.s8    {d8[]}, [%[r2]]        \n"  // d8 = 22 22
              "sub        %[r2], %[r2], #2          \n"

              "vld1.s8    {d9[]}, [%[r3]]!       \n"  // d9 = 30 30
              "vld1.s8    {d10[]}, [%[r3]]!      \n"  // d10 = 31 31
              "vld1.s8    {d11[]}, [%[r3]]       \n"  // d11 = 32 32
              "sub        %[r3], %[r3], #2          \n"

              "vld1.s8    {d12-d15}, [%[ktmp]]!    \n"  // d12 d13 d14 d15 = 0~7

              "vsli.64    d0, d1, #32         \n"  // d0 = 00 01

              "vsli.64    d3, d4, #32         \n"  // d3 = 10 11

              "vmull.s8   q8, d0, d12         \n"

              "vsli.64    d2, d3, #32         \n"  // d2 = 02 10

              "vmull.s8   q9, d3, d12         \n"

              "vsli.64    d5, d6, #32         \n"  // d5 = 12 20

              "vmlal.s8   q8, d2, d13         \n"

              "vsli.64    d4, d5, #32         \n"  // d4 = 11 12

              "vmlal.s8   q9, d5, d13         \n"

              "vsli.64    d7, d8, #32         \n"  // d7 = 21 22

              "vmlal.s8   q8, d4, d14         \n"

              "vsli.64    d6, d7, #32         \n"  // d6 = 20 21

              "vmlal.s8   q9, d7, d14         \n"

              "vsli.64    d9, d10, #32        \n"  // d9 = 30 31

              "vld1.s32   {d20[0]}, [%[outptr0]]      \n"
              "vld1.s32   {d20[1]}, [%[outptr1]]      \n"

              "add        %[outptr0], %[outptr0], %[outw], lsl #2 \n"
              "add        %[outptr1], %[outptr1], %[outw], lsl #2 \n"

              "vmlal.s8   q8, d6, d15         \n"

              "vsli.64    d8, d11, #32        \n"  // d8 = 22 32

              "vmlal.s8   q9, d9, d15         \n"

              "vld1.s8    {d14}, [%[ktmp]]         \n"
              "add        %[ktmp], #4              \n"

              "vld1.s32   {d21[0]}, [%[outptr2]]      \n"
              "vld1.s32   {d21[1]}, [%[outptr3]]      \n"

              "add        %[outptr2], %[outptr2], %[outw], lsl #2 \n"
              "add        %[outptr3], %[outptr3], %[outw], lsl #2 \n"

              "vadd.s16   d12, d16, d17       \n"

              "vadd.s16   d13, d18, d19       \n"  // q6 = sum0123 sum0123n

              "vsli.64    d14, d14, #32       \n"  // d14 = 0~3 0~3

              "vld1.s32   {d22[0]}, [%[outptr0]]      \n"
              "vld1.s32   {d22[1]}, [%[outptr1]]      \n"

              "vmlal.s8   q6, d8, d14         \n"

              "sub        %[ktmp], #36             \n"

              ///
              "vld1.s32   {d23[0]}, [%[outptr2]]      \n"
              "vld1.s32   {d23[1]}, [%[outptr3]]      \n"

              "sub        %[outptr0], %[outptr0], %[outw], lsl #2 \n"
              "sub        %[outptr1], %[outptr1], %[outw], lsl #2 \n"

              // addw
              "vaddw.s16  q10, q10, d12       \n"
              "vaddw.s16  q11, q11, d13       \n"

              "sub        %[outptr2], %[outptr2], %[outw], lsl #2 \n"
              "sub        %[outptr3], %[outptr3], %[outw], lsl #2 \n"

              "vst1.s32   {d20[0]}, [%[outptr0]]      \n"
              "vst1.s32   {d20[1]}, [%[outptr1]]      \n"

              "add        %[outptr0], %[outptr0], %[outw], lsl #2 \n"
              "add        %[outptr1], %[outptr1], %[outw], lsl #2 \n"

              "vst1.s32   {d21[0]}, [%[outptr2]]      \n"
              "vst1.s32   {d21[1]}, [%[outptr3]]      \n"

              "add        %[outptr2], %[outptr2], %[outw], lsl #2 \n"
              "add        %[outptr3], %[outptr3], %[outw], lsl #2 \n"

              "vst1.s32   {d22[0]}, [%[outptr0]]!     \n"
              "vst1.s32   {d22[1]}, [%[outptr1]]!     \n"

              "sub        %[outptr0], %[outptr0], %[outw], lsl #2 \n"
              "sub        %[outptr1], %[outptr1], %[outw], lsl #2 \n"

              "vst1.s32   {d23[0]}, [%[outptr2]]!     \n"
              "vst1.s32   {d23[1]}, [%[outptr3]]!     \n"

              "sub        %[outptr2], %[outptr2], %[outw], lsl #2 \n"
              "sub        %[outptr3], %[outptr3], %[outw], lsl #2 \n"

              : [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1),
                [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3), [r0] "+r"(r0),
                [r1] "+r"(r1), [r2] "+r"(r2), [r3] "+r"(r3), [ktmp] "+r"(ktmp)
              : [outw] "r"(outw)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "q11");
          r0++;
          r1++;
          r2++;
          r3++;
        }

        r0 += 2 + w;
        r1 += 2 + w;
        r2 += 2 + w;
        r3 += 2 + w;

        outptr0 += outw;
        outptr1 += outw;
        outptr2 += outw;
        outptr3 += outw;
      }

      for (; i < outh; i++) {
        int nn = outw >> 3;
        int remain = outw & 7;
        if (nn > 0) {
          asm volatile(
              "0:                         \n"
              //                     "pld        [%[ktmp], #256]      \n"
              "vld1.s8    {d0-d3}, [%[ktmp]]!  \n"  // d0=k00 k01  d1=k02 k10
                                                    // d2=k11
              // k12  d3=k20 k21

              "pld        [%[r0], #128]      \n"
              "vld1.s8    {d4-d5}, [%[r0]]   \n"  // d4=r00 d5=r00n
              "add        %[r0], #8          \n"

              "vdup.s8    d8, d0[0]       \n"
              "vdup.s8    d9, d0[1]       \n"
              "vdup.s8    d10, d0[2]      \n"
              "vdup.s8    d11, d0[3]      \n"

              "vmull.s8   q8, d4, d8      \n"
              "vmull.s8   q9, d4, d9      \n"

              "vext.s8    d24, d4, d5, #1 \n"  // d24=r01

              "vdup.s8    d12, d0[4]      \n"
              "vdup.s8    d13, d0[5]      \n"

              "vmull.s8   q10, d4, d10    \n"
              "vmull.s8   q11, d4, d11    \n"

              "vdup.s8    d14, d0[6]      \n"
              "vdup.s8    d15, d0[7]      \n"

              "vmlal.s8   q8, d24, d12    \n"
              "vmlal.s8   q9, d24, d13    \n"

              "vext.s8    d25, d4, d5, #2 \n"  // d25=r02

              "vdup.s8    d8, d1[0]       \n"
              "vdup.s8    d9, d1[1]       \n"

              "vmlal.s8   q10, d24, d14   \n"
              "vmlal.s8   q11, d24, d15   \n"

              "vdup.s8    d10, d1[2]      \n"
              "vdup.s8    d11, d1[3]      \n"

              "vmlal.s8   q8, d25, d8     \n"
              "vmlal.s8   q9, d25, d9     \n"

              "pld        [%[r1], #128]      \n"
              "vld1.s8    {d6-d7}, [%[r1]]   \n"  // d6=r10 d7=r10n
              "add        %[r1], #8          \n"

              "vdup.s8    d12, d1[4]      \n"
              "vdup.s8    d13, d1[5]      \n"

              "vmlal.s8   q10, d25, d10   \n"
              "vmlal.s8   q11, d25, d11   \n"

              "vdup.s8    d14, d1[6]      \n"
              "vdup.s8    d15, d1[7]      \n"

              "vmlal.s8   q8, d6, d12     \n"
              "vmlal.s8   q9, d6, d13     \n"

              "vext.s8    d26, d6, d7, #1 \n"  // d26=r11

              "vdup.s8    d8, d2[0]       \n"
              "vdup.s8    d9, d2[1]       \n"

              "vmlal.s8   q10, d6, d14    \n"
              "vmlal.s8   q11, d6, d15    \n"

              "vdup.s8    d10, d2[2]      \n"
              "vdup.s8    d11, d2[3]      \n"

              "vmlal.s8   q8, d26, d8     \n"
              "vmlal.s8   q9, d26, d9     \n"

              "vext.s8    d27, d6, d7, #2 \n"  // d27=r12

              "vdup.s8    d12, d2[4]      \n"
              "vdup.s8    d13, d2[5]      \n"

              "vmlal.s8   q10, d26, d10   \n"
              "vmlal.s8   q11, d26, d11   \n"

              "vdup.s8    d14, d2[6]      \n"
              "vdup.s8    d15, d2[7]      \n"

              "vmlal.s8   q8, d27, d12    \n"
              "vmlal.s8   q9, d27, d13    \n"

              "pld        [%[r2], #128]      \n"
              "vld1.s8    {d4-d5}, [%[r2]]   \n"  // d4=r20 d5=r20n
              "add        %[r2], #8          \n"

              "vdup.s8    d8, d3[0]       \n"
              "vdup.s8    d9, d3[1]       \n"

              "vmlal.s8   q10, d27, d14   \n"
              "vmlal.s8   q11, d27, d15   \n"

              "vdup.s8    d10, d3[2]      \n"
              "vdup.s8    d11, d3[3]      \n"

              "vmlal.s8   q8, d4, d8      \n"
              "vmlal.s8   q9, d4, d9      \n"

              "vext.s8    d24, d4, d5, #1 \n"  // d24=r21

              "vdup.s8    d12, d3[4]      \n"
              "vdup.s8    d13, d3[5]      \n"

              "vmlal.s8   q10, d4, d10    \n"
              "vmlal.s8   q11, d4, d11    \n"

              "vdup.s8    d14, d3[6]      \n"
              "vdup.s8    d15, d3[7]      \n"

              "vmlal.s8   q8, d24, d12    \n"
              "vmlal.s8   q9, d24, d13    \n"

              //                     "pld        [%[ktmp], #128]      \n"
              "vld1.s8    {d0}, [%[ktmp]]      \n"
              "add        %[ktmp], #4          \n"

              "vext.s8    d25, d4, d5, #2 \n"  // d25=r22

              "vdup.s8    d8, d0[0]       \n"
              "vdup.s8    d9, d0[1]       \n"

              "vmlal.s8   q10, d24, d14   \n"
              "vmlal.s8   q11, d24, d15   \n"

              "vdup.s8    d10, d0[2]      \n"
              "vdup.s8    d11, d0[3]      \n"

              "pld        [%[outptr0], #256]      \n"
              "vld1.s32   {d12-d15}, [%[outptr0]] \n"

              "vmlal.s8   q8, d25, d8     \n"
              "vmlal.s8   q9, d25, d9     \n"

              "pld        [%[outptr1], #256]      \n"
              "vld1.s32   {d0-d3}, [%[outptr1]]   \n"

              "vaddw.s16  q6, q6, d16     \n"
              "vaddw.s16  q7, q7, d17     \n"

              "vmlal.s8   q10, d25, d10   \n"
              "vmlal.s8   q11, d25, d11   \n"

              "vaddw.s16  q0, q0, d18     \n"
              "vaddw.s16  q1, q1, d19     \n"

              "pld        [%[outptr2], #256]      \n"
              "vld1.s32   {d16-d19}, [%[outptr2]]  \n"

              "vst1.s32   {d12-d15}, [%[outptr0]]! \n"

              "pld        [%[outptr3], #256]      \n"
              "vld1.s32   {d4-d7}, [%[outptr3]]   \n"

              "vst1.s32   {d0-d3}, [%[outptr1]]!  \n"

              "vaddw.s16  q8, q8, d20     \n"
              "vaddw.s16  q9, q9, d21     \n"
              "vaddw.s16  q2, q2, d22     \n"
              "vaddw.s16  q3, q3, d23     \n"

              "sub        %[ktmp], #36         \n"

              "vst1.s32   {d16-d19}, [%[outptr2]]! \n"

              "subs       %[nn], #1          \n"

              "vst1.s32   {d4-d7}, [%[outptr3]]!  \n"

              "bne        0b              \n"

              : [nn] "+r"(nn), [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1),
                [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3), [r0] "+r"(r0),
                [r1] "+r"(r1), [r2] "+r"(r2), [ktmp] "+r"(ktmp)
              :
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "q11", "q12", "q13");
        }

        for (; remain > 0; remain--) {
          asm volatile(
              "vld1.s8    {d0[]}, [%[r0]]!       \n"
              "vld1.s8    {d1[]}, [%[r0]]!       \n"

              "vld1.s8    {d4-d7}, [%[ktmp]]!      \n"  // d4 d5 d6 d7 = 0~7

              "vsli.64    d0, d1, #32         \n"  // d0 = 00 01

              "vld1.s8    {d2[]}, [%[r0]]        \n"
              "sub        %[r0], %[r0], #2          \n"
              "vld1.s8    {d3[]}, [%[r1]]!       \n"

              "vsli.64    d2, d3, #32         \n"  // d2 = 02 10

              "vmull.s8   q8, d0, d4          \n"

              "vld1.s8    {d0[]}, [%[r1]]!       \n"
              "vld1.s8    {d1[]}, [%[r1]]        \n"
              "sub        %[r1], %[r1], #2          \n"

              "vsli.64    d0, d1, #32         \n"  // d0 = 11 12

              "vmlal.s8   q8, d2, d5          \n"

              "vld1.s8    {d2[]}, [%[r2]]!       \n"
              "vld1.s8    {d3[]}, [%[r2]]!       \n"

              "vsli.64    d2, d3, #32         \n"  // d2 = 20 21

              "vmlal.s8   q8, d0, d6          \n"

              "vld1.s8    {d0[]}, [%[r2]]        \n"
              "sub        %[r2], %[r2], #2          \n"
              "veor       d1, d1, d1          \n"

              "vld1.s8    {d4}, [%[ktmp]]          \n"  // d4 = 0~4 xxxx
              "sub        %[ktmp], #32             \n"

              "vsli.64    d0, d1, #32         \n"  // d0 = 22 zero

              "vmlal.s8   q8, d2, d7          \n"

              "vld1.s32   {d20[0]}, [%[outptr0]]      \n"

              "vmlal.s8   q8, d0, d4          \n"

              "vld1.s32   {d20[1]}, [%[outptr1]]      \n"

              "vadd.s16   d16, d16, d17       \n"

              "vld1.s32   {d21[0]}, [%[outptr2]]      \n"
              "vld1.s32   {d21[1]}, [%[outptr3]]      \n"

              "vaddw.s16  q10, q10, d16       \n"

              "vst1.s32   {d20[0]}, [%[outptr0]]!     \n"
              "vst1.s32   {d20[1]}, [%[outptr1]]!     \n"
              "vst1.s32   {d21[0]}, [%[outptr2]]!     \n"
              "vst1.s32   {d21[1]}, [%[outptr3]]!     \n"

              : [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1),
                [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3), [r0] "+r"(r0),
                [r1] "+r"(r1), [r2] "+r"(r2), [ktmp] "+r"(ktmp)
              :
              : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q10");
          r0++;
          r1++;
          r2++;
        }

        r0 += 2;
        r1 += 2;
        r2 += 2;
      }

      ktmp += 4 * 9;
    }
  }

#pragma omp parallel for
  for (int p = remain_outch_start; p < outch; p++) {
    const int8_t* ktmp =
        weight_tm.Slice(p / 4 + p % 4, p / 4 + p % 4 + 1).data<int8_t>();

    for (int q = 0; q < inch; q++) {
      int32_t* outptr0 = out_data + p * outh * outw;
      int32_t* outptr0n = outptr0 + outw;
      const int8_t* img0 = in_data + q * h * w;
      const int8_t* r0 = img0;
      const int8_t* r1 = img0 + w;
      const int8_t* r2 = img0 + w * 2;
      const int8_t* r3 = img0 + w * 3;
      int8x8_t _k00 = vdup_n_s8(ktmp[0]);
      int8x8_t _k01 = vdup_n_s8(ktmp[1]);
      int8x8_t _k02 = vdup_n_s8(ktmp[2]);
      int8x8_t _k10 = vdup_n_s8(ktmp[3]);
      int8x8_t _k11 = vdup_n_s8(ktmp[4]);
      int8x8_t _k12 = vdup_n_s8(ktmp[5]);
      int8x8_t _k20 = vdup_n_s8(ktmp[6]);
      int8x8_t _k21 = vdup_n_s8(ktmp[7]);
      int8x8_t _k22 = vdup_n_s8(ktmp[8]);

      int i = 0;
      for (; i + 1 < outh; i += 2) {
        int nn = outw >> 3;
        int remain = outw & 7;
        if (nn > 0) {
          asm volatile(
              "0:                         \n"

              "pld        [%[r0], #128]      \n"
              "vld1.s8    {d4-d5}, [%[r0]]   \n"  // d4=r00 d5=r00n
              "add        %[r0], #8          \n"

              "pld        [%[r3], #128]      \n"
              "vld1.s8    {d6-d7}, [%[r3]]   \n"  // d6=r30 d7=r30n
              "add        %[r3], #8          \n"

              "vext.s8    d8, d4, d5, #1  \n"  // d8=r01
              "vext.s8    d10, d6, d7, #1 \n"  // d10=r31

              "vmull.s8   q8, d4, %[_k00]    \n"
              "vmull.s8   q9, d6, %[_k20]    \n"

              "vext.s8    d9, d4, d5, #2  \n"  // d9=r02
              "vext.s8    d11, d6, d7, #2 \n"  // d11=r32

              "vmlal.s8   q8, d8, %[_k01]    \n"
              "vmlal.s8   q9, d10, %[_k21]   \n"

              "pld        [%[r1], #128]      \n"
              "vld1.s8    {d4-d5}, [%[r1]]   \n"  // d4=r10 d5=r10n
              "add        %[r1], #8          \n"

              "vmlal.s8   q8, d9, %[_k02]    \n"
              "vmlal.s8   q9, d11, %[_k22]   \n"

              "vext.s8    d8, d4, d5, #1  \n"  // d8=r11

              "vmlal.s8   q8, d4, %[_k10]    \n"
              "vmlal.s8   q9, d4, %[_k00]    \n"

              "vext.s8    d9, d4, d5, #2  \n"  // d9=r12

              "vmlal.s8   q8, d8, %[_k11]    \n"
              "vmlal.s8   q9, d8, %[_k01]    \n"

              "pld        [%[r2], #128]      \n"
              "vld1.s8    {d6-d7}, [%[r2]]   \n"  // d6=r20 d7=r20n
              "add        %[r2], #8          \n"

              "vmlal.s8   q8, d9, %[_k12]    \n"
              "vmlal.s8   q9, d9, %[_k02]    \n"

              "vext.s8    d10, d6, d7, #1 \n"  // d10=r21

              "vmlal.s8   q8, d6, %[_k20]    \n"
              "vmlal.s8   q9, d6, %[_k10]    \n"

              "vext.s8    d11, d6, d7, #2 \n"  // d11=r22

              "vmlal.s8   q8, d10, %[_k21]   \n"
              "vmlal.s8   q9, d10, %[_k11]   \n"

              "pld        [%[outptr0], #256]      \n"
              "vld1.s32   {d0-d3}, [%[outptr0]]   \n"

              "vmlal.s8   q8, d11, %[_k22]   \n"
              "vmlal.s8   q9, d11, %[_k12]   \n"

              "pld        [%[outptr0n], #256]      \n"
              "vld1.s32   {d12-d15}, [%[outptr0n]] \n"

              "vaddw.s16  q0, q0, d16     \n"
              "vaddw.s16  q1, q1, d17     \n"
              "vaddw.s16  q6, q6, d18     \n"
              "vaddw.s16  q7, q7, d19     \n"

              "vst1.s32   {d0-d3}, [%[outptr0]]!  \n"

              "subs       %[nn], #1          \n"

              "vst1.s32   {d12-d15}, [%[outptr0n]]! \n"

              "bne        0b              \n"

              : [nn] "+r"(nn), [outptr0] "+r"(outptr0),
                [outptr0n] "+r"(outptr0n), [r0] "+r"(r0), [r1] "+r"(r1),
                [r2] "+r"(r2), [r3] "+r"(r3)
              : [_k00] "w"(_k00), [_k01] "w"(_k01), [_k02] "w"(_k02),
                [_k10] "w"(_k10), [_k11] "w"(_k11), [_k12] "w"(_k12),
                [_k20] "w"(_k20), [_k21] "w"(_k21), [_k22] "w"(_k22)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9");
        }

        for (; remain > 0; remain--) {
          asm volatile(
              "vld1.s8    {d0[0]}, [%[r0]]!  \n"
              "vld1.s8    {d0[1]}, [%[r0]]!  \n"
              "vld1.s8    {d0[2]}, [%[r0]]   \n"
              "sub        %[r0], #2          \n"

              "vld1.s8    {d0[3]}, [%[r1]]!  \n"
              "vld1.s8    {d0[4]}, [%[r1]]!  \n"
              "vld1.s8    {d0[5]}, [%[r1]]   \n"
              "sub        %[r1], #2          \n"

              "vld1.s8    {d0[6]}, [%[r2]]!  \n"
              "vld1.s8    {d0[7]}, [%[r2]]!  \n"  // d0=r

              "vld1.s8    {d4[]}, [%[r2]]    \n"  // d4=r22
              "sub        %[r2], #2          \n"

              "vext.s8    d1, d0, d4, #3  \n"

              "vld1.s8    {d1[6]}, [%[r3]]!  \n"
              "vld1.s8    {d1[7]}, [%[r3]]!  \n"  // d1=rn

              "vld1.s8    {d2}, [%[ktmp]]!     \n"  // d2=k01234567

              "vld1.s8    {d5[]}, [%[r3]]    \n"  // d5=r32
              "sub        %[r3], #2          \n"

              "veor       d3, d3          \n"

              "vmull.s8   q8, d0, d2      \n"
              "vmull.s8   q9, d1, d2      \n"

              "vld1.s8    {d3[0]}, [%[ktmp]]   \n"  // d3=k8 ... zeros
              "sub        %[ktmp], #8          \n"

              "vmlal.s8   q8, d4, d3      \n"
              "vmlal.s8   q9, d5, d3      \n"

              "vld1.s32   {d6[0]}, [%[outptr0]]   \n"

              "vadd.s16   d16, d16, d17   \n"
              "vadd.s16   d18, d18, d19   \n"

              "vld1.s32   {d6[1]}, [%[outptr0n]]   \n"

              "vpadd.s16  d16, d16, d18   \n"
              "vpadal.s16 d6, d16         \n"

              "vst1.s32   {d6[0]}, [%[outptr0]]!  \n"
              "vst1.s32   {d6[1]}, [%[outptr0n]]!  \n"

              : [outptr0] "+r"(outptr0), [outptr0n] "+r"(outptr0n),
                [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), [r3] "+r"(r3),
                [ktmp] "+r"(ktmp)
              :
              : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9");
          r0++;
          r1++;
          r2++;
          r3++;
        }

        r0 += 2 + w;
        r1 += 2 + w;
        r2 += 2 + w;
        r3 += 2 + w;

        outptr0 += outw;
        outptr0n += outw;
      }

      for (; i < outh; i++) {
        int nn = outw >> 3;
        int remain = outw & 7;
        if (nn > 0) {
          asm volatile(
              "0:                         \n"

              "pld        [%[r0], #128]      \n"
              "vld1.s8    {d4-d5}, [%[r0]]   \n"  // d4=r00 d5=r00n
              "add        %[r0], #8          \n"

              "vext.s8    d8, d4, d5, #1  \n"  // d8=r01

              "vmull.s8   q8, d4, %[_k00]    \n"

              "vext.s8    d9, d4, d5, #2  \n"  // d9=r02

              "vmull.s8   q9, d8, %[_k01]    \n"

              "pld        [%[r1], #128]      \n"
              "vld1.s8    {d6-d7}, [%[r1]]   \n"  // d6=r10 d7=r10n
              "add        %[r1], #8          \n"

              "vmlal.s8   q8, d9, %[_k02]    \n"

              "vext.s8    d10, d6, d7, #1 \n"  // d10=r11

              "vmlal.s8   q9, d6, %[_k10]    \n"

              "vext.s8    d11, d6, d7, #2 \n"  // d11=r12

              "vmlal.s8   q8, d10, %[_k11]   \n"

              "pld        [%[r2], #128]      \n"
              "vld1.s8    {d4-d5}, [%[r2]]   \n"  // d4=r20 d5=r20n
              "add        %[r2], #8          \n"

              "vmlal.s8   q9, d11, %[_k12]   \n"

              "vext.s8    d8, d4, d5, #1  \n"  // d8=r21

              "vmlal.s8   q8, d4, %[_k20]    \n"

              "vext.s8    d9, d4, d5, #2  \n"  // d9=r22

              "vmlal.s8   q9, d8, %[_k21]    \n"

              "vmlal.s8   q8, d9, %[_k22]    \n"

              "pld        [%[outptr0], #256]      \n"
              "vld1.s32   {d0-d3}, [%[outptr0]]   \n"

              "vadd.s16   q8, q8, q9      \n"

              "vaddw.s16  q0, q0, d16     \n"
              "vaddw.s16  q1, q1, d17     \n"

              "subs       %[nn], #1          \n"

              "vst1.s32   {d0-d3}, [%[outptr0]]!  \n"

              "bne        0b              \n"

              : [nn] "+r"(nn), [outptr0] "+r"(outptr0), [r0] "+r"(r0),
                [r1] "+r"(r1), [r2] "+r"(r2)
              : [_k00] "w"(_k00), [_k01] "w"(_k01), [_k02] "w"(_k02),
                [_k10] "w"(_k10), [_k11] "w"(_k11), [_k12] "w"(_k12),
                [_k20] "w"(_k20), [_k21] "w"(_k21), [_k22] "w"(_k22)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q8", "q9");
        }

        for (; remain > 0; remain--) {
          int sum0 = 0;
          sum0 += r0[0] * ktmp[0];
          sum0 += r0[1] * ktmp[1];
          sum0 += r0[2] * ktmp[2];
          sum0 += r1[0] * ktmp[3];
          sum0 += r1[1] * ktmp[4];
          sum0 += r1[2] * ktmp[5];
          sum0 += r2[0] * ktmp[6];
          sum0 += r2[1] * ktmp[7];
          sum0 += r2[2] * ktmp[8];

          *outptr0 += sum0;

          r0++;
          r1++;
          r2++;
          outptr0++;
        }

        r0 += 2;
        r1 += 2;
        r2 += 2;
      }

      ktmp += 9;
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
