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

#include "operators/math/depthwise_conv3x3.h"

namespace paddle_mobile {
namespace operators {
namespace math {

// template<>
// void DepthwiseConv3x3<int8_t, int32_t>(
//     const framework::Tensor *input, const framework::Tensor *filter,
//     const std::vector<int> &strides, framework::Tensor *output) {
//   PADDLE_MOBILE_THROW_EXCEPTION(
//       "Depthwise conv with generic strides has not been implemented.");
// }

template <>
void DepthwiseConv3x3s1<int8_t, int32_t>(const framework::Tensor &input,
                                         const framework::Tensor &filter,
                                         const std::vector<int> &paddings,
                                         framework::Tensor *output) {
  const int8_t *input_data = input.data<int8_t>();
  const int8_t *filter_data = filter.data<int8_t>();
  int32_t *out_data = output->mutable_data<int32_t>();
  // make sure that batch size is 1
  int input_c = input.dims()[1];
  int input_h = input.dims()[2];
  int input_w = input.dims()[3];
  int output_c = output->dims()[1];
  int output_h = output->dims()[2];
  int output_w = output->dims()[3];
  int image_size = input_h * input_w;
  int out_image_size = output_h * output_w;
#if __aarch64__
  // TODO(hjchen2)
#else
  #pragma omp parallel for
  for (int g = 0; g < input_c; ++g) {
    const int8_t* input_ptr = input_data + g * image_size;
    const int8_t* filter_ptr = filter_data + g * 9;
    int32_t* output_ptr = out_data + g * out_image_size;
    int loops = (input_w - 2) / 6;
    int remain = input_w - 2 - loops * 6;
    for (int h = 0; h < input_h - 5 /*(input_h - 2) - 3*/; h += 4) {
      const int8_t* input_ptr0 = input_ptr + h * input_w;
      const int8_t* input_ptr1 = input_ptr0 + input_w;
      const int8_t* input_ptr2 = input_ptr1 + input_w;
      const int8_t* input_ptr3 = input_ptr2 + input_w;
      const int8_t* input_ptr4 = input_ptr3 + input_w;
      const int8_t* input_ptr5 = input_ptr4 + input_w;
      int32_t* output_ptr0 = output_ptr + h * output_w;
      int32_t* output_ptr1 = output_ptr0 + output_w;
      int32_t* output_ptr2 = output_ptr1 + output_w;
      int32_t* output_ptr3 = output_ptr2 + output_w;
      int loop = loops;
      asm volatile(
          "vld1.32    {q0}, [%[filter_ptr]] \n"
          "vmovl.s8   q14, d0               \n"
          "vmovl.s8   q15, d1               \n"
          "vdup.s16   d0, d28[0]            \n"
          "vdup.s16   d1, d28[1]            \n"
          "vdup.s16   d2, d28[2]            \n"
          "vdup.s16   d3, d28[3]            \n"
          "vdup.s16   d4, d29[0]            \n"
          "vdup.s16   d5, d29[1]            \n"
          "vdup.s16   d6, d29[2]            \n"
          "vdup.s16   d7, d29[3]            \n"
          "vdup.s16   d8, d30[0]            \n"
          :
          : [filter_ptr] "r"(filter_ptr)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q14", "q15");
      asm volatile(
          "mov        r0, #6                \n"
          "cmp        %[loop], #0           \n"
          "ble        start_remain_%=       \n"
          // loop 6 widths
          "loop_4h6w_%=:                          \n"
          "vld1.32    {d9}, [%[input_ptr0]], r0   \n"
          "vld1.32    {d10}, [%[input_ptr1]], r0  \n"
          "vld1.32    {d11}, [%[input_ptr2]], r0  \n"
          "vext.s8    d12, d9, d9, #1       \n"
          "vext.s8    d13, d9, d9, #2       \n"
          "vmovl.s8   q7, d9                \n"
          "vmovl.s8   q8, d12               \n"
          "vmovl.s8   q9, d13               \n"
          "vmull.s16  q10, d14, d0          \n"
          "vmlal.s16  q10, d16, d1          \n"
          "vmlal.s16  q10, d18, d2          \n"
          "vmull.s16  q11, d15, d0          \n"
          "vmlal.s16  q11, d17, d1          \n"
          "vmlal.s16  q11, d19, d2          \n"

          "vext.s8    d12, d10, d10, #1     \n"
          "vext.s8    d13, d10, d10, #2     \n"
          "vmovl.s8   q7, d10               \n"
          "vmovl.s8   q8, d12               \n"
          "vmovl.s8   q9, d13               \n"
          "vmlal.s16  q10, d14, d3          \n"
          "vmlal.s16  q10, d16, d4          \n"
          "vmlal.s16  q10, d18, d5          \n"
          "vmlal.s16  q11, d15, d3          \n"
          "vmlal.s16  q11, d17, d4          \n"
          "vmlal.s16  q11, d19, d5          \n"

          "vmull.s16  q12, d14, d0          \n"
          "vmlal.s16  q12, d16, d1          \n"
          "vmlal.s16  q12, d18, d2          \n"
          "vmull.s16  q13, d15, d0          \n"
          "vmlal.s16  q13, d17, d1          \n"
          "vmlal.s16  q13, d19, d2          \n"

          "vext.s8    d12, d11, d11, #1     \n"
          "vext.s8    d13, d11, d11, #2     \n"
          "vmovl.s8   q7, d11               \n"
          "vmovl.s8   q8, d12               \n"
          "vmovl.s8   q9, d13               \n"
          "vmlal.s16  q10, d14, d6          \n"
          "vmlal.s16  q10, d16, d7          \n"
          "vmlal.s16  q10, d18, d8          \n"
          "vmlal.s16  q11, d15, d6          \n"
          "vmlal.s16  q11, d17, d7          \n"
          "vmlal.s16  q11, d19, d8          \n"
          // store row 0, reuse q10/q11
          "vst1.32    {d20-d22}, [%[output_ptr0]]! \n"

          "vmlal.s16  q12, d14, d3          \n"
          "vmlal.s16  q12, d16, d4          \n"
          "vmlal.s16  q12, d18, d5          \n"
          "vmlal.s16  q13, d15, d3          \n"
          "vmlal.s16  q13, d17, d4          \n"
          "vmlal.s16  q13, d19, d5          \n"

          "vmull.s16  q14, d14, d0          \n"
          "vmlal.s16  q14, d16, d1          \n"
          "vmlal.s16  q14, d18, d2          \n"
          "vmull.s16  q15, d15, d0          \n"
          "vmlal.s16  q15, d17, d1          \n"
          "vmlal.s16  q15, d19, d2          \n"

          "vld1.32    {d9}, [%[input_ptr3]], r0    \n"
          "vld1.32    {d10}, [%[input_ptr4]], r0   \n"
          "vld1.32    {d11}, [%[input_ptr5]], r0   \n"
          "vext.s8    d12, d9, d9, #1       \n"
          "vext.s8    d13, d9, d9, #2       \n"
          "vmovl.s8   q7, d9                \n"
          "vmovl.s8   q8, d12               \n"
          "vmovl.s8   q9, d13               \n"
          "vmlal.s16  q12, d14, d6          \n"
          "vmlal.s16  q12, d16, d7          \n"
          "vmlal.s16  q12, d18, d8          \n"
          "vmlal.s16  q13, d15, d6          \n"
          "vmlal.s16  q13, d17, d7          \n"
          "vmlal.s16  q13, d19, d8          \n"
          // store row 1
          "vst1.32    {d24-d26}, [%[output_ptr1]]! \n"

          "vmlal.s16  q14, d14, d3          \n"
          "vmlal.s16  q14, d16, d4          \n"
          "vmlal.s16  q14, d18, d5          \n"
          "vmlal.s16  q15, d15, d3          \n"
          "vmlal.s16  q15, d17, d4          \n"
          "vmlal.s16  q15, d19, d5          \n"

          "vmull.s16  q10, d14, d0          \n"
          "vmlal.s16  q10, d16, d1          \n"
          "vmlal.s16  q10, d18, d2          \n"
          "vmull.s16  q11, d15, d0          \n"
          "vmlal.s16  q11, d17, d1          \n"
          "vmlal.s16  q11, d19, d2          \n"

          "vext.s8    d12, d10, d10, #1     \n"
          "vext.s8    d13, d10, d10, #2     \n"
          "vmovl.s8   q7, d10               \n"
          "vmovl.s8   q8, d12               \n"
          "vmovl.s8   q9, d13               \n"
          "vmlal.s16  q14, d14, d6          \n"
          "vmlal.s16  q14, d16, d7          \n"
          "vmlal.s16  q14, d18, d8          \n"
          "vmlal.s16  q15, d15, d6          \n"
          "vmlal.s16  q15, d17, d7          \n"
          "vmlal.s16  q15, d19, d8          \n"
          // store row 2
          "vst1.32    {d28-d30}, [%[output_ptr2]]! \n"

          "vmlal.s16  q10, d14, d3          \n"
          "vmlal.s16  q10, d16, d4          \n"
          "vmlal.s16  q10, d18, d5          \n"
          "vmlal.s16  q11, d15, d3          \n"
          "vmlal.s16  q11, d17, d4          \n"
          "vmlal.s16  q11, d19, d5          \n"

          "vext.s8    d12, d11, d11, #1     \n"
          "vext.s8    d13, d11, d11, #2     \n"
          "vmovl.s8   q7, d11               \n"
          "vmovl.s8   q8, d12               \n"
          "vmovl.s8   q9, d13               \n"
          "vmlal.s16  q10, d14, d6          \n"
          "vmlal.s16  q10, d16, d7          \n"
          "vmlal.s16  q10, d18, d8          \n"
          "vmlal.s16  q11, d15, d6          \n"
          "vmlal.s16  q11, d17, d7          \n"
          "vmlal.s16  q11, d19, d8          \n"
          // store row 3
          "vst1.32    {d20-d22}, [%[output_ptr3]]! \n"

          "subs       %[loop], #1           \n"
          "bne        loop_4h6w_%=          \n"

          "start_remain_%=:                 \n"
          "cmp        %[remain], #0         \n"
          "ble        end_%=                \n"

          "vld1.32    {d9}, [%[input_ptr0]] \n"
          "vmovl.s8   q7, d9                \n"
          "vext.s8    d9, d9, d9, #1        \n"
          "vmovl.s8   q8, d9                \n"
          "vext.s8    d9, d9, d9, #1        \n"
          "vmovl.s8   q9, d9                \n"
          "vmull.s16  q10, d14, d0          \n"
          "vmlal.s16  q10, d16, d1          \n"
          "vmlal.s16  q10, d18, d2          \n"
          "vld1.32    {d9}, [%[input_ptr1]] \n"
          "vmull.s16  q11, d15, d0          \n"
          "vmlal.s16  q11, d17, d1          \n"
          "vmlal.s16  q11, d19, d2          \n"

          "vmovl.s8   q7, d9                \n"
          "vext.s8    d9, d9, d9, #1        \n"
          "vmovl.s8   q8, d9                \n"
          "vext.s8    d9, d9, d9, #1        \n"
          "vmovl.s8   q9, d9                \n"
          "vmlal.s16  q10, d14, d3          \n"
          "vmlal.s16  q10, d16, d4          \n"
          "vmlal.s16  q10, d18, d5          \n"
          "vmlal.s16  q11, d15, d3          \n"
          "vmlal.s16  q11, d17, d4          \n"
          "vmlal.s16  q11, d19, d5          \n"

          "vmull.s16  q12, d14, d0          \n"
          "vmlal.s16  q12, d16, d1          \n"
          "vmlal.s16  q12, d18, d2          \n"
          "vld1.32    {d9}, [%[input_ptr2]] \n"
          "vmull.s16  q13, d15, d0          \n"
          "vmlal.s16  q13, d17, d1          \n"
          "vmlal.s16  q13, d19, d2          \n"

          "vmovl.s8   q7, d9                \n"
          "vext.s8    d9, d9, d9, #1        \n"
          "vmovl.s8   q8, d9                \n"
          "vext.s8    d9, d9, d9, #1        \n"
          "vmovl.s8   q9, d9                \n"
          "vmlal.s16  q10, d14, d6          \n"
          "vmlal.s16  q10, d16, d7          \n"
          "vmlal.s16  q10, d18, d8          \n"
          "vmlal.s16  q11, d15, d6          \n"
          "vmlal.s16  q11, d17, d7          \n"
          "vmlal.s16  q11, d19, d8          \n"

          "vmlal.s16  q12, d14, d3          \n"
          "vmlal.s16  q12, d16, d4          \n"
          "vmlal.s16  q12, d18, d5          \n"
          "vmlal.s16  q13, d15, d3          \n"
          "vmlal.s16  q13, d17, d4          \n"
          "vmlal.s16  q13, d19, d5          \n"

          "vmull.s16  q14, d14, d0          \n"
          "vmlal.s16  q14, d16, d1          \n"
          "vmlal.s16  q14, d18, d2          \n"
          "vld1.32    {d9}, [%[input_ptr3]] \n"
          "vmull.s16  q15, d15, d0          \n"
          "vmlal.s16  q15, d17, d1          \n"
          "vmlal.s16  q15, d19, d2          \n"

          "vmovl.s8   q7, d9                \n"
          "vext.s8    d9, d9, d9, #1        \n"
          "vmovl.s8   q8, d9                \n"
          "vext.s8    d9, d9, d9, #1        \n"
          "vmovl.s8   q9, d9                \n"
          "vmlal.s16  q12, d14, d6          \n"
          "vmlal.s16  q12, d16, d7          \n"
          "vmlal.s16  q12, d18, d8          \n"
          "vmlal.s16  q13, d15, d6          \n"
          "vmlal.s16  q13, d17, d7          \n"
          "vmlal.s16  q13, d19, d8          \n"

          "vmlal.s16  q14, d14, d3          \n"
          "vmlal.s16  q14, d16, d4          \n"
          "vmlal.s16  q14, d18, d5          \n"
          "vmlal.s16  q15, d15, d3          \n"
          "vmlal.s16  q15, d17, d4          \n"
          "vmlal.s16  q15, d19, d5          \n"

          "vmull.s16  q5, d14, d0           \n"
          "vmlal.s16  q5, d16, d1           \n"
          "vmlal.s16  q5, d18, d2           \n"
          "vld1.32    {d9}, [%[input_ptr4]] \n"
          "vmull.s16  q6, d15, d0           \n"
          "vmlal.s16  q6, d17, d1           \n"
          "vmlal.s16  q6, d19, d2           \n"

          "vmovl.s8   q7, d9                \n"
          "vext.s8    d9, d9, d9, #1        \n"
          "vmovl.s8   q8, d9                \n"
          "vext.s8    d9, d9, d9, #1        \n"
          "vmovl.s8   q9, d9                \n"
          "vmlal.s16  q14, d14, d6          \n"
          "vmlal.s16  q14, d16, d7          \n"
          "vmlal.s16  q14, d18, d8          \n"
          "vmlal.s16  q15, d15, d6          \n"
          "vmlal.s16  q15, d17, d7          \n"
          "vmlal.s16  q15, d19, d8          \n"

          "vmlal.s16  q5, d14, d3           \n"
          "vmlal.s16  q5, d16, d4           \n"
          "vmlal.s16  q5, d18, d5           \n"
          "vld1.32    {d9}, [%[input_ptr5]] \n"
          "vmlal.s16  q6, d15, d3           \n"
          "vmlal.s16  q6, d17, d4           \n"
          "vmlal.s16  q6, d19, d5           \n"

          "vmovl.s8   q7, d9                \n"
          "vext.s8    d9, d9, d9, #1        \n"
          "vmovl.s8   q8, d9                \n"
          "vext.s8    d9, d9, d9, #1        \n"
          "vmovl.s8   q9, d9                \n"
          "vmlal.s16  q5, d14, d6           \n"
          "vmlal.s16  q5, d16, d7           \n"
          "vmlal.s16  q5, d18, d8           \n"
          "vmlal.s16  q6, d15, d6           \n"
          "vmlal.s16  q6, d17, d7           \n"
          "vmlal.s16  q6, d19, d8           \n"

          "cmp        %[remain], #4               \n"
          "blt        store_4h2w_%=               \n"
          "vst1.32    {q10}, [%[output_ptr0]]!    \n"
          "vst1.32    {q12}, [%[output_ptr1]]!    \n"
          "vst1.32    {q14}, [%[output_ptr2]]!    \n"
          "vst1.32    {q5}, [%[output_ptr3]]!     \n"
          "cmp        %[remain], #5               \n"
          "blt        end_%=                      \n"
          "vst1.32    {d22[0]}, [%[output_ptr0]]! \n"
          "vst1.32    {d26[0]}, [%[output_ptr1]]! \n"
          "vst1.32    {d30[0]}, [%[output_ptr2]]! \n"
          "vst1.32    {d12[0]}, [%[output_ptr3]]! \n"
          "b          end_%=                      \n"

          "store_4h2w_%=:                         \n"
          "cmp        %[remain], #2               \n"
          "blt        store_4h1w_%=               \n"
          "vst1.32    {d20}, [%[output_ptr0]]!    \n"
          "vst1.32    {d24}, [%[output_ptr1]]!    \n"
          "vst1.32    {d28}, [%[output_ptr2]]!    \n"
          "vst1.32    {d10}, [%[output_ptr3]]!    \n"
          "cmp        %[remain], #3               \n"
          "blt        end_%=                      \n"
          "vst1.32    {d21[0]}, [%[output_ptr0]]! \n"
          "vst1.32    {d25[0]}, [%[output_ptr1]]! \n"
          "vst1.32    {d29[0]}, [%[output_ptr2]]! \n"
          "vst1.32    {d11[0]}, [%[output_ptr3]]! \n"
          "b          end_%=                      \n"

          "store_4h1w_%=:                         \n"
          "cmp        %[remain], #1               \n"
          "blt        end_%=                      \n"
          "vst1.32    {d20[0]}, [%[output_ptr0]]! \n"
          "vst1.32    {d24[0]}, [%[output_ptr1]]! \n"
          "vst1.32    {d28[0]}, [%[output_ptr2]]! \n"
          "vst1.32    {d10[0]}, [%[output_ptr3]]! \n"
          "end_%=:                                \n"
          : [output_ptr0] "+r"(output_ptr0), [output_ptr1] "+r"(output_ptr1),
            [output_ptr2] "+r"(output_ptr2), [output_ptr3] "+r"(output_ptr3),
            [input_ptr0] "+r"(input_ptr0), [input_ptr1] "+r"(input_ptr1),
            [input_ptr2] "+r"(input_ptr2), [input_ptr3] "+r"(input_ptr3),
            [input_ptr4] "+r"(input_ptr4), [input_ptr5] "+r"(input_ptr5),
            [loop] "+r"(loop)
          : [remain] "r"(remain)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "r0");
    }
    // remain height
    int start_h = (input_h - 2) & 0xFFFC;
    for (int h = start_h; h < input_h - 3 /*(input_h - 2) - 1*/; h += 2) {
      const int8_t* input_ptr0 = input_ptr + h * input_w;
      const int8_t* input_ptr1 = input_ptr0 + input_w;
      const int8_t* input_ptr2 = input_ptr1 + input_w;
      const int8_t* input_ptr3 = input_ptr2 + input_w;
      int32_t* output_ptr0 = output_ptr + h * output_w;
      int32_t* output_ptr1 = output_ptr0 + output_w;
      int loop = loops;
      asm volatile(
          "vld1.32    {q0}, [%[filter_ptr]] \n"
          "vmovl.s8   q14, d0               \n"
          "vmovl.s8   q15, d1               \n"
          "vdup.s16   d0, d28[0]            \n"
          "vdup.s16   d1, d28[1]            \n"
          "vdup.s16   d2, d28[2]            \n"
          "vdup.s16   d3, d28[3]            \n"
          "vdup.s16   d4, d29[0]            \n"
          "vdup.s16   d5, d29[1]            \n"
          "vdup.s16   d6, d29[2]            \n"
          "vdup.s16   d7, d29[3]            \n"
          "vdup.s16   d8, d30[0]            \n"
          :
          : [filter_ptr] "r"(filter_ptr)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q14", "q15");
      asm volatile(
          "mov        r0, #6                \n"
          "cmp        %[loop], #0           \n"
          "ble        start_remain_%=       \n"
          // loop 6 widths
          "loop_2h6w_%=:                          \n"
          "vld1.32    {d9}, [%[input_ptr0]], r0   \n"
          "vld1.32    {d10}, [%[input_ptr1]], r0  \n"
          "vld1.32    {d11}, [%[input_ptr2]], r0  \n"
          "vext.s8    d12, d9, d9, #1       \n"
          "vext.s8    d13, d9, d9, #2       \n"
          "vmovl.s8   q7, d9                \n"
          "vmovl.s8   q8, d12               \n"
          "vmovl.s8   q9, d13               \n"
          "vmull.s16  q10, d14, d0          \n"
          "vmlal.s16  q10, d16, d1          \n"
          "vmlal.s16  q10, d18, d2          \n"
          "vmull.s16  q11, d15, d0          \n"
          "vmlal.s16  q11, d17, d1          \n"
          "vmlal.s16  q11, d19, d2          \n"

          "vext.s8    d12, d10, d10, #1     \n"
          "vext.s8    d13, d10, d10, #2     \n"
          "vmovl.s8   q7, d10               \n"
          "vmovl.s8   q8, d12               \n"
          "vmovl.s8   q9, d13               \n"
          "vmlal.s16  q10, d14, d3          \n"
          "vmlal.s16  q10, d16, d4          \n"
          "vmlal.s16  q10, d18, d5          \n"
          "vmlal.s16  q11, d15, d3          \n"
          "vmlal.s16  q11, d17, d4          \n"
          "vmlal.s16  q11, d19, d5          \n"

          "vmull.s16  q12, d14, d0          \n"
          "vmlal.s16  q12, d16, d1          \n"
          "vmlal.s16  q12, d18, d2          \n"
          "vmull.s16  q13, d15, d0          \n"
          "vmlal.s16  q13, d17, d1          \n"
          "vmlal.s16  q13, d19, d2          \n"

          "vext.s8    d12, d11, d11, #1     \n"
          "vext.s8    d13, d11, d11, #2     \n"
          "vmovl.s8   q7, d11               \n"
          "vmovl.s8   q8, d12               \n"
          "vmovl.s8   q9, d13               \n"
          "vmlal.s16  q10, d14, d6          \n"
          "vmlal.s16  q10, d16, d7          \n"
          "vmlal.s16  q10, d18, d8          \n"
          "vmlal.s16  q11, d15, d6          \n"
          "vmlal.s16  q11, d17, d7          \n"
          "vmlal.s16  q11, d19, d8          \n"
          // store row 0, reuse q10/q11
          "vst1.32    {d20-d22}, [%[output_ptr0]]! \n"

          "vmlal.s16  q12, d14, d3          \n"
          "vmlal.s16  q12, d16, d4          \n"
          "vmlal.s16  q12, d18, d5          \n"
          "vmlal.s16  q13, d15, d3          \n"
          "vmlal.s16  q13, d17, d4          \n"
          "vmlal.s16  q13, d19, d5          \n"

          "vld1.32    {d9}, [%[input_ptr3]], r0    \n"
          "vext.s8    d12, d9, d9, #1       \n"
          "vext.s8    d13, d9, d9, #2       \n"
          "vmovl.s8   q7, d9                \n"
          "vmovl.s8   q8, d12               \n"
          "vmovl.s8   q9, d13               \n"
          "vmlal.s16  q12, d14, d6          \n"
          "vmlal.s16  q12, d16, d7          \n"
          "vmlal.s16  q12, d18, d8          \n"
          "vmlal.s16  q13, d15, d6          \n"
          "vmlal.s16  q13, d17, d7          \n"
          "vmlal.s16  q13, d19, d8          \n"
          // store row 1
          "vst1.32    {d24-d26}, [%[output_ptr1]]! \n"

          "subs       %[loop], #1            \n"
          "bne        loop_2h6w_%=           \n"

          "start_remain_%=:                  \n"
          "cmp %[remain], #0                 \n"
          "ble end_%=                        \n"

          "vld1.32    {d9}, [%[input_ptr0]]  \n"
          "vld1.32    {d10}, [%[input_ptr1]] \n"
          "vld1.32    {d11}, [%[input_ptr2]] \n"
          "vext.s8    d12, d9, d9, #1        \n"
          "vext.s8    d13, d9, d9, #2        \n"
          "vmovl.s8   q7, d9                 \n"
          "vmovl.s8   q8, d12                \n"
          "vmovl.s8   q9, d13                \n"
          "vmull.s16  q10, d14, d0           \n"
          "vmlal.s16  q10, d16, d1           \n"
          "vmlal.s16  q10, d18, d2           \n"
          "vmull.s16  q11, d15, d0           \n"
          "vmlal.s16  q11, d17, d1           \n"
          "vmlal.s16  q11, d19, d2           \n"

          "vext.s8    d12, d10, d10, #1      \n"
          "vext.s8    d13, d10, d10, #2      \n"
          "vmovl.s8   q7, d10                \n"
          "vmovl.s8   q8, d12                \n"
          "vmovl.s8   q9, d13                \n"
          "vmlal.s16  q10, d14, d3           \n"
          "vmlal.s16  q10, d16, d4           \n"
          "vmlal.s16  q10, d18, d5           \n"
          "vmlal.s16  q11, d15, d3           \n"
          "vmlal.s16  q11, d17, d4           \n"
          "vmlal.s16  q11, d19, d5           \n"

          "vmull.s16  q12, d14, d0           \n"
          "vmlal.s16  q12, d16, d1           \n"
          "vmlal.s16  q12, d18, d2           \n"
          "vmull.s16  q13, d15, d0           \n"
          "vmlal.s16  q13, d17, d1           \n"
          "vmlal.s16  q13, d19, d2           \n"

          "vext.s8    d12, d11, d11, #1      \n"
          "vext.s8    d13, d11, d11, #2      \n"
          "vmovl.s8   q7, d11                \n"
          "vmovl.s8   q8, d12                \n"
          "vmovl.s8   q9, d13                \n"
          "vmlal.s16  q10, d14, d6           \n"
          "vmlal.s16  q10, d16, d7           \n"
          "vmlal.s16  q10, d18, d8           \n"
          "vmlal.s16  q11, d15, d6           \n"
          "vmlal.s16  q11, d17, d7           \n"
          "vmlal.s16  q11, d19, d8           \n"

          "vmlal.s16  q12, d14, d3           \n"
          "vmlal.s16  q12, d16, d4           \n"
          "vmlal.s16  q12, d18, d5           \n"
          "vmlal.s16  q13, d15, d3           \n"
          "vmlal.s16  q13, d17, d4           \n"
          "vmlal.s16  q13, d19, d5           \n"

          "vld1.32    {d9}, [%[input_ptr3]]  \n"
          "vext.s8    d12, d9, d9, #1        \n"
          "vext.s8    d13, d9, d9, #2        \n"
          "vmovl.s8   q7, d9                 \n"
          "vmovl.s8   q8, d12                \n"
          "vmovl.s8   q9, d13                \n"
          "vmlal.s16  q12, d14, d6           \n"
          "vmlal.s16  q12, d16, d7           \n"
          "vmlal.s16  q12, d18, d8           \n"
          "vmlal.s16  q13, d15, d6           \n"
          "vmlal.s16  q13, d17, d7           \n"
          "vmlal.s16  q13, d19, d8           \n"

          "cmp        %[remain], #4               \n"
          "blt        store_2h2w_%=               \n"
          "vst1.32    {q10}, [%[output_ptr0]]!    \n"
          "vst1.32    {q12}, [%[output_ptr1]]!    \n"
          "cmp        %[remain], #5               \n"
          "blt        end_%=                      \n"
          "vst1.32    {d22[0]}, [%[output_ptr0]]! \n"
          "vst1.32    {d26[0]}, [%[output_ptr1]]! \n"
          "b          end_%=                      \n"

          "store_2h2w_%=:                         \n"
          "cmp        %[remain], #2               \n"
          "blt        store_2h1w_%=               \n"
          "vst1.32    {d20}, [%[output_ptr0]]!    \n"
          "vst1.32    {d24}, [%[output_ptr1]]!    \n"
          "cmp        %[remain], #3               \n"
          "blt        end_%=                      \n"
          "vst1.32    {d21[0]}, [%[output_ptr0]]! \n"
          "vst1.32    {d25[0]}, [%[output_ptr1]]! \n"
          "b          end_%=                      \n"

          "store_2h1w_%=:                         \n"
          "cmp        %[remain], #1               \n"
          "blt        end_%=                      \n"
          "vst1.32    {d20[0]}, [%[output_ptr0]]! \n"
          "vst1.32    {d24[0]}, [%[output_ptr1]]! \n"
          "end_%=:                                \n"
          : [output_ptr0] "+r"(output_ptr0), [output_ptr1] "+r"(output_ptr1),
            [input_ptr0] "+r"(input_ptr0), [input_ptr1] "+r"(input_ptr1),
            [input_ptr2] "+r"(input_ptr2), [input_ptr3] "+r"(input_ptr3),
            [loop] "+r"(loop)
          : [remain] "r"(remain)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
            "q8", "q9", "q10", "q11", "q12", "q13", "r0");
    }

    start_h = (input_h - 2) & 0xFFFE;
    if (start_h < input_h - 2) {
      const int8_t* input_ptr0 = input_ptr + start_h * input_w;
      const int8_t* input_ptr1 = input_ptr0 + input_w;
      const int8_t* input_ptr2 = input_ptr1 + input_w;
      int32_t* output_ptr0 = output_ptr + start_h * output_w;
      int loop = loops;
      asm volatile(
          "vld1.32    {q0}, [%[filter_ptr]] \n"
          "vmovl.s8   q14, d0               \n"
          "vmovl.s8   q15, d1               \n"
          "vdup.s16   d0, d28[0]            \n"
          "vdup.s16   d1, d28[1]            \n"
          "vdup.s16   d2, d28[2]            \n"
          "vdup.s16   d3, d28[3]            \n"
          "vdup.s16   d4, d29[0]            \n"
          "vdup.s16   d5, d29[1]            \n"
          "vdup.s16   d6, d29[2]            \n"
          "vdup.s16   d7, d29[3]            \n"
          "vdup.s16   d8, d30[0]            \n"
          :
          : [filter_ptr] "r"(filter_ptr)
          : "memory", "q0", "q1", "q2", "q3", "q4", "q14", "q15");
      asm volatile(
          "mov        r0, #6                \n"
          "cmp        %[loop], #0           \n"
          "ble        start_remain_%=       \n"
          // loop 6 widths
          "loop_1h6w_%=:                          \n"
          "vld1.32    {d9}, [%[input_ptr0]], r0   \n"
          "vld1.32    {d10}, [%[input_ptr1]], r0  \n"
          "vld1.32    {d11}, [%[input_ptr2]], r0  \n"
          "vext.s8    d12, d9, d9, #1       \n"
          "vext.s8    d13, d9, d9, #2       \n"
          "vmovl.s8   q7, d9                \n"
          "vmovl.s8   q8, d12               \n"
          "vmovl.s8   q9, d13               \n"
          "vmull.s16  q10, d14, d0          \n"
          "vmlal.s16  q10, d16, d1          \n"
          "vmlal.s16  q10, d18, d2          \n"
          "vmull.s16  q11, d15, d0          \n"
          "vmlal.s16  q11, d17, d1          \n"
          "vmlal.s16  q11, d19, d2          \n"

          "vext.s8    d12, d10, d10, #1     \n"
          "vext.s8    d13, d10, d10, #2     \n"
          "vmovl.s8   q7, d10               \n"
          "vmovl.s8   q8, d12               \n"
          "vmovl.s8   q9, d13               \n"
          "vmlal.s16  q10, d14, d3          \n"
          "vmlal.s16  q10, d16, d4          \n"
          "vmlal.s16  q10, d18, d5          \n"
          "vmlal.s16  q11, d15, d3          \n"
          "vmlal.s16  q11, d17, d4          \n"
          "vmlal.s16  q11, d19, d5          \n"

          "vext.s8    d12, d11, d11, #1     \n"
          "vext.s8    d13, d11, d11, #2     \n"
          "vmovl.s8   q7, d11               \n"
          "vmovl.s8   q8, d12               \n"
          "vmovl.s8   q9, d13               \n"
          "vmlal.s16  q10, d14, d6          \n"
          "vmlal.s16  q10, d16, d7          \n"
          "vmlal.s16  q10, d18, d8          \n"
          "vmlal.s16  q11, d15, d6          \n"
          "vmlal.s16  q11, d17, d7          \n"
          "vmlal.s16  q11, d19, d8          \n"
          // store row 0, reuse q10/q11
          "vst1.32    {d20-d22}, [%[output_ptr0]]! \n"

          "subs       %[loop], #1            \n"
          "bne        loop_1h6w_%=           \n"

          "start_remain_%=:                  \n"
          "cmp %[remain], #0                 \n"
          "ble end_%=                        \n"

          "vld1.32    {d9}, [%[input_ptr0]]  \n"
          "vld1.32    {d10}, [%[input_ptr1]] \n"
          "vld1.32    {d11}, [%[input_ptr2]] \n"
          "vext.s8    d12, d9, d9, #1        \n"
          "vext.s8    d13, d9, d9, #2        \n"
          "vmovl.s8   q7, d9                 \n"
          "vmovl.s8   q8, d12                \n"
          "vmovl.s8   q9, d13                \n"
          "vmull.s16  q10, d14, d0           \n"
          "vmlal.s16  q10, d16, d1           \n"
          "vmlal.s16  q10, d18, d2           \n"
          "vmull.s16  q11, d15, d0           \n"
          "vmlal.s16  q11, d17, d1           \n"
          "vmlal.s16  q11, d19, d2           \n"

          "vext.s8    d12, d10, d10, #1      \n"
          "vext.s8    d13, d10, d10, #2      \n"
          "vmovl.s8   q7, d10                \n"
          "vmovl.s8   q8, d12                \n"
          "vmovl.s8   q9, d13                \n"
          "vmlal.s16  q10, d14, d3           \n"
          "vmlal.s16  q10, d16, d4           \n"
          "vmlal.s16  q10, d18, d5           \n"
          "vmlal.s16  q11, d15, d3           \n"
          "vmlal.s16  q11, d17, d4           \n"
          "vmlal.s16  q11, d19, d5           \n"

          "vext.s8    d12, d11, d11, #1      \n"
          "vext.s8    d13, d11, d11, #2      \n"
          "vmovl.s8   q7, d11                \n"
          "vmovl.s8   q8, d12                \n"
          "vmovl.s8   q9, d13                \n"
          "vmlal.s16  q10, d14, d6           \n"
          "vmlal.s16  q10, d16, d7           \n"
          "vmlal.s16  q10, d18, d8           \n"
          "vmlal.s16  q11, d15, d6           \n"
          "vmlal.s16  q11, d17, d7           \n"
          "vmlal.s16  q11, d19, d8           \n"

          "cmp        %[remain], #4               \n"
          "blt        store_1h2w_%=               \n"
          "vst1.32    {q10}, [%[output_ptr0]]!    \n"
          "cmp        %[remain], #5               \n"
          "blt        end_%=                      \n"
          "vst1.32    {d22[0]}, [%[output_ptr0]]! \n"
          "b          end_%=                      \n"

          "store_1h2w_%=:                         \n"
          "cmp        %[remain], #2               \n"
          "blt        store_1h1w_%=               \n"
          "vst1.32    {d20}, [%[output_ptr0]]!    \n"
          "cmp        %[remain], #3               \n"
          "blt        end_%=                      \n"
          "vst1.32    {d21[0]}, [%[output_ptr0]]! \n"
          "b          end_%=                      \n"

          "store_1h1w_%=:                         \n"
          "cmp        %[remain], #1               \n"
          "blt        end_%=                      \n"
          "vst1.32    {d20[0]}, [%[output_ptr0]]! \n"
          "end_%=:                                \n"
          : [output_ptr0] "+r"(output_ptr0), [input_ptr0] "+r"(input_ptr0),
            [input_ptr1] "+r"(input_ptr1), [input_ptr2] "+r"(input_ptr2),
            [loop] "+r"(loop)
          : [remain] "r"(remain)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
            "q8", "q9", "q10", "q11", "r0");
    }
  }
#endif  // __aarch64__
}

template <>
void DepthwiseConv3x3s2<int8_t, int32_t>(const framework::Tensor &input,
                                         const framework::Tensor &filter,
                                         const std::vector<int> &paddings,
                                         framework::Tensor *output) {
  const int8_t *input_data = input.data<int8_t>();
  const int8_t *filter_data = filter.data<int8_t>();
  int32_t *out_data = output->mutable_data<int32_t>();
  // make sure that batch size is 1
  int input_c = input.dims()[1];
  int input_h = input.dims()[2];
  int input_w = input.dims()[3];
  int output_c = output->dims()[1];
  int output_h = output->dims()[2];
  int output_w = output->dims()[3];
  int image_size = input_h * input_w;
  int out_image_size = output_h * output_w;
#if __aarch64__
  // TODO(hjchen2)
#else
  #pragma omp parallel for
  for (int g = 0; g < input_c; ++g) {
    const int8_t* input_ptr = input_data + g * image_size;
    const int8_t* filter_ptr = filter_data + g * 9;
    int32_t* output_ptr = out_data + g * out_image_size;
    int loops = output_w / 6;
    int remain = output_w - loops * 6;
    for (int h = 0; h < input_h - 6 /*(input_h - 1) - 5*/; h += 6) {
      const int8_t* input_ptr0 = input_ptr + h * input_w;
      const int8_t* input_ptr1 = input_ptr0 + input_w;
      const int8_t* input_ptr2 = input_ptr1 + input_w;
      const int8_t* input_ptr3 = input_ptr2 + input_w;
      const int8_t* input_ptr4 = input_ptr3 + input_w;
      const int8_t* input_ptr5 = input_ptr4 + input_w;
      const int8_t* input_ptr6 = input_ptr5 + input_w;
      int32_t* output_ptr0 = output_ptr + (h >> 1) * output_w;
      int32_t* output_ptr1 = output_ptr0 + output_w;
      int32_t* output_ptr2 = output_ptr1 + output_w;
      int loop = loops;
      asm volatile(
          "vld1.32    {q0}, [%[filter_ptr]] \n"
          "vmovl.s8   q14, d0               \n"
          "vmovl.s8   q15, d1               \n"
          "vdup.s16   d0, d28[0]            \n"
          "vdup.s16   d1, d28[1]            \n"
          "vdup.s16   d2, d28[2]            \n"
          "vdup.s16   d3, d28[3]            \n"
          "vdup.s16   d4, d29[0]            \n"
          "vdup.s16   d5, d29[1]            \n"
          "vdup.s16   d6, d29[2]            \n"
          "vdup.s16   d7, d29[3]            \n"
          "vdup.s16   d8, d30[0]            \n"
          :
          : [filter_ptr] "r"(filter_ptr)
          : "memory", "q0", "q1", "q2", "q3", "q4", "q14", "q15");
      asm volatile(
          "mov        r0, #12               \n"
          "cmp        %[loop], #0           \n"
          "ble        start_remain_%=       \n"
          // loop 6 widths
          "loop_3h6w_%=:                               \n"
          "vld2.8     {d10, d11}, [%[input_ptr0]], r0  \n"
          "vld2.8     {d12, d13}, [%[input_ptr1]], r0  \n"
          "vld2.8     {d14, d15}, [%[input_ptr2]], r0  \n"
          "vext.s8    d9, d10, d10, #1      \n"
          "vmovl.s8   q10, d9               \n"
          "vmovl.s8   q8, d10               \n"
          "vmovl.s8   q9, d11               \n"
          "vmull.s16  q11, d16, d0          \n"
          "vmlal.s16  q11, d18, d1          \n"
          "vmlal.s16  q11, d20, d2          \n"
          "vmull.s16  q12, d17, d0          \n"
          "vmlal.s16  q12, d19, d1          \n"
          "vmlal.s16  q12, d21, d2          \n"

          "vext.s8    d9, d12, d12, #1      \n"
          "vmovl.s8   q10, d9               \n"
          "vmovl.s8   q8, d12               \n"
          "vmovl.s8   q9, d13               \n"
          "vmlal.s16  q11, d16, d3          \n"
          "vmlal.s16  q11, d18, d4          \n"
          "vmlal.s16  q11, d20, d5          \n"
          "vmlal.s16  q12, d17, d3          \n"
          "vmlal.s16  q12, d19, d4          \n"
          "vmlal.s16  q12, d21, d5          \n"

          "vext.s8    d9, d14, d14, #1      \n"
          "vmovl.s8   q10, d9               \n"
          "vmovl.s8   q8, d14               \n"
          "vmovl.s8   q9, d15               \n"
          "vmlal.s16  q11, d16, d6          \n"
          "vmlal.s16  q11, d18, d7          \n"
          "vmlal.s16  q11, d20, d8          \n"
          "vmlal.s16  q12, d17, d6          \n"
          "vmlal.s16  q12, d19, d7          \n"
          "vmlal.s16  q12, d21, d8          \n"
          // store row 0, reuse q11/q12
          "vst1.32    {d22-d24}, [%[output_ptr0]]! \n"

          "vmull.s16  q13, d16, d0          \n"
          "vmlal.s16  q13, d18, d1          \n"
          "vmlal.s16  q13, d20, d2          \n"
          "vmull.s16  q14, d17, d0          \n"
          "vmlal.s16  q14, d19, d1          \n"
          "vmlal.s16  q14, d21, d2          \n"

          "vld2.8     {d10, d11}, [%[input_ptr3]], r0  \n"
          "vld2.8     {d12, d13}, [%[input_ptr4]], r0  \n"
          "vld2.8     {d14, d15}, [%[input_ptr5]], r0  \n"
          "vext.s8    d9, d10, d10, #1      \n"
          "vmovl.s8   q10, d9               \n"
          "vmovl.s8   q8, d10               \n"
          "vmovl.s8   q9, d11               \n"
          "vmlal.s16  q13, d16, d3          \n"
          "vmlal.s16  q13, d18, d4          \n"
          "vmlal.s16  q13, d20, d5          \n"
          "vmlal.s16  q14, d17, d3          \n"
          "vmlal.s16  q14, d19, d4          \n"
          "vmlal.s16  q14, d21, d5          \n"

          "vext.s8    d9, d12, d12, #1      \n"
          "vmovl.s8   q10, d9               \n"
          "vmovl.s8   q8, d12               \n"
          "vmovl.s8   q9, d13               \n"
          "vmlal.s16  q13, d16, d6          \n"
          "vmlal.s16  q13, d18, d7          \n"
          "vmlal.s16  q13, d20, d8          \n"
          "vmlal.s16  q14, d17, d6          \n"
          "vmlal.s16  q14, d19, d7          \n"
          "vmlal.s16  q14, d21, d8          \n"
          // store row 1
          "vst1.32    {d26-d28}, [%[output_ptr1]]! \n"

          "vmull.s16  q11, d16, d0          \n"
          "vmlal.s16  q11, d18, d1          \n"
          "vmlal.s16  q11, d20, d2          \n"
          "vmull.s16  q12, d17, d0          \n"
          "vmlal.s16  q12, d19, d1          \n"
          "vmlal.s16  q12, d21, d2          \n"

          "vext.s8    d9, d14, d14, #1      \n"
          "vmovl.s8   q10, d9               \n"
          "vmovl.s8   q8, d14               \n"
          "vmovl.s8   q9, d15               \n"
          "vmlal.s16  q11, d16, d3          \n"
          "vmlal.s16  q11, d18, d4          \n"
          "vmlal.s16  q11, d20, d5          \n"
          "vmlal.s16  q12, d17, d3          \n"
          "vmlal.s16  q12, d19, d4          \n"
          "vmlal.s16  q12, d21, d5          \n"

          "vld2.8     {d10, d11}, [%[input_ptr6]], r0  \n"
          "vext.s8    d9, d10, d10, #1      \n"
          "vmovl.s8   q10, d9               \n"
          "vmovl.s8   q8, d10               \n"
          "vmovl.s8   q9, d11               \n"
          "vmlal.s16  q11, d16, d6          \n"
          "vmlal.s16  q11, d18, d7          \n"
          "vmlal.s16  q11, d20, d8          \n"
          "vmlal.s16  q12, d17, d6          \n"
          "vmlal.s16  q12, d19, d7          \n"
          "vmlal.s16  q12, d21, d8          \n"
          // store row 2
          "vst1.32    {d22-d24}, [%[output_ptr2]]! \n"

          "subs       %[loop], #1           \n"
          "bne        loop_3h6w_%=          \n"

          "start_remain_%=:                 \n"
          "cmp        %[remain], #0         \n"
          "ble        end_%=                \n"

          "vld2.8     {d10, d11}, [%[input_ptr0]]  \n"
          "vld2.8     {d12, d13}, [%[input_ptr1]]  \n"
          "vext.s8    d9, d10, d10, #1      \n"
          "vmovl.s8   q9, d9                \n"
          "vmovl.s8   q7, d10               \n"
          "vmovl.s8   q8, d11               \n"
          "vmull.s16  q10, d14, d0          \n"
          "vmlal.s16  q10, d16, d1          \n"
          "vmlal.s16  q10, d18, d2          \n"
          "vmull.s16  q11, d15, d0          \n"
          "vmlal.s16  q11, d17, d1          \n"
          "vmlal.s16  q11, d19, d2          \n"

          "vext.s8    d9, d12, d12, #1      \n"
          "vmovl.s8   q9, d9                \n"
          "vmovl.s8   q7, d12               \n"
          "vmovl.s8   q8, d13               \n"
          "vmlal.s16  q10, d14, d3          \n"
          "vmlal.s16  q10, d16, d4          \n"
          "vmlal.s16  q10, d18, d5          \n"
          "vmlal.s16  q11, d15, d3          \n"
          "vmlal.s16  q11, d17, d4          \n"
          "vmlal.s16  q11, d19, d5          \n"

          "vld2.8     {d10, d11}, [%[input_ptr2]]  \n"
          "vld2.8     {d12, d13}, [%[input_ptr3]]  \n"
          "vext.s8    d9, d10, d10, #1      \n"
          "vmovl.s8   q9, d9                \n"
          "vmovl.s8   q7, d10               \n"
          "vmovl.s8   q8, d11               \n"
          "vmlal.s16  q10, d14, d6          \n"
          "vmlal.s16  q10, d16, d7          \n"
          "vmlal.s16  q10, d18, d8          \n"
          "vmlal.s16  q11, d15, d6          \n"
          "vmlal.s16  q11, d17, d7          \n"
          "vmlal.s16  q11, d19, d8          \n"

          "vmull.s16  q12, d14, d0          \n"
          "vmlal.s16  q12, d16, d1          \n"
          "vmlal.s16  q12, d18, d2          \n"
          "vmull.s16  q13, d15, d0          \n"
          "vmlal.s16  q13, d17, d1          \n"
          "vmlal.s16  q13, d19, d2          \n"

          "vext.s8    d9, d12, d12, #1      \n"
          "vmovl.s8   q9, d9                \n"
          "vmovl.s8   q7, d12               \n"
          "vmovl.s8   q8, d13               \n"
          "vmlal.s16  q12, d14, d3          \n"
          "vmlal.s16  q12, d16, d4          \n"
          "vmlal.s16  q12, d18, d5          \n"
          "vmlal.s16  q13, d15, d3          \n"
          "vmlal.s16  q13, d17, d4          \n"
          "vmlal.s16  q13, d19, d5          \n"

          "vld2.8     {d10, d11}, [%[input_ptr4]]  \n"
          "vld2.8     {d12, d13}, [%[input_ptr5]]  \n"
          "vext.s8    d9, d10, d10, #1      \n"
          "vmovl.s8   q9, d9                \n"
          "vmovl.s8   q7, d10               \n"
          "vmovl.s8   q8, d11               \n"
          "vmlal.s16  q12, d14, d6          \n"
          "vmlal.s16  q12, d16, d7          \n"
          "vmlal.s16  q12, d18, d8          \n"
          "vmlal.s16  q13, d15, d6          \n"
          "vmlal.s16  q13, d17, d7          \n"
          "vmlal.s16  q13, d19, d8          \n"

          "vmull.s16  q14, d14, d0          \n"
          "vmlal.s16  q14, d16, d1          \n"
          "vmlal.s16  q14, d18, d2          \n"
          "vmull.s16  q15, d15, d0          \n"
          "vmlal.s16  q15, d17, d1          \n"
          "vmlal.s16  q15, d19, d2          \n"

          "vext.s8    d9, d12, d12, #1      \n"
          "vmovl.s8   q9, d9                \n"
          "vmovl.s8   q7, d12               \n"
          "vmovl.s8   q8, d13               \n"
          "vmlal.s16  q14, d14, d3          \n"
          "vmlal.s16  q14, d16, d4          \n"
          "vmlal.s16  q14, d18, d5          \n"
          "vmlal.s16  q15, d15, d3          \n"
          "vmlal.s16  q15, d17, d4          \n"
          "vmlal.s16  q15, d19, d5          \n"

          "vld2.8     {d10, d11}, [%[input_ptr6]]  \n"
          "vext.s8    d9, d10, d10, #1      \n"
          "vmovl.s8   q9, d9                \n"
          "vmovl.s8   q7, d10               \n"
          "vmovl.s8   q8, d11               \n"
          "vmlal.s16  q14, d14, d6          \n"
          "vmlal.s16  q14, d16, d7          \n"
          "vmlal.s16  q14, d18, d8          \n"
          "vmlal.s16  q15, d15, d6          \n"
          "vmlal.s16  q15, d17, d7          \n"
          "vmlal.s16  q15, d19, d8          \n"

          "cmp        %[remain], #4               \n"
          "blt        store_3h2w_%=               \n"
          "vst1.32    {q10}, [%[output_ptr0]]!    \n"
          "vst1.32    {q12}, [%[output_ptr1]]!    \n"
          "vst1.32    {q14}, [%[output_ptr2]]!    \n"
          "cmp        %[remain], #5               \n"
          "blt        end_%=                      \n"
          "vst1.32    {d22[0]}, [%[output_ptr0]]! \n"
          "vst1.32    {d26[0]}, [%[output_ptr1]]! \n"
          "vst1.32    {d30[0]}, [%[output_ptr2]]! \n"
          "b          end_%=                      \n"

          "store_3h2w_%=:                         \n"
          "cmp        %[remain], #2               \n"
          "blt        store_3h1w_%=               \n"
          "vst1.32    {d20}, [%[output_ptr0]]!    \n"
          "vst1.32    {d24}, [%[output_ptr1]]!    \n"
          "vst1.32    {d28}, [%[output_ptr2]]!    \n"
          "cmp        %[remain], #3               \n"
          "blt        end_%=                      \n"
          "vst1.32    {d21[0]}, [%[output_ptr0]]! \n"
          "vst1.32    {d25[0]}, [%[output_ptr1]]! \n"
          "vst1.32    {d29[0]}, [%[output_ptr2]]! \n"
          "b          end_%=                      \n"

          "store_3h1w_%=:                         \n"
          "cmp        %[remain], #1               \n"
          "blt        end_%=                      \n"
          "vst1.32    {d20[0]}, [%[output_ptr0]]! \n"
          "vst1.32    {d24[0]}, [%[output_ptr1]]! \n"
          "vst1.32    {d28[0]}, [%[output_ptr2]]! \n"
          "end_%=:                                \n"
          : [output_ptr0] "+r"(output_ptr0), [output_ptr1] "+r"(output_ptr1),
            [output_ptr2] "+r"(output_ptr2), [input_ptr6] "+r"(input_ptr6),
            [input_ptr0] "+r"(input_ptr0), [input_ptr1] "+r"(input_ptr1),
            [input_ptr2] "+r"(input_ptr2), [input_ptr3] "+r"(input_ptr3),
            [input_ptr4] "+r"(input_ptr4), [input_ptr5] "+r"(input_ptr5),
            [loop] "+r"(loop)
          : [remain] "r"(remain)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "r0");
    }

    int start_h = (output_h / 3) * 6;
    for (int h = start_h; h < input_h - 2 /*(input_h - 1) - 1*/; h += 2) {
      const int8_t* input_ptr0 = input_ptr + h * input_w;
      const int8_t* input_ptr1 = input_ptr0 + input_w;
      const int8_t* input_ptr2 = input_ptr1 + input_w;
      int32_t* output_ptr0 = output_ptr + (h >> 1) * output_w;
      int loop = loops;
      asm volatile(
          "vld1.32    {q0}, [%[filter_ptr]] \n"
          "vmovl.s8   q14, d0               \n"
          "vmovl.s8   q15, d1               \n"
          "vdup.s16   d0, d28[0]            \n"
          "vdup.s16   d1, d28[1]            \n"
          "vdup.s16   d2, d28[2]            \n"
          "vdup.s16   d3, d28[3]            \n"
          "vdup.s16   d4, d29[0]            \n"
          "vdup.s16   d5, d29[1]            \n"
          "vdup.s16   d6, d29[2]            \n"
          "vdup.s16   d7, d29[3]            \n"
          "vdup.s16   d8, d30[0]            \n"
          :
          : [filter_ptr] "r"(filter_ptr)
          : "memory", "q0", "q1", "q2", "q3", "q4", "q14", "q15");
      asm volatile(
          "mov        r0, #12               \n"
          "cmp        %[loop], #0           \n"
          "ble        start_remain_%=       \n"
          // loop 6 widths
          "loop_1h6w_%=:                               \n"
          "vld2.8     {d10, d11}, [%[input_ptr0]], r0  \n"
          "vld2.8     {d12, d13}, [%[input_ptr1]], r0  \n"
          "vld2.8     {d14, d15}, [%[input_ptr2]], r0  \n"
          "vext.s8    d9, d10, d10, #1      \n"
          "vmovl.s8   q10, d9               \n"
          "vmovl.s8   q8, d10               \n"
          "vmovl.s8   q9, d11               \n"
          "vmull.s16  q11, d16, d0          \n"
          "vmlal.s16  q11, d18, d1          \n"
          "vmlal.s16  q11, d20, d2          \n"
          "vmull.s16  q12, d17, d0          \n"
          "vmlal.s16  q12, d19, d1          \n"
          "vmlal.s16  q12, d21, d2          \n"

          "vext.s8    d9, d12, d12, #1      \n"
          "vmovl.s8   q10, d9               \n"
          "vmovl.s8   q8, d12               \n"
          "vmovl.s8   q9, d13               \n"
          "vmlal.s16  q11, d16, d3          \n"
          "vmlal.s16  q11, d18, d4          \n"
          "vmlal.s16  q11, d20, d5          \n"
          "vmlal.s16  q12, d17, d3          \n"
          "vmlal.s16  q12, d19, d4          \n"
          "vmlal.s16  q12, d21, d5          \n"

          "vext.s8    d9, d14, d14, #1      \n"
          "vmovl.s8   q10, d9               \n"
          "vmovl.s8   q8, d14               \n"
          "vmovl.s8   q9, d15               \n"
          "vmlal.s16  q11, d16, d6          \n"
          "vmlal.s16  q11, d18, d7          \n"
          "vmlal.s16  q11, d20, d8          \n"
          "vmlal.s16  q12, d17, d6          \n"
          "vmlal.s16  q12, d19, d7          \n"
          "vmlal.s16  q12, d21, d8          \n"
          // store row 0
          "vst1.32    {d22-d24}, [%[output_ptr0]]! \n"

          "subs       %[loop], #1           \n"
          "bne        loop_1h6w_%=          \n"

          "start_remain_%=:                 \n"
          "cmp        %[remain], #0         \n"
          "ble        end_%=                \n"
          "vld2.8     {d10, d11}, [%[input_ptr0]]  \n"
          "vld2.8     {d12, d13}, [%[input_ptr1]]  \n"
          "vld2.8     {d14, d15}, [%[input_ptr2]]  \n"
          "vext.s8    d9, d10, d10, #1      \n"
          "vmovl.s8   q10, d9               \n"
          "vmovl.s8   q8, d10               \n"
          "vmovl.s8   q9, d11               \n"
          "vmull.s16  q11, d16, d0          \n"
          "vmlal.s16  q11, d18, d1          \n"
          "vmlal.s16  q11, d20, d2          \n"
          "vmull.s16  q12, d17, d0          \n"
          "vmlal.s16  q12, d19, d1          \n"
          "vmlal.s16  q12, d21, d2          \n"

          "vext.s8    d9, d12, d12, #1      \n"
          "vmovl.s8   q10, d9               \n"
          "vmovl.s8   q8, d12               \n"
          "vmovl.s8   q9, d13               \n"
          "vmlal.s16  q11, d16, d3          \n"
          "vmlal.s16  q11, d18, d4          \n"
          "vmlal.s16  q11, d20, d5          \n"
          "vmlal.s16  q12, d17, d3          \n"
          "vmlal.s16  q12, d19, d4          \n"
          "vmlal.s16  q12, d21, d5          \n"

          "vext.s8    d9, d14, d14, #1      \n"
          "vmovl.s8   q10, d9               \n"
          "vmovl.s8   q8, d14               \n"
          "vmovl.s8   q9, d15               \n"
          "vmlal.s16  q11, d16, d6          \n"
          "vmlal.s16  q11, d18, d7          \n"
          "vmlal.s16  q11, d20, d8          \n"
          "vmlal.s16  q12, d17, d6          \n"
          "vmlal.s16  q12, d19, d7          \n"
          "vmlal.s16  q12, d21, d8          \n"

          "cmp        %[remain], #4               \n"
          "blt        store_1h2w_%=               \n"
          "vst1.32    {q11}, [%[output_ptr0]]!    \n"
          "cmp        %[remain], #5               \n"
          "blt        end_%=                      \n"
          "vst1.32    {d24[0]}, [%[output_ptr0]]! \n"
          "b          end_%=                      \n"

          "store_1h2w_%=:                         \n"
          "cmp        %[remain], #2               \n"
          "blt        store_1h1w_%=               \n"
          "vst1.32    {d22}, [%[output_ptr0]]!    \n"
          "cmp        %[remain], #3               \n"
          "blt        end_%=                      \n"
          "vst1.32    {d23[0]}, [%[output_ptr0]]! \n"
          "b          end_%=                      \n"

          "store_1h1w_%=:                         \n"
          "cmp        %[remain], #1               \n"
          "blt        end_%=                      \n"
          "vst1.32    {d22[0]}, [%[output_ptr0]]! \n"
          "end_%=:                                \n"
          : [output_ptr0] "+r"(output_ptr0), [input_ptr0] "+r"(input_ptr0),
            [input_ptr1] "+r"(input_ptr1), [input_ptr2] "+r"(input_ptr2),
            [loop] "+r"(loop)
          : [remain] "r"(remain)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
            "q8", "q9", "q10", "q11", "q12", "r0");
    }
  }
#endif  // __aarch64__
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
