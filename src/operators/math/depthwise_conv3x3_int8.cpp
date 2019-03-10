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

#if defined(__ARM_NEON__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include "operators/math/depthwise_conv3x3.h"

namespace paddle_mobile {
namespace operators {
namespace math {

#define DEPTHWISE_CONV_NORMAL_BORDER(start, end)                         \
  for (int w = start; w < end; ++w) {                                    \
    const int w_in_start = -padding_w + w * Stride_w;                    \
    const int w_in_end = w_in_start + 3;                                 \
    const int w_start = w_in_start > 0 ? w_in_start : 0;                 \
    const int w_end = w_in_end < input_w ? w_in_end : input_w;           \
    int32_t value = 0;                                                   \
    for (int h_in = h_start; h_in < h_end; ++h_in) {                     \
      for (int w_in = w_start; w_in < w_end; ++w_in) {                   \
        value += filter[(h_in - h_in_start) * 3 + (w_in - w_in_start)] * \
                 input[h_in * input_w + w_in];                           \
      }                                                                  \
    }                                                                    \
    output_ptr[w] = value;                                               \
  }

template <int Stride = 1>
inline void Depth3x3NormalRowLoadInput(const int8_t *input, int16x8_t *y) {
  y[0] = vmovl_s8(vld1_s8(input));
  y[1] = vextq_s16(y[0], y[0], 1);
  y[2] = vextq_s16(y[1], y[1], 1);
}

template <>
inline void Depth3x3NormalRowLoadInput<2>(const int8_t *input, int16x8_t *y) {
  int8x8x2_t x0 = vld2_s8(input);
  y[0] = vmovl_s8(x0.val[0]);
  y[1] = vmovl_s8(x0.val[1]);
  y[2] = vextq_s16(y[0], y[0], 1);
}

template <int Stride_h, int Stride_w>
inline void DepthwiseConv3x3NormalRow(const int8_t *input, const int8_t *filter,
                                      const int h_output, const int input_h,
                                      const int input_w, const int padding_h,
                                      const int padding_w, const int output_w,
                                      int32_t *output, int16x4_t *ker) {
  const int h_in_start = -padding_h + h_output * Stride_h;
  const int h_in_end = h_in_start + 3;
  const int h_start = h_in_start > 0 ? h_in_start : 0;
  const int h_end = h_in_end < input_h ? h_in_end : input_h;

  const int valid_w_start = (padding_w + Stride_w - 1) / Stride_w;
  const int valid_w_end = (input_w + padding_w - 3) / Stride_w + 1;
  int32_t *output_ptr = output + h_output * output_w;
  // border left
  DEPTHWISE_CONV_NORMAL_BORDER(0, valid_w_start)
  // middle
  int remain_start = valid_w_start;
  int output_tiles = (valid_w_end - valid_w_start) / 6;
  remain_start = valid_w_start + output_tiles * 6;
  int32x4_t _sum0, _sum1;
  int16x8_t _y[3];
  for (int w = 0; w < output_tiles * 6; w += 6) {
    _sum0 = veorq_s32(_sum0, _sum0);
    _sum1 = veorq_s32(_sum1, _sum1);
    int output_offset = valid_w_start + w;
    int input_w_offset = output_offset * Stride_w - padding_w;
    for (int h_in = h_start; h_in < h_end; ++h_in) {
      int index = h_in - h_in_start;
      Depth3x3NormalRowLoadInput<Stride_w>(
          input + h_in * input_w + input_w_offset, _y);
      _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_y[0]), ker[index], 0);
      _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_y[1]), ker[index], 1);
      _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_y[2]), ker[index], 2);
      _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_y[0]), ker[index], 0);
      _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_y[1]), ker[index], 1);
      _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_y[2]), ker[index], 2);
    }
    vst1q_s32(output_ptr + output_offset, _sum0);
    vst1_s32(output_ptr + output_offset + 4, vget_low_s32(_sum1));
  }
  for (int w = remain_start; w < valid_w_end; ++w) {
    int32_t value = 0;
    int input_start = -padding_w + w * Stride_w;
    for (int h_in = h_start; h_in < h_end; ++h_in) {
      for (int j = 0; j < 3; ++j) {
        value += filter[(h_in - h_in_start) * 3 + j] *
                 input[h_in * input_w + j + input_start];
      }
    }
    output_ptr[w] = value;
  }
  // border right
  DEPTHWISE_CONV_NORMAL_BORDER(valid_w_end, output_w)
}

template <>
void DepthwiseConv3x3S1<int8_t, int32_t>(const framework::Tensor &input,
                                         const framework::Tensor &filter,
                                         const std::vector<int> &paddings,
                                         framework::Tensor *output) {
  const int8_t *input_data = input.data<int8_t>();
  const int8_t *filter_data = filter.data<int8_t>();
  int32_t *out_data = output->mutable_data<int32_t>();
  int input_h = input.dims()[2];
  int input_w = input.dims()[3];
  int output_h = output->dims()[2];
  int output_w = output->dims()[3];
  int padding_h = paddings[0];
  int padding_w = paddings[1];
  int image_size = input_h * input_w;
  int out_image_size = output_h * output_w;
  int valid_h_start = padding_h;
  int valid_h_end = output_h - valid_h_start;
  int valid_h = valid_h_end - valid_h_start;
  int valid_w_start = padding_w;
  int valid_w_end = output_w - valid_w_start;
  int valid_w = valid_w_end - valid_w_start;

  #pragma omp parallel for
  for (int g = 0; g < input.dims()[1]; ++g) {
    const int8_t *input_ptr = input_data + g * image_size;
    const int8_t *filter_ptr = filter_data + g * 9;
    int32_t *output_ptr = out_data + g * out_image_size;

    const int8_t *filter_ptr0 = filter_ptr;
    const int8_t *filter_ptr1 = filter_ptr0 + 3;
    const int8_t *filter_ptr2 = filter_ptr1 + 3;
    int16x4_t _k0 = vget_low_s16(vmovl_s8(vld1_s8(filter_ptr0)));
    int16x4_t _k1 = vget_low_s16(vmovl_s8(vld1_s8(filter_ptr1)));
    int16x4_t _k2 = vget_low_s16(vmovl_s8(vld1_s8(filter_ptr2)));
    int16x8_t _ker0 = vcombine_s16(_k0, _k1);
    int16x8_t _ker1 = vcombine_s16(_k2, _k2);
    int16x4_t zero = vdup_n_s16(0);
    int16x4_t _ker[3] = {_k0, _k1, _k2};
    // top
    for (int h = 0; h < valid_h_start; ++h) {
      DepthwiseConv3x3NormalRow<1, 1>(input_ptr, filter_ptr, h, input_h,
                                      input_w, padding_h, padding_w, output_w,
                                      output_ptr, _ker);
    }
    // valid
    int output_w_tiles = valid_w / 6;
    int output_w_remain = valid_w - output_w_tiles * 6;
    for (int h = valid_h_start; h < valid_h_end - 3; h += 4) {
      const int8_t *input_ptr0 = input_ptr + (h - padding_h) * input_w;
      const int8_t *input_ptr1 = input_ptr0 + input_w;
      const int8_t *input_ptr2 = input_ptr1 + input_w;
      const int8_t *input_ptr3 = input_ptr2 + input_w;
      const int8_t *input_ptr4 = input_ptr3 + input_w;
      const int8_t *input_ptr5 = input_ptr4 + input_w;
      int32_t *output_ptr0 = output_ptr + h * output_w;
      int32_t *output_ptr1 = output_ptr0 + output_w;
      int32_t *output_ptr2 = output_ptr1 + output_w;
      int32_t *output_ptr3 = output_ptr2 + output_w;
      // pad left
      if (padding_w) {
        int16x4_t row0 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr0)));
        int16x4_t row1 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr1)));
        int16x4_t row2 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr2)));
        int16x4_t row3 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr3)));
        int16x4_t row4 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr4)));
        int16x4_t row5 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr5)));
        int32x4_t acc;
        for (int w = valid_w_start - 1; w >= 0; --w) {
          int padding = padding_w - w;
          if (padding >= 3) {
            output_ptr0[w] = 0;
            output_ptr1[w] = 0;
            output_ptr2[w] = 0;
            output_ptr3[w] = 0;
          } else {
            row0 = vext_s16(zero, row0, 3);
            row1 = vext_s16(zero, row1, 3);
            row2 = vext_s16(zero, row2, 3);
            row3 = vext_s16(zero, row3, 3);
            row4 = vext_s16(zero, row4, 3);
            row5 = vext_s16(zero, row5, 3);
            acc = vmull_s16(row0, _ker[0]);
            acc = vmlal_s16(acc, row1, _ker[1]);
            acc = vmlal_s16(acc, row2, _ker[2]);
            output_ptr0[w] = vgetq_lane_s32(acc, 1) + vgetq_lane_s32(acc, 2);
            acc = vmull_s16(row1, _ker[0]);
            acc = vmlal_s16(acc, row2, _ker[1]);
            acc = vmlal_s16(acc, row3, _ker[2]);
            output_ptr1[w] = vgetq_lane_s32(acc, 1) + vgetq_lane_s32(acc, 2);
            acc = vmull_s16(row2, _ker[0]);
            acc = vmlal_s16(acc, row3, _ker[1]);
            acc = vmlal_s16(acc, row4, _ker[2]);
            output_ptr2[w] = vgetq_lane_s32(acc, 1) + vgetq_lane_s32(acc, 2);
            acc = vmull_s16(row3, _ker[0]);
            acc = vmlal_s16(acc, row4, _ker[1]);
            acc = vmlal_s16(acc, row5, _ker[2]);
            output_ptr3[w] = vgetq_lane_s32(acc, 1) + vgetq_lane_s32(acc, 2);
          }
        }
        output_ptr0 += valid_w_start;
        output_ptr1 += valid_w_start;
        output_ptr2 += valid_w_start;
        output_ptr3 += valid_w_start;
      }
#if __aarch64__
#else
      // valid
      int loop = output_w_tiles;
      asm volatile(
          "cmp        %[loop], #0                  \n"
          "ble        start_remain_%=              \n"
          "mov        r0, #6                       \n"
          // loop 6 width
          "loop_4h6w_%=:                           \n"
          "vld1.32    {d9}, [%[input_ptr0]], r0    \n"
          "vld1.32    {d10}, [%[input_ptr1]], r0   \n"
          "vld1.32    {d11}, [%[input_ptr2]], r0   \n"
          "vext.s8    d12, d9, d9, #1              \n"
          "vext.s8    d13, d9, d9, #2              \n"
          "vmovl.s8   q7, d9                       \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmull.s16  q10, d14, %e[ker0][0]        \n"
          "vmlal.s16  q10, d16, %e[ker0][1]        \n"
          "vmlal.s16  q10, d18, %e[ker0][2]        \n"
          "vmull.s16  q11, d15, %e[ker0][0]        \n"
          "vmlal.s16  q11, d17, %e[ker0][1]        \n"
          "vmlal.s16  q11, d19, %e[ker0][2]        \n"

          "vext.s8    d12, d10, d10, #1            \n"
          "vext.s8    d13, d10, d10, #2            \n"
          "vmovl.s8   q7, d10                      \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmlal.s16  q10, d14, %f[ker0][0]        \n"
          "vmlal.s16  q10, d16, %f[ker0][1]        \n"
          "vmlal.s16  q10, d18, %f[ker0][2]        \n"
          "vmlal.s16  q11, d15, %f[ker0][0]        \n"
          "vmlal.s16  q11, d17, %f[ker0][1]        \n"
          "vmlal.s16  q11, d19, %f[ker0][2]        \n"

          "vmull.s16  q12, d14, %e[ker0][0]        \n"
          "vmlal.s16  q12, d16, %e[ker0][1]        \n"
          "vmlal.s16  q12, d18, %e[ker0][2]        \n"
          "vmull.s16  q13, d15, %e[ker0][0]        \n"
          "vmlal.s16  q13, d17, %e[ker0][1]        \n"
          "vmlal.s16  q13, d19, %e[ker0][2]        \n"

          "vext.s8    d12, d11, d11, #1            \n"
          "vext.s8    d13, d11, d11, #2            \n"
          "vmovl.s8   q7, d11                      \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmlal.s16  q10, d14, %e[ker1][0]        \n"
          "vmlal.s16  q10, d16, %e[ker1][1]        \n"
          "vmlal.s16  q10, d18, %e[ker1][2]        \n"
          "vmlal.s16  q11, d15, %e[ker1][0]        \n"
          "vmlal.s16  q11, d17, %e[ker1][1]        \n"
          "vmlal.s16  q11, d19, %e[ker1][2]        \n"
          // store row 0, reuse q10/q11
          "vst1.32    {d20-d22}, [%[output_ptr0]]! \n"

          "vmlal.s16  q12, d14, %f[ker0][0]        \n"
          "vmlal.s16  q12, d16, %f[ker0][1]        \n"
          "vmlal.s16  q12, d18, %f[ker0][2]        \n"
          "vmlal.s16  q13, d15, %f[ker0][0]        \n"
          "vmlal.s16  q13, d17, %f[ker0][1]        \n"
          "vmlal.s16  q13, d19, %f[ker0][2]        \n"

          "vmull.s16  q14, d14, %e[ker0][0]        \n"
          "vmlal.s16  q14, d16, %e[ker0][1]        \n"
          "vmlal.s16  q14, d18, %e[ker0][2]        \n"
          "vmull.s16  q15, d15, %e[ker0][0]        \n"
          "vmlal.s16  q15, d17, %e[ker0][1]        \n"
          "vmlal.s16  q15, d19, %e[ker0][2]        \n"

          "vld1.32    {d9}, [%[input_ptr3]], r0    \n"
          "vld1.32    {d10}, [%[input_ptr4]], r0   \n"
          "vld1.32    {d11}, [%[input_ptr5]], r0   \n"
          "vext.s8    d12, d9, d9, #1              \n"
          "vext.s8    d13, d9, d9, #2              \n"
          "vmovl.s8   q7, d9                       \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmlal.s16  q12, d14, %e[ker1][0]        \n"
          "vmlal.s16  q12, d16, %e[ker1][1]        \n"
          "vmlal.s16  q12, d18, %e[ker1][2]        \n"
          "vmlal.s16  q13, d15, %e[ker1][0]        \n"
          "vmlal.s16  q13, d17, %e[ker1][1]        \n"
          "vmlal.s16  q13, d19, %e[ker1][2]        \n"
          // store row 1
          "vst1.32    {d24-d26}, [%[output_ptr1]]! \n"

          "vmlal.s16  q14, d14, %f[ker0][0]        \n"
          "vmlal.s16  q14, d16, %f[ker0][1]        \n"
          "vmlal.s16  q14, d18, %f[ker0][2]        \n"
          "vmlal.s16  q15, d15, %f[ker0][0]        \n"
          "vmlal.s16  q15, d17, %f[ker0][1]        \n"
          "vmlal.s16  q15, d19, %f[ker0][2]        \n"

          "vmull.s16  q10, d14, %e[ker0][0]        \n"
          "vmlal.s16  q10, d16, %e[ker0][1]        \n"
          "vmlal.s16  q10, d18, %e[ker0][2]        \n"
          "vmull.s16  q11, d15, %e[ker0][0]        \n"
          "vmlal.s16  q11, d17, %e[ker0][1]        \n"
          "vmlal.s16  q11, d19, %e[ker0][2]        \n"

          "vext.s8    d12, d10, d10, #1            \n"
          "vext.s8    d13, d10, d10, #2            \n"
          "vmovl.s8   q7, d10                      \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmlal.s16  q14, d14, %e[ker1][0]        \n"
          "vmlal.s16  q14, d16, %e[ker1][1]        \n"
          "vmlal.s16  q14, d18, %e[ker1][2]        \n"
          "vmlal.s16  q15, d15, %e[ker1][0]        \n"
          "vmlal.s16  q15, d17, %e[ker1][1]        \n"
          "vmlal.s16  q15, d19, %e[ker1][2]        \n"
          // store row 2
          "vst1.32    {d28-d30}, [%[output_ptr2]]! \n"

          "vmlal.s16  q10, d14, %f[ker0][0]        \n"
          "vmlal.s16  q10, d16, %f[ker0][1]        \n"
          "vmlal.s16  q10, d18, %f[ker0][2]        \n"
          "vmlal.s16  q11, d15, %f[ker0][0]        \n"
          "vmlal.s16  q11, d17, %f[ker0][1]        \n"
          "vmlal.s16  q11, d19, %f[ker0][2]        \n"

          "vext.s8    d12, d11, d11, #1            \n"
          "vext.s8    d13, d11, d11, #2            \n"
          "vmovl.s8   q7, d11                      \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmlal.s16  q10, d14, %e[ker1][0]        \n"
          "vmlal.s16  q10, d16, %e[ker1][1]        \n"
          "vmlal.s16  q10, d18, %e[ker1][2]        \n"
          "vmlal.s16  q11, d15, %e[ker1][0]        \n"
          "vmlal.s16  q11, d17, %e[ker1][1]        \n"
          "vmlal.s16  q11, d19, %e[ker1][2]        \n"
          // store row 3
          "vst1.32    {d20-d22}, [%[output_ptr3]]! \n"

          "subs       %[loop], #1                  \n"
          "bne        loop_4h6w_%=                 \n"

          "start_remain_%=:                        \n"
          "cmp        %[remain], #0                \n"
          "ble        end_%=                       \n"

          "mov        r0, %[remain]                \n"
          "vld1.32    {d9}, [%[input_ptr0]], r0    \n"
          "vmovl.s8   q7, d9                       \n"
          "vext.s8    d9, d9, d9, #1               \n"
          "vmovl.s8   q8, d9                       \n"
          "vext.s8    d9, d9, d9, #1               \n"
          "vmovl.s8   q9, d9                       \n"
          "vmull.s16  q10, d14, %e[ker0][0]        \n"
          "vmlal.s16  q10, d16, %e[ker0][1]        \n"
          "vmlal.s16  q10, d18, %e[ker0][2]        \n"
          "vld1.32    {d9}, [%[input_ptr1]], r0    \n"
          "vmull.s16  q11, d15, %e[ker0][0]        \n"
          "vmlal.s16  q11, d17, %e[ker0][1]        \n"
          "vmlal.s16  q11, d19, %e[ker0][2]        \n"

          "vmovl.s8   q7, d9                       \n"
          "vext.s8    d9, d9, d9, #1               \n"
          "vmovl.s8   q8, d9                       \n"
          "vext.s8    d9, d9, d9, #1               \n"
          "vmovl.s8   q9, d9                       \n"
          "vmlal.s16  q10, d14, %f[ker0][0]        \n"
          "vmlal.s16  q10, d16, %f[ker0][1]        \n"
          "vmlal.s16  q10, d18, %f[ker0][2]        \n"
          "vmlal.s16  q11, d15, %f[ker0][0]        \n"
          "vmlal.s16  q11, d17, %f[ker0][1]        \n"
          "vmlal.s16  q11, d19, %f[ker0][2]        \n"

          "vmull.s16  q12, d14, %e[ker0][0]        \n"
          "vmlal.s16  q12, d16, %e[ker0][1]        \n"
          "vmlal.s16  q12, d18, %e[ker0][2]        \n"
          "vld1.32    {d9}, [%[input_ptr2]], r0    \n"
          "vmull.s16  q13, d15, %e[ker0][0]        \n"
          "vmlal.s16  q13, d17, %e[ker0][1]        \n"
          "vmlal.s16  q13, d19, %e[ker0][2]        \n"

          "vmovl.s8   q7, d9                       \n"
          "vext.s8    d9, d9, d9, #1               \n"
          "vmovl.s8   q8, d9                       \n"
          "vext.s8    d9, d9, d9, #1               \n"
          "vmovl.s8   q9, d9                       \n"
          "vmlal.s16  q10, d14, %e[ker1][0]        \n"
          "vmlal.s16  q10, d16, %e[ker1][1]        \n"
          "vmlal.s16  q10, d18, %e[ker1][2]        \n"
          "vmlal.s16  q11, d15, %e[ker1][0]        \n"
          "vmlal.s16  q11, d17, %e[ker1][1]        \n"
          "vmlal.s16  q11, d19, %e[ker1][2]        \n"

          "vmlal.s16  q12, d14, %f[ker0][0]        \n"
          "vmlal.s16  q12, d16, %f[ker0][1]        \n"
          "vmlal.s16  q12, d18, %f[ker0][2]        \n"
          "vmlal.s16  q13, d15, %f[ker0][0]        \n"
          "vmlal.s16  q13, d17, %f[ker0][1]        \n"
          "vmlal.s16  q13, d19, %f[ker0][2]        \n"

          "vmull.s16  q14, d14, %e[ker0][0]        \n"
          "vmlal.s16  q14, d16, %e[ker0][1]        \n"
          "vmlal.s16  q14, d18, %e[ker0][2]        \n"
          "vld1.32    {d9}, [%[input_ptr3]], r0    \n"
          "vmull.s16  q15, d15, %e[ker0][0]        \n"
          "vmlal.s16  q15, d17, %e[ker0][1]        \n"
          "vmlal.s16  q15, d19, %e[ker0][2]        \n"

          "vmovl.s8   q7, d9                       \n"
          "vext.s8    d9, d9, d9, #1               \n"
          "vmovl.s8   q8, d9                       \n"
          "vext.s8    d9, d9, d9, #1               \n"
          "vmovl.s8   q9, d9                       \n"
          "vmlal.s16  q12, d14, %e[ker1][0]        \n"
          "vmlal.s16  q12, d16, %e[ker1][1]        \n"
          "vmlal.s16  q12, d18, %e[ker1][2]        \n"
          "vmlal.s16  q13, d15, %e[ker1][0]        \n"
          "vmlal.s16  q13, d17, %e[ker1][1]        \n"
          "vmlal.s16  q13, d19, %e[ker1][2]        \n"

          "vmlal.s16  q14, d14, %f[ker0][0]        \n"
          "vmlal.s16  q14, d16, %f[ker0][1]        \n"
          "vmlal.s16  q14, d18, %f[ker0][2]        \n"
          "vmlal.s16  q15, d15, %f[ker0][0]        \n"
          "vmlal.s16  q15, d17, %f[ker0][1]        \n"
          "vmlal.s16  q15, d19, %f[ker0][2]        \n"

          "vmull.s16  q5, d14, %e[ker0][0]         \n"
          "vmlal.s16  q5, d16, %e[ker0][1]         \n"
          "vmlal.s16  q5, d18, %e[ker0][2]         \n"
          "vld1.32    {d9}, [%[input_ptr4]], r0    \n"
          "vmull.s16  q6, d15, %e[ker0][0]         \n"
          "vmlal.s16  q6, d17, %e[ker0][1]         \n"
          "vmlal.s16  q6, d19, %e[ker0][2]         \n"

          "vmovl.s8   q7, d9                       \n"
          "vext.s8    d9, d9, d9, #1               \n"
          "vmovl.s8   q8, d9                       \n"
          "vext.s8    d9, d9, d9, #1               \n"
          "vmovl.s8   q9, d9                       \n"
          "vmlal.s16  q14, d14, %e[ker1][0]        \n"
          "vmlal.s16  q14, d16, %e[ker1][1]        \n"
          "vmlal.s16  q14, d18, %e[ker1][2]        \n"
          "vmlal.s16  q15, d15, %e[ker1][0]        \n"
          "vmlal.s16  q15, d17, %e[ker1][1]        \n"
          "vmlal.s16  q15, d19, %e[ker1][2]        \n"

          "vmlal.s16  q5, d14, %f[ker0][0]         \n"
          "vmlal.s16  q5, d16, %f[ker0][1]         \n"
          "vmlal.s16  q5, d18, %f[ker0][2]         \n"
          "vld1.32    {d9}, [%[input_ptr5]], r0    \n"
          "vmlal.s16  q6, d15, %f[ker0][0]         \n"
          "vmlal.s16  q6, d17, %f[ker0][1]         \n"
          "vmlal.s16  q6, d19, %f[ker0][2]         \n"

          "vmovl.s8   q7, d9                       \n"
          "vext.s8    d9, d9, d9, #1               \n"
          "vmovl.s8   q8, d9                       \n"
          "vext.s8    d9, d9, d9, #1               \n"
          "vmovl.s8   q9, d9                       \n"
          "vmlal.s16  q5, d14, %e[ker1][0]         \n"
          "vmlal.s16  q5, d16, %e[ker1][1]         \n"
          "vmlal.s16  q5, d18, %e[ker1][2]         \n"
          "vmlal.s16  q6, d15, %e[ker1][0]         \n"
          "vmlal.s16  q6, d17, %e[ker1][1]         \n"
          "vmlal.s16  q6, d19, %e[ker1][2]         \n"

          "cmp        %[remain], #4                \n"
          "blt        store_4h2w_%=                \n"
          "vst1.32    {q10}, [%[output_ptr0]]!     \n"
          "vst1.32    {q12}, [%[output_ptr1]]!     \n"
          "vst1.32    {q14}, [%[output_ptr2]]!     \n"
          "vst1.32    {q5}, [%[output_ptr3]]!      \n"
          "cmp        %[remain], #5                \n"
          "blt        end_%=                       \n"
          "vst1.32    {d22[0]}, [%[output_ptr0]]!  \n"
          "vst1.32    {d26[0]}, [%[output_ptr1]]!  \n"
          "vst1.32    {d30[0]}, [%[output_ptr2]]!  \n"
          "vst1.32    {d12[0]}, [%[output_ptr3]]!  \n"
          "b          end_%=                       \n"

          "store_4h2w_%=:                          \n"
          "cmp        %[remain], #2                \n"
          "blt        store_4h1w_%=                \n"
          "vst1.32    {d20}, [%[output_ptr0]]!     \n"
          "vst1.32    {d24}, [%[output_ptr1]]!     \n"
          "vst1.32    {d28}, [%[output_ptr2]]!     \n"
          "vst1.32    {d10}, [%[output_ptr3]]!     \n"
          "cmp        %[remain], #3                \n"
          "blt        end_%=                       \n"
          "vst1.32    {d21[0]}, [%[output_ptr0]]!  \n"
          "vst1.32    {d25[0]}, [%[output_ptr1]]!  \n"
          "vst1.32    {d29[0]}, [%[output_ptr2]]!  \n"
          "vst1.32    {d11[0]}, [%[output_ptr3]]!  \n"
          "b          end_%=                       \n"

          "store_4h1w_%=:                          \n"
          "cmp        %[remain], #1                \n"
          "blt        end_%=                       \n"
          "vst1.32    {d20[0]}, [%[output_ptr0]]!  \n"
          "vst1.32    {d24[0]}, [%[output_ptr1]]!  \n"
          "vst1.32    {d28[0]}, [%[output_ptr2]]!  \n"
          "vst1.32    {d10[0]}, [%[output_ptr3]]!  \n"
          "end_%=:                                 \n"
          : [output_ptr0] "+r"(output_ptr0), [output_ptr1] "+r"(output_ptr1),
            [output_ptr2] "+r"(output_ptr2), [output_ptr3] "+r"(output_ptr3),
            [input_ptr0] "+r"(input_ptr0), [input_ptr1] "+r"(input_ptr1),
            [input_ptr2] "+r"(input_ptr2), [input_ptr3] "+r"(input_ptr3),
            [input_ptr4] "+r"(input_ptr4), [input_ptr5] "+r"(input_ptr5),
            [loop] "+r"(loop)
          : [remain] "r"(output_w_remain), [ker0] "w"(_ker0), [ker1] "w"(_ker1)
          : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
            "q12", "q13", "q14", "q15", "r0");
#endif  // __aarch64__
      // pad right
      if (padding_w) {
        int16x4_t row0 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr0 - 2)));
        int16x4_t row1 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr1 - 2)));
        int16x4_t row2 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr2 - 2)));
        int16x4_t row3 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr3 - 2)));
        int16x4_t row4 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr4 - 2)));
        int16x4_t row5 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr5 - 2)));
        row0 = vext_s16(row0, zero, 2);
        row1 = vext_s16(row1, zero, 2);
        row2 = vext_s16(row2, zero, 2);
        row3 = vext_s16(row3, zero, 2);
        row4 = vext_s16(row4, zero, 2);
        row5 = vext_s16(row5, zero, 2);
        int32x4_t acc;
        for (int w = valid_w_end; w < output_w; ++w) {
          int padding = w + 3 - (padding_w + input_w);
          if (padding >= 3) {
            *output_ptr0 = 0;
            *output_ptr1 = 0;
            *output_ptr2 = 0;
            *output_ptr3 = 0;
          } else {
            acc = vmull_s16(row0, _ker[0]);
            acc = vmlal_s16(acc, row1, _ker[1]);
            acc = vmlal_s16(acc, row2, _ker[2]);
            *output_ptr0 = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1);
            acc = vmull_s16(row1, _ker[0]);
            acc = vmlal_s16(acc, row2, _ker[1]);
            acc = vmlal_s16(acc, row3, _ker[2]);
            *output_ptr1 = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1);
            acc = vmull_s16(row2, _ker[0]);
            acc = vmlal_s16(acc, row3, _ker[1]);
            acc = vmlal_s16(acc, row4, _ker[2]);
            *output_ptr2 = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1);
            acc = vmull_s16(row3, _ker[0]);
            acc = vmlal_s16(acc, row4, _ker[1]);
            acc = vmlal_s16(acc, row5, _ker[2]);
            *output_ptr3 = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1);

            row0 = vext_s16(row0, zero, 1);
            row1 = vext_s16(row1, zero, 1);
            row2 = vext_s16(row2, zero, 1);
            row3 = vext_s16(row3, zero, 1);
            row4 = vext_s16(row4, zero, 1);
            row5 = vext_s16(row5, zero, 1);
          }
          output_ptr0++;
          output_ptr1++;
          output_ptr2++;
          output_ptr3++;
        }
      }
    }
    // remain height
    int start_h = valid_h_start + (valid_h & 0xFFFC);
    for (int h = start_h; h < valid_h_end - 1; h += 2) {
      const int8_t *input_ptr0 = input_ptr + (h - padding_h) * input_w;
      const int8_t *input_ptr1 = input_ptr0 + input_w;
      const int8_t *input_ptr2 = input_ptr1 + input_w;
      const int8_t *input_ptr3 = input_ptr2 + input_w;
      int32_t *output_ptr0 = output_ptr + h * output_w;
      int32_t *output_ptr1 = output_ptr0 + output_w;
      // pad left
      if (padding_w) {
        int16x4_t row0 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr0)));
        int16x4_t row1 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr1)));
        int16x4_t row2 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr2)));
        int16x4_t row3 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr3)));
        int32x4_t acc;
        for (int w = valid_w_start - 1; w >= 0; --w) {
          int padding = padding_w - w;
          if (padding >= 3) {
            output_ptr0[w] = 0;
            output_ptr1[w] = 0;
          } else {
            row0 = vext_s16(zero, row0, 3);
            row1 = vext_s16(zero, row1, 3);
            row2 = vext_s16(zero, row2, 3);
            row3 = vext_s16(zero, row3, 3);
            acc = vmull_s16(row0, _ker[0]);
            acc = vmlal_s16(acc, row1, _ker[1]);
            acc = vmlal_s16(acc, row2, _ker[2]);
            output_ptr0[w] = vgetq_lane_s32(acc, 1) + vgetq_lane_s32(acc, 2);
            acc = vmull_s16(row1, _ker[0]);
            acc = vmlal_s16(acc, row2, _ker[1]);
            acc = vmlal_s16(acc, row3, _ker[2]);
            output_ptr1[w] = vgetq_lane_s32(acc, 1) + vgetq_lane_s32(acc, 2);
          }
        }
        output_ptr0 += valid_w_start;
        output_ptr1 += valid_w_start;
      }
        // valid
#if __aarch64__
#else
      int loop = output_w_tiles;
      asm volatile(
          "cmp        %[loop], #0                  \n"
          "ble        start_remain_%=              \n"
          "mov        r0, #6                       \n"
          // loop 6 widths
          "loop_2h6w_%=:                           \n"
          "vld1.32    {d9}, [%[input_ptr0]], r0    \n"
          "vld1.32    {d10}, [%[input_ptr1]], r0   \n"
          "vld1.32    {d11}, [%[input_ptr2]], r0   \n"
          "vext.s8    d12, d9, d9, #1              \n"
          "vext.s8    d13, d9, d9, #2              \n"
          "vmovl.s8   q7, d9                       \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmull.s16  q10, d14, %e[ker0][0]        \n"
          "vmlal.s16  q10, d16, %e[ker0][1]        \n"
          "vmlal.s16  q10, d18, %e[ker0][2]        \n"
          "vmull.s16  q11, d15, %e[ker0][0]        \n"
          "vmlal.s16  q11, d17, %e[ker0][1]        \n"
          "vmlal.s16  q11, d19, %e[ker0][2]        \n"

          "vext.s8    d12, d10, d10, #1            \n"
          "vext.s8    d13, d10, d10, #2            \n"
          "vmovl.s8   q7, d10                      \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmlal.s16  q10, d14, %f[ker0][0]        \n"
          "vmlal.s16  q10, d16, %f[ker0][1]        \n"
          "vmlal.s16  q10, d18, %f[ker0][2]        \n"
          "vmlal.s16  q11, d15, %f[ker0][0]        \n"
          "vmlal.s16  q11, d17, %f[ker0][1]        \n"
          "vmlal.s16  q11, d19, %f[ker0][2]        \n"

          "vmull.s16  q12, d14, %e[ker0][0]        \n"
          "vmlal.s16  q12, d16, %e[ker0][1]        \n"
          "vmlal.s16  q12, d18, %e[ker0][2]        \n"
          "vmull.s16  q13, d15, %e[ker0][0]        \n"
          "vmlal.s16  q13, d17, %e[ker0][1]        \n"
          "vmlal.s16  q13, d19, %e[ker0][2]        \n"

          "vext.s8    d12, d11, d11, #1            \n"
          "vext.s8    d13, d11, d11, #2            \n"
          "vmovl.s8   q7, d11                      \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmlal.s16  q10, d14, %e[ker1][0]        \n"
          "vmlal.s16  q10, d16, %e[ker1][1]        \n"
          "vmlal.s16  q10, d18, %e[ker1][2]        \n"
          "vmlal.s16  q11, d15, %e[ker1][0]        \n"
          "vmlal.s16  q11, d17, %e[ker1][1]        \n"
          "vmlal.s16  q11, d19, %e[ker1][2]        \n"
          // store row 0, reuse q10/q11
          "vst1.32    {d20-d22}, [%[output_ptr0]]! \n"

          "vmlal.s16  q12, d14, %f[ker0][0]        \n"
          "vmlal.s16  q12, d16, %f[ker0][1]        \n"
          "vmlal.s16  q12, d18, %f[ker0][2]        \n"
          "vmlal.s16  q13, d15, %f[ker0][0]        \n"
          "vmlal.s16  q13, d17, %f[ker0][1]        \n"
          "vmlal.s16  q13, d19, %f[ker0][2]        \n"

          "vld1.32    {d9}, [%[input_ptr3]], r0    \n"
          "vext.s8    d12, d9, d9, #1              \n"
          "vext.s8    d13, d9, d9, #2              \n"
          "vmovl.s8   q7, d9                       \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmlal.s16  q12, d14, %e[ker1][0]        \n"
          "vmlal.s16  q12, d16, %e[ker1][1]        \n"
          "vmlal.s16  q12, d18, %e[ker1][2]        \n"
          "vmlal.s16  q13, d15, %e[ker1][0]        \n"
          "vmlal.s16  q13, d17, %e[ker1][1]        \n"
          "vmlal.s16  q13, d19, %e[ker1][2]        \n"
          // store row 1
          "vst1.32    {d24-d26}, [%[output_ptr1]]! \n"

          "subs       %[loop], #1                  \n"
          "bne        loop_2h6w_%=                 \n"

          "start_remain_%=:                        \n"
          "cmp        %[remain], #0                \n"
          "ble        end_%=                       \n"

          "mov        r0, %[remain]                \n"
          "vld1.32    {d9}, [%[input_ptr0]], r0    \n"
          "vld1.32    {d10}, [%[input_ptr1]], r0   \n"
          "vld1.32    {d11}, [%[input_ptr2]], r0   \n"
          "vext.s8    d12, d9, d9, #1              \n"
          "vext.s8    d13, d9, d9, #2              \n"
          "vmovl.s8   q7, d9                       \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmull.s16  q10, d14, %e[ker0][0]        \n"
          "vmlal.s16  q10, d16, %e[ker0][1]        \n"
          "vmlal.s16  q10, d18, %e[ker0][2]        \n"
          "vmull.s16  q11, d15, %e[ker0][0]        \n"
          "vmlal.s16  q11, d17, %e[ker0][1]        \n"
          "vmlal.s16  q11, d19, %e[ker0][2]        \n"

          "vext.s8    d12, d10, d10, #1            \n"
          "vext.s8    d13, d10, d10, #2            \n"
          "vmovl.s8   q7, d10                      \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmlal.s16  q10, d14, %f[ker0][0]        \n"
          "vmlal.s16  q10, d16, %f[ker0][1]        \n"
          "vmlal.s16  q10, d18, %f[ker0][2]        \n"
          "vmlal.s16  q11, d15, %f[ker0][0]        \n"
          "vmlal.s16  q11, d17, %f[ker0][1]        \n"
          "vmlal.s16  q11, d19, %f[ker0][2]        \n"

          "vmull.s16  q12, d14, %e[ker0][0]        \n"
          "vmlal.s16  q12, d16, %e[ker0][1]        \n"
          "vmlal.s16  q12, d18, %e[ker0][2]        \n"
          "vmull.s16  q13, d15, %e[ker0][0]        \n"
          "vmlal.s16  q13, d17, %e[ker0][1]        \n"
          "vmlal.s16  q13, d19, %e[ker0][2]        \n"

          "vext.s8    d12, d11, d11, #1            \n"
          "vext.s8    d13, d11, d11, #2            \n"
          "vmovl.s8   q7, d11                      \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmlal.s16  q10, d14, %e[ker1][0]        \n"
          "vmlal.s16  q10, d16, %e[ker1][1]        \n"
          "vmlal.s16  q10, d18, %e[ker1][2]        \n"
          "vmlal.s16  q11, d15, %e[ker1][0]        \n"
          "vmlal.s16  q11, d17, %e[ker1][1]        \n"
          "vmlal.s16  q11, d19, %e[ker1][2]        \n"

          "vmlal.s16  q12, d14, %f[ker0][0]        \n"
          "vmlal.s16  q12, d16, %f[ker0][1]        \n"
          "vmlal.s16  q12, d18, %f[ker0][2]        \n"
          "vmlal.s16  q13, d15, %f[ker0][0]        \n"
          "vmlal.s16  q13, d17, %f[ker0][1]        \n"
          "vmlal.s16  q13, d19, %f[ker0][2]        \n"

          "vld1.32    {d9}, [%[input_ptr3]], r0    \n"
          "vext.s8    d12, d9, d9, #1              \n"
          "vext.s8    d13, d9, d9, #2              \n"
          "vmovl.s8   q7, d9                       \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmlal.s16  q12, d14, %e[ker1][0]        \n"
          "vmlal.s16  q12, d16, %e[ker1][1]        \n"
          "vmlal.s16  q12, d18, %e[ker1][2]        \n"
          "vmlal.s16  q13, d15, %e[ker1][0]        \n"
          "vmlal.s16  q13, d17, %e[ker1][1]        \n"
          "vmlal.s16  q13, d19, %e[ker1][2]        \n"

          "cmp        %[remain], #4                \n"
          "blt        store_2h2w_%=                \n"
          "vst1.32    {q10}, [%[output_ptr0]]!     \n"
          "vst1.32    {q12}, [%[output_ptr1]]!     \n"
          "cmp        %[remain], #5                \n"
          "blt        end_%=                       \n"
          "vst1.32    {d22[0]}, [%[output_ptr0]]!  \n"
          "vst1.32    {d26[0]}, [%[output_ptr1]]!  \n"
          "b          end_%=                       \n"

          "store_2h2w_%=:                          \n"
          "cmp        %[remain], #2                \n"
          "blt        store_2h1w_%=                \n"
          "vst1.32    {d20}, [%[output_ptr0]]!     \n"
          "vst1.32    {d24}, [%[output_ptr1]]!     \n"
          "cmp        %[remain], #3                \n"
          "blt        end_%=                       \n"
          "vst1.32    {d21[0]}, [%[output_ptr0]]!  \n"
          "vst1.32    {d25[0]}, [%[output_ptr1]]!  \n"
          "b          end_%=                       \n"

          "store_2h1w_%=:                          \n"
          "cmp        %[remain], #1                \n"
          "blt        end_%=                       \n"
          "vst1.32    {d20[0]}, [%[output_ptr0]]!  \n"
          "vst1.32    {d24[0]}, [%[output_ptr1]]!  \n"
          "end_%=:                                 \n"
          : [output_ptr0] "+r"(output_ptr0), [output_ptr1] "+r"(output_ptr1),
            [input_ptr0] "+r"(input_ptr0), [input_ptr1] "+r"(input_ptr1),
            [input_ptr2] "+r"(input_ptr2), [input_ptr3] "+r"(input_ptr3),
            [loop] "+r"(loop)
          : [remain] "r"(output_w_remain), [ker0] "w"(_ker0), [ker1] "w"(_ker1)
          : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
            "q12", "q13", "q14", "q15", "r0");
#endif  // __aarch64__
      // pad right
      if (padding_w) {
        int16x4_t row0 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr0 - 2)));
        int16x4_t row1 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr1 - 2)));
        int16x4_t row2 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr2 - 2)));
        int16x4_t row3 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr3 - 2)));
        row0 = vext_s16(row0, zero, 2);
        row1 = vext_s16(row1, zero, 2);
        row2 = vext_s16(row2, zero, 2);
        row3 = vext_s16(row3, zero, 2);
        int32x4_t acc;
        for (int w = valid_w_end; w < output_w; ++w) {
          int padding = w + 3 - (padding_w + input_w);
          if (padding >= 3) {
            *output_ptr0 = 0;
            *output_ptr1 = 0;
          } else {
            acc = vmull_s16(row0, _ker[0]);
            acc = vmlal_s16(acc, row1, _ker[1]);
            acc = vmlal_s16(acc, row2, _ker[2]);
            *output_ptr0 = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1);
            acc = vmull_s16(row1, _ker[0]);
            acc = vmlal_s16(acc, row2, _ker[1]);
            acc = vmlal_s16(acc, row3, _ker[2]);
            *output_ptr1 = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1);

            row0 = vext_s16(row0, zero, 1);
            row1 = vext_s16(row1, zero, 1);
            row2 = vext_s16(row2, zero, 1);
            row3 = vext_s16(row3, zero, 1);
          }
          output_ptr0++;
          output_ptr1++;
        }
      }
    }

    start_h = valid_h_start + (valid_h & 0xFFFE);
    if (start_h < valid_h_end) {
      const int8_t *input_ptr0 = input_ptr + (start_h - padding_h) * input_w;
      const int8_t *input_ptr1 = input_ptr0 + input_w;
      const int8_t *input_ptr2 = input_ptr1 + input_w;
      int32_t *output_ptr0 = output_ptr + start_h * output_w;
      // pad left
      if (padding_w) {
        int16x4_t row0 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr0)));
        int16x4_t row1 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr1)));
        int16x4_t row2 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr2)));
        int32x4_t acc;
        for (int w = valid_w_start - 1; w >= 0; --w) {
          int padding = padding_w - w;
          if (padding >= 3) {
            output_ptr0[w] = 0;
          } else {
            row0 = vext_s16(zero, row0, 3);
            row1 = vext_s16(zero, row1, 3);
            row2 = vext_s16(zero, row2, 3);
            acc = vmull_s16(row0, _ker[0]);
            acc = vmlal_s16(acc, row1, _ker[1]);
            acc = vmlal_s16(acc, row2, _ker[2]);
            output_ptr0[w] = vgetq_lane_s32(acc, 1) + vgetq_lane_s32(acc, 2);
          }
        }
        output_ptr0 += valid_w_start;
      }
        // valid
#if __aarch64__
#else
      int loop = output_w_tiles;
      asm volatile(
          "cmp        %[loop], #0                  \n"
          "ble        start_remain_%=              \n"
          "mov        r0, #6                       \n"
          // loop 6 widths
          "loop_1h6w_%=:                           \n"
          "vld1.32    {d9}, [%[input_ptr0]], r0    \n"
          "vld1.32    {d10}, [%[input_ptr1]], r0   \n"
          "vld1.32    {d11}, [%[input_ptr2]], r0   \n"
          "vext.s8    d12, d9, d9, #1              \n"
          "vext.s8    d13, d9, d9, #2              \n"
          "vmovl.s8   q7, d9                       \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmull.s16  q10, d14, %e[ker0][0]        \n"
          "vmlal.s16  q10, d16, %e[ker0][1]        \n"
          "vmlal.s16  q10, d18, %e[ker0][2]        \n"
          "vmull.s16  q11, d15, %e[ker0][0]        \n"
          "vmlal.s16  q11, d17, %e[ker0][1]        \n"
          "vmlal.s16  q11, d19, %e[ker0][2]        \n"

          "vext.s8    d12, d10, d10, #1            \n"
          "vext.s8    d13, d10, d10, #2            \n"
          "vmovl.s8   q7, d10                      \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmlal.s16  q10, d14, %f[ker0][0]        \n"
          "vmlal.s16  q10, d16, %f[ker0][1]        \n"
          "vmlal.s16  q10, d18, %f[ker0][2]        \n"
          "vmlal.s16  q11, d15, %f[ker0][0]        \n"
          "vmlal.s16  q11, d17, %f[ker0][1]        \n"
          "vmlal.s16  q11, d19, %f[ker0][2]        \n"

          "vext.s8    d12, d11, d11, #1            \n"
          "vext.s8    d13, d11, d11, #2            \n"
          "vmovl.s8   q7, d11                      \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmlal.s16  q10, d14, %e[ker1][0]        \n"
          "vmlal.s16  q10, d16, %e[ker1][1]        \n"
          "vmlal.s16  q10, d18, %e[ker1][2]        \n"
          "vmlal.s16  q11, d15, %e[ker1][0]        \n"
          "vmlal.s16  q11, d17, %e[ker1][1]        \n"
          "vmlal.s16  q11, d19, %e[ker1][2]        \n"
          // store row 0, reuse q10/q11
          "vst1.32    {d20-d22}, [%[output_ptr0]]! \n"

          "subs       %[loop], #1                  \n"
          "bne        loop_1h6w_%=                 \n"

          "start_remain_%=:                        \n"
          "cmp        %[remain], #0                \n"
          "ble        end_%=                       \n"
          "mov        r0, %[remain]                \n"

          "vld1.32    {d9}, [%[input_ptr0]], r0    \n"
          "vld1.32    {d10}, [%[input_ptr1]], r0   \n"
          "vld1.32    {d11}, [%[input_ptr2]], r0   \n"
          "vext.s8    d12, d9, d9, #1              \n"
          "vext.s8    d13, d9, d9, #2              \n"
          "vmovl.s8   q7, d9                       \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmull.s16  q10, d14, %e[ker0][0]        \n"
          "vmlal.s16  q10, d16, %e[ker0][1]        \n"
          "vmlal.s16  q10, d18, %e[ker0][2]        \n"
          "vmull.s16  q11, d15, %e[ker0][0]        \n"
          "vmlal.s16  q11, d17, %e[ker0][1]        \n"
          "vmlal.s16  q11, d19, %e[ker0][2]        \n"

          "vext.s8    d12, d10, d10, #1            \n"
          "vext.s8    d13, d10, d10, #2            \n"
          "vmovl.s8   q7, d10                      \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmlal.s16  q10, d14, %f[ker0][0]        \n"
          "vmlal.s16  q10, d16, %f[ker0][1]        \n"
          "vmlal.s16  q10, d18, %f[ker0][2]        \n"
          "vmlal.s16  q11, d15, %f[ker0][0]        \n"
          "vmlal.s16  q11, d17, %f[ker0][1]        \n"
          "vmlal.s16  q11, d19, %f[ker0][2]        \n"

          "vext.s8    d12, d11, d11, #1            \n"
          "vext.s8    d13, d11, d11, #2            \n"
          "vmovl.s8   q7, d11                      \n"
          "vmovl.s8   q8, d12                      \n"
          "vmovl.s8   q9, d13                      \n"
          "vmlal.s16  q10, d14, %e[ker1][0]        \n"
          "vmlal.s16  q10, d16, %e[ker1][1]        \n"
          "vmlal.s16  q10, d18, %e[ker1][2]        \n"
          "vmlal.s16  q11, d15, %e[ker1][0]        \n"
          "vmlal.s16  q11, d17, %e[ker1][1]        \n"
          "vmlal.s16  q11, d19, %e[ker1][2]        \n"

          "cmp        %[remain], #4                \n"
          "blt        store_1h2w_%=                \n"
          "vst1.32    {q10}, [%[output_ptr0]]!     \n"
          "cmp        %[remain], #5                \n"
          "blt        end_%=                       \n"
          "vst1.32    {d22[0]}, [%[output_ptr0]]!  \n"
          "b          end_%=                       \n"

          "store_1h2w_%=:                          \n"
          "cmp        %[remain], #2                \n"
          "blt        store_1h1w_%=                \n"
          "vst1.32    {d20}, [%[output_ptr0]]!     \n"
          "cmp        %[remain], #3                \n"
          "blt        end_%=                       \n"
          "vst1.32    {d21[0]}, [%[output_ptr0]]!  \n"
          "b          end_%=                       \n"

          "store_1h1w_%=:                          \n"
          "cmp        %[remain], #1                \n"
          "blt        end_%=                       \n"
          "vst1.32    {d20[0]}, [%[output_ptr0]]!  \n"
          "end_%=:                                 \n"
          : [output_ptr0] "+r"(output_ptr0), [input_ptr0] "+r"(input_ptr0),
            [input_ptr1] "+r"(input_ptr1), [input_ptr2] "+r"(input_ptr2),
            [loop] "+r"(loop)
          : [remain] "r"(output_w_remain), [ker0] "w"(_ker0), [ker1] "w"(_ker1)
          : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
            "q12", "q13", "q14", "q15", "r0");
#endif  // __aarch64__
      // pad right
      if (padding_w) {
        int16x4_t row0 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr0 - 2)));
        int16x4_t row1 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr1 - 2)));
        int16x4_t row2 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr2 - 2)));
        row0 = vext_s16(row0, zero, 2);
        row1 = vext_s16(row1, zero, 2);
        row2 = vext_s16(row2, zero, 2);
        int32x4_t acc;
        for (int w = valid_w_end; w < output_w; ++w) {
          int padding = w + 3 - (padding_w + input_w);
          if (padding >= 3) {
            *output_ptr0 = 0;
          } else {
            acc = vmull_s16(row0, _ker[0]);
            acc = vmlal_s16(acc, row1, _ker[1]);
            acc = vmlal_s16(acc, row2, _ker[2]);
            *output_ptr0 = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1);

            row0 = vext_s16(row0, zero, 1);
            row1 = vext_s16(row1, zero, 1);
            row2 = vext_s16(row2, zero, 1);
          }
          output_ptr0++;
        }
      }
    }
    // bottom
    for (int h = valid_h_end; h < output_h; ++h) {
      DepthwiseConv3x3NormalRow<1, 1>(input_ptr, filter_ptr, h, input_h,
                                      input_w, padding_h, padding_w, output_w,
                                      output_ptr, _ker);
    }
  }
}

template <>
void DepthwiseConv3x3S2<int8_t, int32_t>(const framework::Tensor &input,
                                         const framework::Tensor &filter,
                                         const std::vector<int> &paddings,
                                         framework::Tensor *output) {
  const int8_t *input_data = input.data<int8_t>();
  const int8_t *filter_data = filter.data<int8_t>();
  int32_t *out_data = output->mutable_data<int32_t>();
  int input_h = input.dims()[2];
  int input_w = input.dims()[3];
  int output_h = output->dims()[2];
  int output_w = output->dims()[3];
  int padding_h = paddings[0];
  int padding_w = paddings[1];
  int image_size = input_h * input_w;
  int out_image_size = output_h * output_w;
  int valid_h_start = (padding_h + 1) / 2;
  int valid_h_end = (input_h + padding_h - 1) / 2;
  int valid_h = valid_h_end - valid_h_start;
  int valid_w_start = (padding_w + 1) / 2;
  int valid_w_end = (input_w + padding_w - 1) / 2;
  int valid_w = valid_w_end - valid_w_start;
  // for pad left
  int valid_input_w_start = (valid_w_start << 1) - padding_w;

  //  DLOG << "valid_h_start: " << valid_h_start;
  //  DLOG << "valid_h_end: " << valid_h_end;
  //  DLOG << "valid_w_start: " << valid_w_start;
  //  DLOG << "valid_w_end: " << valid_w_end;

  #pragma omp parallel for
  for (int g = 0; g < input.dims()[1]; ++g) {
    const int8_t *input_ptr = input_data + g * image_size;
    const int8_t *filter_ptr = filter_data + g * 9;
    int32_t *output_ptr = out_data + g * out_image_size;

    const int8_t *filter_ptr0 = filter_ptr;
    const int8_t *filter_ptr1 = filter_ptr0 + 3;
    const int8_t *filter_ptr2 = filter_ptr1 + 3;
    int16x4_t _k0 = vget_low_s16(vmovl_s8(vld1_s8(filter_ptr0)));
    int16x4_t _k1 = vget_low_s16(vmovl_s8(vld1_s8(filter_ptr1)));
    int16x4_t _k2 = vget_low_s16(vmovl_s8(vld1_s8(filter_ptr2)));
    int16x8_t _ker0 = vcombine_s16(_k0, _k1);
    int16x8_t _ker1 = vcombine_s16(_k2, _k2);
    int16x4_t _ker[3] = {_k0, _k1, _k2};

    // top
    for (int h = 0; h < valid_h_start; ++h) {
      DepthwiseConv3x3NormalRow<2, 2>(input_ptr, filter_ptr, h, input_h,
                                      input_w, padding_h, padding_w, output_w,
                                      output_ptr, _ker);
    }
    // valid
    int input_w_start = 2 * valid_w_start - padding_w;
    int output_w_tiles = valid_w / 6;
    int output_w_remain = valid_w - output_w_tiles * 6;
    for (int h = valid_h_start; h < valid_h_end - 2; h += 3) {
      const int8_t *input_ptr0 = input_ptr + (2 * h - padding_h) * input_w;
      const int8_t *input_ptr1 = input_ptr0 + input_w;
      const int8_t *input_ptr2 = input_ptr1 + input_w;
      const int8_t *input_ptr3 = input_ptr2 + input_w;
      const int8_t *input_ptr4 = input_ptr3 + input_w;
      const int8_t *input_ptr5 = input_ptr4 + input_w;
      const int8_t *input_ptr6 = input_ptr5 + input_w;
      int32_t *output_ptr0 = output_ptr + h * output_w;
      int32_t *output_ptr1 = output_ptr0 + output_w;
      int32_t *output_ptr2 = output_ptr1 + output_w;
      // pad left
      if (padding_w) {
        for (int w = valid_w_start - 1; w >= 0; --w) {
          int padding = padding_w - (w << 1);
          if (padding >= 3) {
            output_ptr0[w] = 0;
            output_ptr1[w] = 0;
            output_ptr2[w] = 0;
          } else {
            int16x4_t row0 =
                vget_low_s16(vmovl_s8(vld1_s8(input_ptr0 - padding)));
            int16x4_t row1 =
                vget_low_s16(vmovl_s8(vld1_s8(input_ptr1 - padding)));
            int16x4_t row2 =
                vget_low_s16(vmovl_s8(vld1_s8(input_ptr2 - padding)));
            int16x4_t row3 =
                vget_low_s16(vmovl_s8(vld1_s8(input_ptr3 - padding)));
            int16x4_t row4 =
                vget_low_s16(vmovl_s8(vld1_s8(input_ptr4 - padding)));
            int16x4_t row5 =
                vget_low_s16(vmovl_s8(vld1_s8(input_ptr5 - padding)));
            int16x4_t row6 =
                vget_low_s16(vmovl_s8(vld1_s8(input_ptr6 - padding)));
            int32x4_t acc0 = vmull_s16(row0, _ker[0]);
            acc0 = vmlal_s16(acc0, row1, _ker[1]);
            acc0 = vmlal_s16(acc0, row2, _ker[2]);
            int32x4_t acc1 = vmull_s16(row2, _ker[0]);
            acc1 = vmlal_s16(acc1, row3, _ker[1]);
            acc1 = vmlal_s16(acc1, row4, _ker[2]);
            int32x4_t acc2 = vmull_s16(row4, _ker[0]);
            acc2 = vmlal_s16(acc2, row5, _ker[1]);
            acc2 = vmlal_s16(acc2, row6, _ker[2]);
            int32_t sum0 = vgetq_lane_s32(acc0, 2);
            int32_t sum1 = vgetq_lane_s32(acc1, 2);
            int32_t sum2 = vgetq_lane_s32(acc2, 2);
            if (padding == 1) {
              sum0 += vgetq_lane_s32(acc0, 1);
              sum1 += vgetq_lane_s32(acc1, 1);
              sum2 += vgetq_lane_s32(acc2, 1);
            }
            output_ptr0[w] = sum0;
            output_ptr1[w] = sum1;
            output_ptr2[w] = sum2;
          }
        }
        input_ptr0 += valid_input_w_start;
        input_ptr1 += valid_input_w_start;
        input_ptr2 += valid_input_w_start;
        input_ptr3 += valid_input_w_start;
        input_ptr4 += valid_input_w_start;
        input_ptr5 += valid_input_w_start;
        input_ptr6 += valid_input_w_start;
        output_ptr0 += valid_w_start;
        output_ptr1 += valid_w_start;
        output_ptr2 += valid_w_start;
      }
        // valid
#if __aarch64__
#else
      int loop = output_w_tiles;
      asm volatile(
          "cmp        %[loop], #0                     \n"
          "ble        start_remain_%=                 \n"
          "mov        r0, #12                         \n"
          // loop 6 widths
          "loop_3h6w_%=:                              \n"
          "vld2.8     {d10-d11}, [%[input_ptr0]], r0  \n"
          "vld2.8     {d12-d13}, [%[input_ptr1]], r0  \n"
          "vld2.8     {d14-d15}, [%[input_ptr2]], r0  \n"
          "vext.s8    d9, d10, d10, #1                \n"
          "vmovl.s8   q10, d9                         \n"
          "vmovl.s8   q8, d10                         \n"
          "vmovl.s8   q9, d11                         \n"
          "vmull.s16  q11, d16, %e[ker0][0]           \n"
          "vmlal.s16  q11, d18, %e[ker0][1]           \n"
          "vmlal.s16  q11, d20, %e[ker0][2]           \n"
          "vmull.s16  q12, d17, %e[ker0][0]           \n"
          "vmlal.s16  q12, d19, %e[ker0][1]           \n"
          "vmlal.s16  q12, d21, %e[ker0][2]           \n"

          "vext.s8    d9, d12, d12, #1                \n"
          "vmovl.s8   q10, d9                         \n"
          "vmovl.s8   q8, d12                         \n"
          "vmovl.s8   q9, d13                         \n"
          "vmlal.s16  q11, d16, %f[ker0][0]           \n"
          "vmlal.s16  q11, d18, %f[ker0][1]           \n"
          "vmlal.s16  q11, d20, %f[ker0][2]           \n"
          "vmlal.s16  q12, d17, %f[ker0][0]           \n"
          "vmlal.s16  q12, d19, %f[ker0][1]           \n"
          "vmlal.s16  q12, d21, %f[ker0][2]           \n"

          "vext.s8    d9, d14, d14, #1                \n"
          "vmovl.s8   q10, d9                         \n"
          "vmovl.s8   q8, d14                         \n"
          "vmovl.s8   q9, d15                         \n"
          "vmlal.s16  q11, d16, %e[ker1][0]           \n"
          "vmlal.s16  q11, d18, %e[ker1][1]           \n"
          "vmlal.s16  q11, d20, %e[ker1][2]           \n"
          "vmlal.s16  q12, d17, %e[ker1][0]           \n"
          "vmlal.s16  q12, d19, %e[ker1][1]           \n"
          "vmlal.s16  q12, d21, %e[ker1][2]           \n"
          // store row 0, reuse q11/q12
          "vst1.32    {d22-d24}, [%[output_ptr0]]!    \n"

          "vmull.s16  q13, d16, %e[ker0][0]           \n"
          "vmlal.s16  q13, d18, %e[ker0][1]           \n"
          "vmlal.s16  q13, d20, %e[ker0][2]           \n"
          "vmull.s16  q14, d17, %e[ker0][0]           \n"
          "vmlal.s16  q14, d19, %e[ker0][1]           \n"
          "vmlal.s16  q14, d21, %e[ker0][2]           \n"

          "vld2.8     {d10-d11}, [%[input_ptr3]], r0  \n"
          "vld2.8     {d12-d13}, [%[input_ptr4]], r0  \n"
          "vld2.8     {d14-d15}, [%[input_ptr5]], r0  \n"
          "vext.s8    d9, d10, d10, #1                \n"
          "vmovl.s8   q10, d9                         \n"
          "vmovl.s8   q8, d10                         \n"
          "vmovl.s8   q9, d11                         \n"
          "vmlal.s16  q13, d16, %f[ker0][0]           \n"
          "vmlal.s16  q13, d18, %f[ker0][1]           \n"
          "vmlal.s16  q13, d20, %f[ker0][2]           \n"
          "vmlal.s16  q14, d17, %f[ker0][0]           \n"
          "vmlal.s16  q14, d19, %f[ker0][1]           \n"
          "vmlal.s16  q14, d21, %f[ker0][2]           \n"

          "vext.s8    d9, d12, d12, #1                \n"
          "vmovl.s8   q10, d9                         \n"
          "vmovl.s8   q8, d12                         \n"
          "vmovl.s8   q9, d13                         \n"
          "vmlal.s16  q13, d16, %e[ker1][0]           \n"
          "vmlal.s16  q13, d18, %e[ker1][1]           \n"
          "vmlal.s16  q13, d20, %e[ker1][2]           \n"
          "vmlal.s16  q14, d17, %e[ker1][0]           \n"
          "vmlal.s16  q14, d19, %e[ker1][1]           \n"
          "vmlal.s16  q14, d21, %e[ker1][2]           \n"
          // store row 1
          "vst1.32    {d26-d28}, [%[output_ptr1]]!    \n"

          "vmull.s16  q11, d16, %e[ker0][0]           \n"
          "vmlal.s16  q11, d18, %e[ker0][1]           \n"
          "vmlal.s16  q11, d20, %e[ker0][2]           \n"
          "vmull.s16  q12, d17, %e[ker0][0]           \n"
          "vmlal.s16  q12, d19, %e[ker0][1]           \n"
          "vmlal.s16  q12, d21, %e[ker0][2]           \n"

          "vext.s8    d9, d14, d14, #1                \n"
          "vmovl.s8   q10, d9                         \n"
          "vmovl.s8   q8, d14                         \n"
          "vmovl.s8   q9, d15                         \n"
          "vmlal.s16  q11, d16, %f[ker0][0]           \n"
          "vmlal.s16  q11, d18, %f[ker0][1]           \n"
          "vmlal.s16  q11, d20, %f[ker0][2]           \n"
          "vmlal.s16  q12, d17, %f[ker0][0]           \n"
          "vmlal.s16  q12, d19, %f[ker0][1]           \n"
          "vmlal.s16  q12, d21, %f[ker0][2]           \n"

          "vld2.8     {d10-d11}, [%[input_ptr6]], r0  \n"
          "vext.s8    d9, d10, d10, #1                \n"
          "vmovl.s8   q10, d9                         \n"
          "vmovl.s8   q8, d10                         \n"
          "vmovl.s8   q9, d11                         \n"
          "vmlal.s16  q11, d16, %e[ker1][0]           \n"
          "vmlal.s16  q11, d18, %e[ker1][1]           \n"
          "vmlal.s16  q11, d20, %e[ker1][2]           \n"
          "vmlal.s16  q12, d17, %e[ker1][0]           \n"
          "vmlal.s16  q12, d19, %e[ker1][1]           \n"
          "vmlal.s16  q12, d21, %e[ker1][2]           \n"
          // store row 2
          "vst1.32    {d22-d24}, [%[output_ptr2]]!    \n"

          "subs       %[loop], #1                     \n"
          "bne        loop_3h6w_%=                    \n"

          "start_remain_%=:                           \n"
          "cmp        %[remain], #0                   \n"
          "ble        end_%=                          \n"
          "mov        r0, %[remain], lsl #1           \n"

          "vld2.8     {d10-d11}, [%[input_ptr0]], r0  \n"
          "vld2.8     {d12-d13}, [%[input_ptr1]], r0  \n"
          "vext.s8    d9, d10, d10, #1                \n"
          "vmovl.s8   q9, d9                          \n"
          "vmovl.s8   q7, d10                         \n"
          "vmovl.s8   q8, d11                         \n"
          "vmull.s16  q10, d14, %e[ker0][0]           \n"
          "vmlal.s16  q10, d16, %e[ker0][1]           \n"
          "vmlal.s16  q10, d18, %e[ker0][2]           \n"
          "vmull.s16  q11, d15, %e[ker0][0]           \n"
          "vmlal.s16  q11, d17, %e[ker0][1]           \n"
          "vmlal.s16  q11, d19, %e[ker0][2]           \n"

          "vext.s8    d9, d12, d12, #1                \n"
          "vmovl.s8   q9, d9                          \n"
          "vmovl.s8   q7, d12                         \n"
          "vmovl.s8   q8, d13                         \n"
          "vmlal.s16  q10, d14, %f[ker0][0]           \n"
          "vmlal.s16  q10, d16, %f[ker0][1]           \n"
          "vmlal.s16  q10, d18, %f[ker0][2]           \n"
          "vmlal.s16  q11, d15, %f[ker0][0]           \n"
          "vmlal.s16  q11, d17, %f[ker0][1]           \n"
          "vmlal.s16  q11, d19, %f[ker0][2]           \n"

          "vld2.8     {d10-d11}, [%[input_ptr2]], r0  \n"
          "vld2.8     {d12-d13}, [%[input_ptr3]], r0  \n"
          "vext.s8    d9, d10, d10, #1                \n"
          "vmovl.s8   q9, d9                          \n"
          "vmovl.s8   q7, d10                         \n"
          "vmovl.s8   q8, d11                         \n"
          "vmlal.s16  q10, d14, %e[ker1][0]           \n"
          "vmlal.s16  q10, d16, %e[ker1][1]           \n"
          "vmlal.s16  q10, d18, %e[ker1][2]           \n"
          "vmlal.s16  q11, d15, %e[ker1][0]           \n"
          "vmlal.s16  q11, d17, %e[ker1][1]           \n"
          "vmlal.s16  q11, d19, %e[ker1][2]           \n"

          "vmull.s16  q12, d14, %e[ker0][0]           \n"
          "vmlal.s16  q12, d16, %e[ker0][1]           \n"
          "vmlal.s16  q12, d18, %e[ker0][2]           \n"
          "vmull.s16  q13, d15, %e[ker0][0]           \n"
          "vmlal.s16  q13, d17, %e[ker0][1]           \n"
          "vmlal.s16  q13, d19, %e[ker0][2]           \n"

          "vext.s8    d9, d12, d12, #1                \n"
          "vmovl.s8   q9, d9                          \n"
          "vmovl.s8   q7, d12                         \n"
          "vmovl.s8   q8, d13                         \n"
          "vmlal.s16  q12, d14, %f[ker0][0]           \n"
          "vmlal.s16  q12, d16, %f[ker0][1]           \n"
          "vmlal.s16  q12, d18, %f[ker0][2]           \n"
          "vmlal.s16  q13, d15, %f[ker0][0]           \n"
          "vmlal.s16  q13, d17, %f[ker0][1]           \n"
          "vmlal.s16  q13, d19, %f[ker0][2]           \n"

          "vld2.8     {d10-d11}, [%[input_ptr4]], r0  \n"
          "vld2.8     {d12-d13}, [%[input_ptr5]], r0  \n"
          "vext.s8    d9, d10, d10, #1                \n"
          "vmovl.s8   q9, d9                          \n"
          "vmovl.s8   q7, d10                         \n"
          "vmovl.s8   q8, d11                         \n"
          "vmlal.s16  q12, d14, %e[ker1][0]           \n"
          "vmlal.s16  q12, d16, %e[ker1][1]           \n"
          "vmlal.s16  q12, d18, %e[ker1][2]           \n"
          "vmlal.s16  q13, d15, %e[ker1][0]           \n"
          "vmlal.s16  q13, d17, %e[ker1][1]           \n"
          "vmlal.s16  q13, d19, %e[ker1][2]           \n"

          "vmull.s16  q14, d14, %e[ker0][0]           \n"
          "vmlal.s16  q14, d16, %e[ker0][1]           \n"
          "vmlal.s16  q14, d18, %e[ker0][2]           \n"
          "vmull.s16  q15, d15, %e[ker0][0]           \n"
          "vmlal.s16  q15, d17, %e[ker0][1]           \n"
          "vmlal.s16  q15, d19, %e[ker0][2]           \n"

          "vext.s8    d9, d12, d12, #1                \n"
          "vmovl.s8   q9, d9                          \n"
          "vmovl.s8   q7, d12                         \n"
          "vmovl.s8   q8, d13                         \n"
          "vmlal.s16  q14, d14, %f[ker0][0]           \n"
          "vmlal.s16  q14, d16, %f[ker0][1]           \n"
          "vmlal.s16  q14, d18, %f[ker0][2]           \n"
          "vmlal.s16  q15, d15, %f[ker0][0]           \n"
          "vmlal.s16  q15, d17, %f[ker0][1]           \n"
          "vmlal.s16  q15, d19, %f[ker0][2]           \n"

          "vld2.8     {d10-d11}, [%[input_ptr6]], r0  \n"
          "vext.s8    d9, d10, d10, #1                \n"
          "vmovl.s8   q9, d9                          \n"
          "vmovl.s8   q7, d10                         \n"
          "vmovl.s8   q8, d11                         \n"
          "vmlal.s16  q14, d14, %e[ker1][0]           \n"
          "vmlal.s16  q14, d16, %e[ker1][1]           \n"
          "vmlal.s16  q14, d18, %e[ker1][2]           \n"
          "vmlal.s16  q15, d15, %e[ker1][0]           \n"
          "vmlal.s16  q15, d17, %e[ker1][1]           \n"
          "vmlal.s16  q15, d19, %e[ker1][2]           \n"

          "cmp        %[remain], #4                   \n"
          "blt        store_3h2w_%=                   \n"
          "vst1.32    {q10}, [%[output_ptr0]]!        \n"
          "vst1.32    {q12}, [%[output_ptr1]]!        \n"
          "vst1.32    {q14}, [%[output_ptr2]]!        \n"
          "cmp        %[remain], #5                   \n"
          "blt        end_%=                          \n"
          "vst1.32    {d22[0]}, [%[output_ptr0]]!     \n"
          "vst1.32    {d26[0]}, [%[output_ptr1]]!     \n"
          "vst1.32    {d30[0]}, [%[output_ptr2]]!     \n"
          "b          end_%=                          \n"

          "store_3h2w_%=:                             \n"
          "cmp        %[remain], #2                   \n"
          "blt        store_3h1w_%=                   \n"
          "vst1.32    {d20}, [%[output_ptr0]]!        \n"
          "vst1.32    {d24}, [%[output_ptr1]]!        \n"
          "vst1.32    {d28}, [%[output_ptr2]]!        \n"
          "cmp        %[remain], #3                   \n"
          "blt        end_%=                          \n"
          "vst1.32    {d21[0]}, [%[output_ptr0]]!     \n"
          "vst1.32    {d25[0]}, [%[output_ptr1]]!     \n"
          "vst1.32    {d29[0]}, [%[output_ptr2]]!     \n"
          "b          end_%=                          \n"

          "store_3h1w_%=:                             \n"
          "cmp        %[remain], #1                   \n"
          "blt        end_%=                          \n"
          "vst1.32    {d20[0]}, [%[output_ptr0]]!     \n"
          "vst1.32    {d24[0]}, [%[output_ptr1]]!     \n"
          "vst1.32    {d28[0]}, [%[output_ptr2]]!     \n"
          "end_%=:                                    \n"
          : [output_ptr0] "+r"(output_ptr0), [output_ptr1] "+r"(output_ptr1),
            [output_ptr2] "+r"(output_ptr2), [input_ptr6] "+r"(input_ptr6),
            [input_ptr0] "+r"(input_ptr0), [input_ptr1] "+r"(input_ptr1),
            [input_ptr2] "+r"(input_ptr2), [input_ptr3] "+r"(input_ptr3),
            [input_ptr4] "+r"(input_ptr4), [input_ptr5] "+r"(input_ptr5),
            [loop] "+r"(loop)
          : [remain] "r"(output_w_remain), [ker0] "w"(_ker0), [ker1] "w"(_ker1)
          : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
            "q12", "q13", "q14", "q15", "r0");
#endif  // __aarch64__
      // pad right
      if (padding_w > 0) {
        int16x4_t row0 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr0)));
        int16x4_t row1 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr1)));
        int16x4_t row2 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr2)));
        int16x4_t row3 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr3)));
        int16x4_t row4 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr4)));
        int16x4_t row5 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr5)));
        int16x4_t row6 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr6)));
        int32x4_t acc0, acc1, acc2;
        for (int w = valid_w_end; w < output_w; ++w) {
          int padding = 2 * w + 3 - (padding_w + input_w);
          if (padding >= 3) {
            *output_ptr0 = 0;
            *output_ptr1 = 0;
            *output_ptr2 = 0;
          } else {
            acc0 = vmull_s16(row0, _ker[0]);
            acc0 = vmlal_s16(acc0, row1, _ker[1]);
            acc0 = vmlal_s16(acc0, row2, _ker[2]);
            acc1 = vmull_s16(row2, _ker[0]);
            acc1 = vmlal_s16(acc1, row3, _ker[1]);
            acc1 = vmlal_s16(acc1, row4, _ker[2]);
            acc2 = vmull_s16(row4, _ker[0]);
            acc2 = vmlal_s16(acc2, row5, _ker[1]);
            acc2 = vmlal_s16(acc2, row6, _ker[2]);
            int32_t sum0 = vgetq_lane_s32(acc0, 0);
            int32_t sum1 = vgetq_lane_s32(acc1, 0);
            int32_t sum2 = vgetq_lane_s32(acc2, 0);
            if (padding == 1) {
              sum0 += vgetq_lane_s32(acc0, 1);
              sum1 += vgetq_lane_s32(acc1, 1);
              sum2 += vgetq_lane_s32(acc2, 1);
            }
            *output_ptr0 = sum0;
            *output_ptr1 = sum1;
            *output_ptr2 = sum2;
          }
          output_ptr0++;
          output_ptr1++;
          output_ptr2++;
        }
      }
    }
    // remain height
    int start_h = valid_h_start + valid_h / 3 * 3;
    for (int h = start_h; h < valid_h_end; ++h) {
      const int8_t *input_ptr0 = input_ptr + (2 * h - padding_h) * input_w;
      const int8_t *input_ptr1 = input_ptr0 + input_w;
      const int8_t *input_ptr2 = input_ptr1 + input_w;
      int32_t *output_ptr0 = output_ptr + h * output_w;
      // pad left
      if (padding_w) {
        for (int w = valid_w_start - 1; w >= 0; --w) {
          int padding = padding_w - (w << 1);
          if (padding >= 3) {
            output_ptr0[w] = 0;
          } else {
            int16x4_t row0 =
                vget_low_s16(vmovl_s8(vld1_s8(input_ptr0 - padding)));
            int16x4_t row1 =
                vget_low_s16(vmovl_s8(vld1_s8(input_ptr1 - padding)));
            int16x4_t row2 =
                vget_low_s16(vmovl_s8(vld1_s8(input_ptr2 - padding)));
            int32x4_t acc = vmull_s16(row0, _ker[0]);
            acc = vmlal_s16(acc, row1, _ker[1]);
            acc = vmlal_s16(acc, row2, _ker[2]);
            int32_t sum0 = vgetq_lane_s32(acc, 2);
            if (padding == 1) {
              sum0 += vgetq_lane_s32(acc, 1);
            }
            output_ptr0[w] = sum0;
          }
        }
        input_ptr0 += valid_input_w_start;
        input_ptr1 += valid_input_w_start;
        input_ptr2 += valid_input_w_start;
        output_ptr0 += valid_w_start;
      }
        // valid
#if __aarch64__
#else
      int loop = output_w_tiles;
      asm volatile(
          "cmp        %[loop], #0                      \n"
          "ble        start_remain_%=                  \n"
          "mov        r0, #12                          \n"
          // loop 6 widths
          "loop_1h6w_%=:                               \n"
          "vld2.8     {d10, d11}, [%[input_ptr0]], r0  \n"
          "vld2.8     {d12, d13}, [%[input_ptr1]], r0  \n"
          "vld2.8     {d14, d15}, [%[input_ptr2]], r0  \n"
          "vext.s8    d9, d10, d10, #1                 \n"
          "vmovl.s8   q10, d9                          \n"
          "vmovl.s8   q8, d10                          \n"
          "vmovl.s8   q9, d11                          \n"
          "vmull.s16  q11, d16, %e[ker0][0]            \n"
          "vmlal.s16  q11, d18, %e[ker0][1]            \n"
          "vmlal.s16  q11, d20, %e[ker0][2]            \n"
          "vmull.s16  q12, d17, %e[ker0][0]            \n"
          "vmlal.s16  q12, d19, %e[ker0][1]            \n"
          "vmlal.s16  q12, d21, %e[ker0][2]            \n"

          "vext.s8    d9, d12, d12, #1                 \n"
          "vmovl.s8   q10, d9                          \n"
          "vmovl.s8   q8, d12                          \n"
          "vmovl.s8   q9, d13                          \n"
          "vmlal.s16  q11, d16, %f[ker0][0]            \n"
          "vmlal.s16  q11, d18, %f[ker0][1]            \n"
          "vmlal.s16  q11, d20, %f[ker0][2]            \n"
          "vmlal.s16  q12, d17, %f[ker0][0]            \n"
          "vmlal.s16  q12, d19, %f[ker0][1]            \n"
          "vmlal.s16  q12, d21, %f[ker0][2]            \n"

          "vext.s8    d9, d14, d14, #1                 \n"
          "vmovl.s8   q10, d9                          \n"
          "vmovl.s8   q8, d14                          \n"
          "vmovl.s8   q9, d15                          \n"
          "vmlal.s16  q11, d16, %e[ker1][0]            \n"
          "vmlal.s16  q11, d18, %e[ker1][1]            \n"
          "vmlal.s16  q11, d20, %e[ker1][2]            \n"
          "vmlal.s16  q12, d17, %e[ker1][0]            \n"
          "vmlal.s16  q12, d19, %e[ker1][1]            \n"
          "vmlal.s16  q12, d21, %e[ker1][2]            \n"
          // store row 0
          "vst1.32    {d22-d24}, [%[output_ptr0]]!     \n"

          "subs       %[loop], #1                      \n"
          "bne        loop_1h6w_%=                     \n"

          "start_remain_%=:                            \n"
          "cmp        %[remain], #0                    \n"
          "ble        end_%=                           \n"
          "mov        r0, %[remain], lsl #1            \n"

          "vld2.8     {d10, d11}, [%[input_ptr0]], r0  \n"
          "vld2.8     {d12, d13}, [%[input_ptr1]], r0  \n"
          "vld2.8     {d14, d15}, [%[input_ptr2]], r0  \n"
          "vext.s8    d9, d10, d10, #1                 \n"
          "vmovl.s8   q10, d9                          \n"
          "vmovl.s8   q8, d10                          \n"
          "vmovl.s8   q9, d11                          \n"
          "vmull.s16  q11, d16, %e[ker0][0]            \n"
          "vmlal.s16  q11, d18, %e[ker0][1]            \n"
          "vmlal.s16  q11, d20, %e[ker0][2]            \n"
          "vmull.s16  q12, d17, %e[ker0][0]            \n"
          "vmlal.s16  q12, d19, %e[ker0][1]            \n"
          "vmlal.s16  q12, d21, %e[ker0][2]            \n"

          "vext.s8    d9, d12, d12, #1                 \n"
          "vmovl.s8   q10, d9                          \n"
          "vmovl.s8   q8, d12                          \n"
          "vmovl.s8   q9, d13                          \n"
          "vmlal.s16  q11, d16, %f[ker0][0]            \n"
          "vmlal.s16  q11, d18, %f[ker0][1]            \n"
          "vmlal.s16  q11, d20, %f[ker0][2]            \n"
          "vmlal.s16  q12, d17, %f[ker0][0]            \n"
          "vmlal.s16  q12, d19, %f[ker0][1]            \n"
          "vmlal.s16  q12, d21, %f[ker0][2]            \n"

          "vext.s8    d9, d14, d14, #1                 \n"
          "vmovl.s8   q10, d9                          \n"
          "vmovl.s8   q8, d14                          \n"
          "vmovl.s8   q9, d15                          \n"
          "vmlal.s16  q11, d16, %e[ker1][0]            \n"
          "vmlal.s16  q11, d18, %e[ker1][1]            \n"
          "vmlal.s16  q11, d20, %e[ker1][2]            \n"
          "vmlal.s16  q12, d17, %e[ker1][0]            \n"
          "vmlal.s16  q12, d19, %e[ker1][1]            \n"
          "vmlal.s16  q12, d21, %e[ker1][2]            \n"

          "cmp        %[remain], #4                    \n"
          "blt        store_1h2w_%=                    \n"
          "vst1.32    {q11}, [%[output_ptr0]]!         \n"
          "cmp        %[remain], #5                    \n"
          "blt        end_%=                           \n"
          "vst1.32    {d24[0]}, [%[output_ptr0]]!      \n"
          "b          end_%=                           \n"

          "store_1h2w_%=:                              \n"
          "cmp        %[remain], #2                    \n"
          "blt        store_1h1w_%=                    \n"
          "vst1.32    {d22}, [%[output_ptr0]]!         \n"
          "cmp        %[remain], #3                    \n"
          "blt        end_%=                           \n"
          "vst1.32    {d23[0]}, [%[output_ptr0]]!      \n"
          "b          end_%=                           \n"

          "store_1h1w_%=:                              \n"
          "cmp        %[remain], #1                    \n"
          "blt        end_%=                           \n"
          "vst1.32    {d22[0]}, [%[output_ptr0]]!      \n"
          "end_%=:                                     \n"
          : [output_ptr0] "+r"(output_ptr0), [input_ptr0] "+r"(input_ptr0),
            [input_ptr1] "+r"(input_ptr1), [input_ptr2] "+r"(input_ptr2),
            [loop] "+r"(loop)
          : [remain] "r"(output_w_remain), [ker0] "w"(_ker0), [ker1] "w"(_ker1)
          : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
            "q12", "q13", "q14", "q15", "r0");
#endif  // __aarch64__
      // pad right
      if (padding_w > 0) {
        int16x4_t row0 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr0)));
        int16x4_t row1 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr1)));
        int16x4_t row2 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr2)));
        int32x4_t acc;
        for (int w = valid_w_end; w < output_w; ++w) {
          int padding = 2 * w + 3 - (padding_w + input_w);
          if (padding >= 3) {
            *output_ptr0 = 0;
          } else {
            acc = vmull_s16(row0, _ker[0]);
            acc = vmlal_s16(acc, row1, _ker[1]);
            acc = vmlal_s16(acc, row2, _ker[2]);
            int32_t sum0 = vgetq_lane_s32(acc, 0);
            if (padding == 1) {
              sum0 += vgetq_lane_s32(acc, 1);
            }
            *output_ptr0 = sum0;
          }
          output_ptr0++;
        }
      }
    }
    // bottom
    for (int h = valid_h_end; h < output_h; ++h) {
      DepthwiseConv3x3NormalRow<2, 2>(input_ptr, filter_ptr, h, input_h,
                                      input_w, padding_h, padding_w, output_w,
                                      output_ptr, _ker);
    }
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif  // __ARM_NEON__
