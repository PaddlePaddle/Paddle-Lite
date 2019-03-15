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

#if defined(__ARM_NEON__) && !defined(__aarch64__)

#include <arm_neon.h>
#include "operators/math/depthwise_conv5x5.h"

namespace paddle_mobile {
namespace operators {
namespace math {

#ifndef __aarch64__
inline int32x4_t vpaddq_s32(int32x4_t r0, int32x4_t r1) {
  int32x2_t sum0 = vpadd_s32(vget_low_s32(r0), vget_high_s32(r0));
  int32x2_t sum1 = vpadd_s32(vget_low_s32(r1), vget_high_s32(r1));
  return vcombine_s32(sum0, sum1);
}
#endif

template <int Stride = 1>
inline void Depth5x5NormalRowLoadInput(const int8_t *input, int16x4_t *y) {
  int16x8_t x = vmovl_s8(vld1_s8(input));
  y[0] = vget_low_s16(x);
  y[4] = vget_high_s16(x);
  y[1] = vext_s16(y[0], y[4], 1);
  y[2] = vext_s16(y[0], y[4], 2);
  y[3] = vext_s16(y[0], y[4], 3);
}

template <>
inline void Depth5x5NormalRowLoadInput<2>(const int8_t *input, int16x4_t *y) {
  int8x8x2_t x = vld2_s8(input);
  y[0] = vget_low_s16(vmovl_s8(x.val[0]));
  y[1] = vget_low_s16(vmovl_s8(x.val[1]));
  y[2] = vext_s16(y[0], y[0], 1);
  y[3] = vext_s16(y[1], y[1], 1);
  y[4] = vext_s16(y[0], y[0], 2);
}

#define DEPTHWISE_CONV_NORMAL_BORDER(start, end)                         \
  for (int w = start; w < end; ++w) {                                    \
    const int w_in_start = -padding_w + w * Stride_w;                    \
    const int w_in_end = w_in_start + 5;                                 \
    const int w_start = w_in_start > 0 ? w_in_start : 0;                 \
    const int w_end = w_in_end < input_w ? w_in_end : input_w;           \
    int32_t value = 0;                                                   \
    for (int h_in = h_start; h_in < h_end; ++h_in) {                     \
      for (int w_in = w_start; w_in < w_end; ++w_in) {                   \
        value += filter[(h_in - h_in_start) * 5 + (w_in - w_in_start)] * \
                 input[h_in * input_w + w_in];                           \
      }                                                                  \
    }                                                                    \
    output_ptr[w] = value;                                               \
  }

template <int Stride_h, int Stride_w>
inline void DepthwiseConv5x5NormalRow(const int8_t *input, const int8_t *filter,
                                      const int h_output, const int input_h,
                                      const int input_w, const int padding_h,
                                      const int padding_w, const int output_w,
                                      int32_t *output, int16x4_t *ker,
                                      int16_t *ker1) {
  const int h_in_start = -padding_h + h_output * Stride_h;
  const int h_in_end = h_in_start + 5;
  const int h_start = h_in_start > 0 ? h_in_start : 0;
  const int h_end = h_in_end < input_h ? h_in_end : input_h;

  int valid_w_start = (padding_w + Stride_w - 1) / Stride_w;
  int valid_w_end = output_w - valid_w_start;
  int32_t *output_ptr = output + h_output * output_w;
  // border left
  DEPTHWISE_CONV_NORMAL_BORDER(0, valid_w_start)
  // middle
  int output_tiles = (valid_w_end - valid_w_start) >> 2;
  int16x4_t _x[5];
  int32x4_t _sum;
  // valid w
  for (int w = 0; w < output_tiles * 4; w += 4) {
    _sum = vdupq_n_s32(0);
    int output_offset = valid_w_start + w;
    int input_w_offset = output_offset * Stride_w - padding_w;
    for (int h_in = h_start; h_in < h_end; ++h_in) {
      int index = h_in - h_in_start;
      Depth5x5NormalRowLoadInput<Stride_w>(
          input + h_in * input_w + input_w_offset, _x);
      _sum = vmlal_n_s16(_sum, _x[0], ker1[index]);
      _sum = vmlal_lane_s16(_sum, _x[1], ker[index], 0);
      _sum = vmlal_lane_s16(_sum, _x[2], ker[index], 1);
      _sum = vmlal_lane_s16(_sum, _x[3], ker[index], 2);
      _sum = vmlal_lane_s16(_sum, _x[4], ker[index], 3);
    }
    vst1q_s32(output_ptr + output_offset, _sum);
  }
  // remain valid w
  int remain = (valid_w_end - valid_w_start) & 0x3;
  if (remain > 0) {
    _sum = vdupq_n_s32(0);
    int remain_start = valid_w_start + (output_tiles << 2);
    int input_w_offset = remain_start * Stride_w - padding_w;
    int32_t *output_ptr0 = output_ptr + remain_start;

    for (int h_in = h_start; h_in < h_end; ++h_in) {
      int index = h_in - h_in_start;
      Depth5x5NormalRowLoadInput<Stride_w>(
          input + h_in * input_w + input_w_offset, _x);
      _sum = vmlal_n_s16(_sum, _x[0], ker1[index]);
      _sum = vmlal_lane_s16(_sum, _x[1], ker[index], 0);
      _sum = vmlal_lane_s16(_sum, _x[2], ker[index], 1);
      _sum = vmlal_lane_s16(_sum, _x[3], ker[index], 2);
      _sum = vmlal_lane_s16(_sum, _x[4], ker[index], 3);
    }
    switch (remain) {
      case 1:
        vst1_lane_s32(output_ptr0, vget_low_s32(_sum), 0);
        break;
      case 2:
        vst1_s32(output_ptr0, vget_low_s32(_sum));
        break;
      case 3:
        vst1_s32(output_ptr0, vget_low_s32(_sum));
        vst1_lane_s32(output_ptr0 + 2, vget_high_s32(_sum), 0);
        break;
    }
  }
  // border right
  DEPTHWISE_CONV_NORMAL_BORDER(valid_w_end, output_w)
}

template <>
void DepthwiseConv5x5S1<int8_t, int32_t>(const framework::Tensor &input,
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

#pragma omp parallel for num_threads(framework::threads())
  for (int g = 0; g < input.dims()[1]; ++g) {
    const int8_t *input_ptr = input_data + g * image_size;
    const int8_t *filter_ptr = filter_data + g * 25;
    int32_t *output_ptr = out_data + g * out_image_size;

    const int8_t *filter_ptr0 = filter_ptr;
    const int8_t *filter_ptr1 = filter_ptr0 + 5;
    const int8_t *filter_ptr2 = filter_ptr1 + 5;
    const int8_t *filter_ptr3 = filter_ptr2 + 5;
    const int8_t *filter_ptr4 = filter_ptr3 + 5;
    int16_t kernel[5] = {*filter_ptr0, *filter_ptr1, *filter_ptr2, *filter_ptr3,
                         *filter_ptr4};
    int16x4_t _k0 = vget_low_s16(vmovl_s8(vld1_s8(filter_ptr0 + 1)));
    int16x4_t _k1 = vget_low_s16(vmovl_s8(vld1_s8(filter_ptr1 + 1)));
    int16x4_t _k2 = vget_low_s16(vmovl_s8(vld1_s8(filter_ptr2 + 1)));
    int16x4_t _k3 = vget_low_s16(vmovl_s8(vld1_s8(filter_ptr3 + 1)));
    int16x4_t _k4 = vget_low_s16(vmovl_s8(vld1_s8(filter_ptr4 + 1)));
    int16x4_t _k5 = vld1_s16(kernel);
    int16x4_t _k6 = vld1_s16(kernel + 4);
    int16x8_t _ker0 = vcombine_s16(_k0, _k1);
    int16x8_t _ker1 = vcombine_s16(_k2, _k3);
    int16x8_t _ker2 = vcombine_s16(_k4, _k5);
    int16x8_t _ker3 = vcombine_s16(_k6, _k6);
    int16x4_t _ker[7] = {_k0, _k1, _k2, _k3, _k4, _k5, _k6};

    // pad top
    for (int h = 0; h < valid_h_start; ++h) {
      DepthwiseConv5x5NormalRow<1, 1>(input_ptr, filter_ptr, h, input_h,
                                      input_w, padding_h, padding_w, output_w,
                                      output_ptr, _ker, kernel);
    }

    // output 4x4
    int output_w_tiles = valid_w / 8;
    int output_w_remain = valid_w - output_w_tiles * 8;
    for (int h = valid_h_start; h < valid_h_end - 1; h += 2) {
      const int8_t *input_ptr0 = input_ptr + (h - padding_h) * input_w;
      const int8_t *input_ptr1 = input_ptr0 + input_w;
      const int8_t *input_ptr2 = input_ptr1 + input_w;
      const int8_t *input_ptr3 = input_ptr2 + input_w;
      const int8_t *input_ptr4 = input_ptr3 + input_w;
      const int8_t *input_ptr5 = input_ptr4 + input_w;
      int32_t *output_ptr0 = output_ptr + h * output_w;
      int32_t *output_ptr1 = output_ptr0 + output_w;
      // pad left
      if (padding_w) {
        int16x4_t row0 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr0)));
        int16x4_t row1 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr1)));
        int16x4_t row2 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr2)));
        int16x4_t row3 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr3)));
        int16x4_t row4 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr4)));
        int16x4_t row5 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr5)));
        int16x4_t zero = vdup_n_s16(0);
        int32x4_t acc0, acc1;
        for (int w = valid_w_start - 1; w >= 0; --w) {
          int padding = padding_w - w;
          if (padding >= 5) {
            output_ptr0[w] = 0;
            output_ptr1[w] = 0;
          } else {
            acc0 = vmull_s16(row0, _ker[0]);
            acc0 = vmlal_s16(acc0, row1, _ker[1]);
            acc0 = vmlal_s16(acc0, row2, _ker[2]);
            acc0 = vmlal_s16(acc0, row3, _ker[3]);
            acc0 = vmlal_s16(acc0, row4, _ker[4]);
            acc1 = vmull_s16(row1, _ker[0]);
            acc1 = vmlal_s16(acc1, row2, _ker[1]);
            acc1 = vmlal_s16(acc1, row3, _ker[2]);
            acc1 = vmlal_s16(acc1, row4, _ker[3]);
            acc1 = vmlal_s16(acc1, row5, _ker[4]);
            acc0 = vpaddq_s32(acc0, acc1);
            int32x2_t sum = vpadd_s32(vget_low_s32(acc0), vget_high_s32(acc0));
            vst1_lane_s32(output_ptr0 + w, sum, 0);
            vst1_lane_s32(output_ptr1 + w, sum, 1);

            row0 = vext_s16(zero, row0, 3);
            row1 = vext_s16(zero, row1, 3);
            row2 = vext_s16(zero, row2, 3);
            row3 = vext_s16(zero, row3, 3);
            row4 = vext_s16(zero, row4, 3);
            row5 = vext_s16(zero, row5, 3);
          }
        }
        output_ptr0 += valid_w_start;
        output_ptr1 += valid_w_start;
      }
      // valid
      int loop = output_w_tiles;
      int w_remain = output_w_remain;
      asm volatile(
          "cmp        %[loop], #0                     \n"
          "ble        start_remain4_%=                \n"
          "mov        r0, #8                          \n"
          "loop_2h8w_%=:                              \n"
          "vld1.s8    {d10-d11}, [%[input_ptr0]], r0  \n"
          "vld1.s8    {d12-d13}, [%[input_ptr1]], r0  \n"
          "vld1.s8    {d14-d15}, [%[input_ptr2]], r0  \n"
          "vmovl.s8   q8, d10                         \n"
          "vmovl.s8   q9, d11                         \n"
          "vmull.s16  q12, d16, %f[ker2][0]           \n"
          "vmull.s16  q13, d17, %f[ker2][0]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %e[ker0][0]           \n"
          "vmlal.s16  q13, d21, %e[ker0][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %e[ker0][1]           \n"
          "vmlal.s16  q13, d21, %e[ker0][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %e[ker0][2]           \n"
          "vmlal.s16  q13, d21, %e[ker0][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %e[ker0][3]           \n"
          "vmlal.s16  q13, d21, %e[ker0][3]           \n"

          "vmovl.s8   q8, d12                         \n"
          "vmovl.s8   q9, d13                         \n"
          "vmlal.s16  q12, d16, %f[ker2][1]           \n"
          "vmlal.s16  q13, d17, %f[ker2][1]           \n"
          "vmull.s16  q14, d16, %f[ker2][0]           \n"
          "vmull.s16  q15, d17, %f[ker2][0]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %f[ker0][0]           \n"
          "vmlal.s16  q13, d21, %f[ker0][0]           \n"
          "vmlal.s16  q14, d20, %e[ker0][0]           \n"
          "vmlal.s16  q15, d21, %e[ker0][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %f[ker0][1]           \n"
          "vmlal.s16  q13, d21, %f[ker0][1]           \n"
          "vmlal.s16  q14, d20, %e[ker0][1]           \n"
          "vmlal.s16  q15, d21, %e[ker0][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %f[ker0][2]           \n"
          "vmlal.s16  q13, d21, %f[ker0][2]           \n"
          "vmlal.s16  q14, d20, %e[ker0][2]           \n"
          "vmlal.s16  q15, d21, %e[ker0][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %f[ker0][3]           \n"
          "vmlal.s16  q13, d21, %f[ker0][3]           \n"
          "vmlal.s16  q14, d20, %e[ker0][3]           \n"
          "vmlal.s16  q15, d21, %e[ker0][3]           \n"

          "vmovl.s8   q8, d14                         \n"
          "vmovl.s8   q9, d15                         \n"
          "vmlal.s16  q12, d16, %f[ker2][2]           \n"
          "vmlal.s16  q13, d17, %f[ker2][2]           \n"
          "vmlal.s16  q14, d16, %f[ker2][1]           \n"
          "vmlal.s16  q15, d17, %f[ker2][1]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %e[ker1][0]           \n"
          "vmlal.s16  q13, d21, %e[ker1][0]           \n"
          "vmlal.s16  q14, d20, %f[ker0][0]           \n"
          "vmlal.s16  q15, d21, %f[ker0][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %e[ker1][1]           \n"
          "vmlal.s16  q13, d21, %e[ker1][1]           \n"
          "vmlal.s16  q14, d20, %f[ker0][1]           \n"
          "vmlal.s16  q15, d21, %f[ker0][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %e[ker1][2]           \n"
          "vmlal.s16  q13, d21, %e[ker1][2]           \n"
          "vmlal.s16  q14, d20, %f[ker0][2]           \n"
          "vmlal.s16  q15, d21, %f[ker0][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %e[ker1][3]           \n"
          "vmlal.s16  q13, d21, %e[ker1][3]           \n"
          "vmlal.s16  q14, d20, %f[ker0][3]           \n"
          "vmlal.s16  q15, d21, %f[ker0][3]           \n"

          "vld1.s8    {d10-d11}, [%[input_ptr3]], r0  \n"
          "vld1.s8    {d12-d13}, [%[input_ptr4]], r0  \n"
          "vld1.s8    {d14-d15}, [%[input_ptr5]], r0  \n"
          "vmovl.s8   q8, d10                         \n"
          "vmovl.s8   q9, d11                         \n"
          "vmlal.s16  q12, d16, %f[ker2][3]           \n"
          "vmlal.s16  q13, d17, %f[ker2][3]           \n"
          "vmlal.s16  q14, d16, %f[ker2][2]           \n"
          "vmlal.s16  q15, d17, %f[ker2][2]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %f[ker1][0]           \n"
          "vmlal.s16  q13, d21, %f[ker1][0]           \n"
          "vmlal.s16  q14, d20, %e[ker1][0]           \n"
          "vmlal.s16  q15, d21, %e[ker1][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %f[ker1][1]           \n"
          "vmlal.s16  q13, d21, %f[ker1][1]           \n"
          "vmlal.s16  q14, d20, %e[ker1][1]           \n"
          "vmlal.s16  q15, d21, %e[ker1][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %f[ker1][2]           \n"
          "vmlal.s16  q13, d21, %f[ker1][2]           \n"
          "vmlal.s16  q14, d20, %e[ker1][2]           \n"
          "vmlal.s16  q15, d21, %e[ker1][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %f[ker1][3]           \n"
          "vmlal.s16  q13, d21, %f[ker1][3]           \n"
          "vmlal.s16  q14, d20, %e[ker1][3]           \n"
          "vmlal.s16  q15, d21, %e[ker1][3]           \n"

          "vmovl.s8   q8, d12                         \n"
          "vmovl.s8   q9, d13                         \n"
          "vmlal.s16  q12, d16, %e[ker3][0]           \n"
          "vmlal.s16  q13, d17, %e[ker3][0]           \n"
          "vmlal.s16  q14, d16, %f[ker2][3]           \n"
          "vmlal.s16  q15, d17, %f[ker2][3]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %e[ker2][0]           \n"
          "vmlal.s16  q13, d21, %e[ker2][0]           \n"
          "vmlal.s16  q14, d20, %f[ker1][0]           \n"
          "vmlal.s16  q15, d21, %f[ker1][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %e[ker2][1]           \n"
          "vmlal.s16  q13, d21, %e[ker2][1]           \n"
          "vmlal.s16  q14, d20, %f[ker1][1]           \n"
          "vmlal.s16  q15, d21, %f[ker1][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %e[ker2][2]           \n"
          "vmlal.s16  q13, d21, %e[ker2][2]           \n"
          "vmlal.s16  q14, d20, %f[ker1][2]           \n"
          "vmlal.s16  q15, d21, %f[ker1][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %e[ker2][3]           \n"
          "vmlal.s16  q13, d21, %e[ker2][3]           \n"
          "vmlal.s16  q14, d20, %f[ker1][3]           \n"
          "vmlal.s16  q15, d21, %f[ker1][3]           \n"

          "vmovl.s8   q8, d14                         \n"
          "vmovl.s8   q9, d15                         \n"
          "vmlal.s16  q14, d16, %e[ker3][0]           \n"
          "vmlal.s16  q15, d17, %e[ker3][0]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q14, d20, %e[ker2][0]           \n"
          "vmlal.s16  q15, d21, %e[ker2][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q14, d20, %e[ker2][1]           \n"
          "vmlal.s16  q15, d21, %e[ker2][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q14, d20, %e[ker2][2]           \n"
          "vmlal.s16  q15, d21, %e[ker2][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q14, d20, %e[ker2][3]           \n"
          "vmlal.s16  q15, d21, %e[ker2][3]           \n"

          // restore output
          "vst1.32    {q12-q13}, [%[output_ptr0]]!    \n"
          "vst1.32    {q14-q15}, [%[output_ptr1]]!    \n"
          "subs       %[loop], #1                     \n"
          "bne        loop_2h8w_%=                    \n"

          "start_remain4_%=:                          \n"
          "cmp        %[remain], #4                   \n"
          "blt        start_remain_%=                 \n"
          "mov        r0, #4                          \n"
          "vld1.s8    {d10}, [%[input_ptr0]], r0      \n"
          "vld1.s8    {d12}, [%[input_ptr1]], r0      \n"
          "vld1.s8    {d14}, [%[input_ptr2]], r0      \n"
          "vmovl.s8   q8, d10                         \n"
          "vmull.s16  q12, d16, %f[ker2][0]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %e[ker0][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %e[ker0][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %e[ker0][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %e[ker0][3]           \n"

          "vmovl.s8   q8, d12                         \n"
          "vmlal.s16  q12, d16, %f[ker2][1]           \n"
          "vmull.s16  q14, d16, %f[ker2][0]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %f[ker0][0]           \n"
          "vmlal.s16  q14, d20, %e[ker0][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %f[ker0][1]           \n"
          "vmlal.s16  q14, d20, %e[ker0][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %f[ker0][2]           \n"
          "vmlal.s16  q14, d20, %e[ker0][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %f[ker0][3]           \n"
          "vmlal.s16  q14, d20, %e[ker0][3]           \n"

          "vmovl.s8   q8, d14                         \n"
          "vmlal.s16  q12, d16, %f[ker2][2]           \n"
          "vmlal.s16  q14, d16, %f[ker2][1]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %e[ker1][0]           \n"
          "vmlal.s16  q14, d20, %f[ker0][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %e[ker1][1]           \n"
          "vmlal.s16  q14, d20, %f[ker0][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %e[ker1][2]           \n"
          "vmlal.s16  q14, d20, %f[ker0][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %e[ker1][3]           \n"
          "vmlal.s16  q14, d20, %f[ker0][3]           \n"

          "vld1.s8    {d10}, [%[input_ptr3]], r0      \n"
          "vld1.s8    {d12}, [%[input_ptr4]], r0      \n"
          "vld1.s8    {d14}, [%[input_ptr5]], r0      \n"
          "vmovl.s8   q8, d10                         \n"
          "vmlal.s16  q12, d16, %f[ker2][3]           \n"
          "vmlal.s16  q14, d16, %f[ker2][2]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %f[ker1][0]           \n"
          "vmlal.s16  q14, d20, %e[ker1][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %f[ker1][1]           \n"
          "vmlal.s16  q14, d20, %e[ker1][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %f[ker1][2]           \n"
          "vmlal.s16  q14, d20, %e[ker1][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %f[ker1][3]           \n"
          "vmlal.s16  q14, d20, %e[ker1][3]           \n"

          "vmovl.s8   q8, d12                         \n"
          "vmlal.s16  q12, d16, %e[ker3][0]           \n"
          "vmlal.s16  q14, d16, %f[ker2][3]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %e[ker2][0]           \n"
          "vmlal.s16  q14, d20, %f[ker1][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %e[ker2][1]           \n"
          "vmlal.s16  q14, d20, %f[ker1][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %e[ker2][2]           \n"
          "vmlal.s16  q14, d20, %f[ker1][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %e[ker2][3]           \n"
          "vmlal.s16  q14, d20, %f[ker1][3]           \n"

          "vmovl.s8   q8, d14                         \n"
          "vmlal.s16  q14, d16, %e[ker3][0]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q14, d20, %e[ker2][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q14, d20, %e[ker2][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q14, d20, %e[ker2][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q14, d20, %e[ker2][3]           \n"

          // restore output
          "vst1.32    {d24-d25}, [%[output_ptr0]]!    \n"
          "vst1.32    {d28-d29}, [%[output_ptr1]]!    \n"
          "sub        %[remain], #4                   \n"

          "start_remain_%=:                           \n"
          "cmp        %[remain], #0                   \n"
          "ble        end_%=                          \n"
          "mov        r0, %[remain]                   \n"
          "vld1.s8    {d10}, [%[input_ptr0]], r0      \n"
          "vld1.s8    {d12}, [%[input_ptr1]], r0      \n"
          "vld1.s8    {d14}, [%[input_ptr2]], r0      \n"
          "vmovl.s8   q8, d10                         \n"
          "vmull.s16  q12, d16, %f[ker2][0]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %e[ker0][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %e[ker0][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %e[ker0][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %e[ker0][3]           \n"

          "vmovl.s8   q8, d12                         \n"
          "vmlal.s16  q12, d16, %f[ker2][1]           \n"
          "vmull.s16  q14, d16, %f[ker2][0]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %f[ker0][0]           \n"
          "vmlal.s16  q14, d20, %e[ker0][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %f[ker0][1]           \n"
          "vmlal.s16  q14, d20, %e[ker0][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %f[ker0][2]           \n"
          "vmlal.s16  q14, d20, %e[ker0][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %f[ker0][3]           \n"
          "vmlal.s16  q14, d20, %e[ker0][3]           \n"

          "vmovl.s8   q8, d14                         \n"
          "vmlal.s16  q12, d16, %f[ker2][2]           \n"
          "vmlal.s16  q14, d16, %f[ker2][1]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %e[ker1][0]           \n"
          "vmlal.s16  q14, d20, %f[ker0][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %e[ker1][1]           \n"
          "vmlal.s16  q14, d20, %f[ker0][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %e[ker1][2]           \n"
          "vmlal.s16  q14, d20, %f[ker0][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %e[ker1][3]           \n"
          "vmlal.s16  q14, d20, %f[ker0][3]           \n"

          "vld1.s8    {d10}, [%[input_ptr3]], r0      \n"
          "vld1.s8    {d12}, [%[input_ptr4]], r0      \n"
          "vld1.s8    {d14}, [%[input_ptr5]], r0      \n"
          "vmovl.s8   q8, d10                         \n"
          "vmlal.s16  q12, d16, %f[ker2][3]           \n"
          "vmlal.s16  q14, d16, %f[ker2][2]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %f[ker1][0]           \n"
          "vmlal.s16  q14, d20, %e[ker1][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %f[ker1][1]           \n"
          "vmlal.s16  q14, d20, %e[ker1][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %f[ker1][2]           \n"
          "vmlal.s16  q14, d20, %e[ker1][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %f[ker1][3]           \n"
          "vmlal.s16  q14, d20, %e[ker1][3]           \n"

          "vmovl.s8   q8, d12                         \n"
          "vmlal.s16  q12, d16, %e[ker3][0]           \n"
          "vmlal.s16  q14, d16, %f[ker2][3]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %e[ker2][0]           \n"
          "vmlal.s16  q14, d20, %f[ker1][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %e[ker2][1]           \n"
          "vmlal.s16  q14, d20, %f[ker1][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %e[ker2][2]           \n"
          "vmlal.s16  q14, d20, %f[ker1][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %e[ker2][3]           \n"
          "vmlal.s16  q14, d20, %f[ker1][3]           \n"

          "vmovl.s8   q8, d14                         \n"
          "vmlal.s16  q14, d16, %e[ker3][0]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q14, d20, %e[ker2][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q14, d20, %e[ker2][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q14, d20, %e[ker2][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q14, d20, %e[ker2][3]           \n"

          "cmp        %[remain], #2                   \n"
          "blt        store_2h1w_%=                   \n"
          "vst1.32    {d24}, [%[output_ptr0]]!        \n"
          "vst1.32    {d28}, [%[output_ptr1]]!        \n"
          "cmp        %[remain], #3                   \n"
          "blt        end_%=                          \n"
          "vst1.32    {d25[0]}, [%[output_ptr0]]!     \n"
          "vst1.32    {d29[0]}, [%[output_ptr1]]!     \n"
          "b          end_%=                          \n"

          "store_2h1w_%=:                             \n"
          "vst1.32    {d24[0]}, [%[output_ptr0]]!     \n"
          "vst1.32    {d28[0]}, [%[output_ptr1]]!     \n"
          "end_%=:                                    \n"
          : [input_ptr0] "+r"(input_ptr0), [input_ptr1] "+r"(input_ptr1),
            [input_ptr2] "+r"(input_ptr2), [input_ptr3] "+r"(input_ptr3),
            [input_ptr4] "+r"(input_ptr4), [input_ptr5] "+r"(input_ptr5),
            [output_ptr0] "+r"(output_ptr0), [output_ptr1] "+r"(output_ptr1),
            [loop] "+r"(loop), [remain] "+r"(w_remain)
          : [ker0] "w"(_ker0), [ker1] "w"(_ker1), [ker2] "w"(_ker2),
            [ker3] "w"(_ker3)
          : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
            "q12", "q13", "q14", "q15", "r0");
      // pad right
      if (padding_w) {
        int16x4_t row0 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr0)));
        int16x4_t row1 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr1)));
        int16x4_t row2 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr2)));
        int16x4_t row3 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr3)));
        int16x4_t row4 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr4)));
        int16x4_t row5 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr5)));
        int16x4_t zero = vdup_n_s16(0);
        int32x4_t acc0, acc1;
        for (int w = valid_w_end; w < output_w; ++w) {
          int padding = w + 5 - (padding_w + input_w);
          if (padding >= 5) {
            *output_ptr0 = 0;
            *output_ptr1 = 0;
          } else {
            int iw = w - valid_w_end;
            int32_t sum0 = input_ptr0[iw] * filter_ptr0[0] +
                           input_ptr1[iw] * filter_ptr1[0] +
                           input_ptr2[iw] * filter_ptr2[0] +
                           input_ptr3[iw] * filter_ptr3[0] +
                           input_ptr4[iw] * filter_ptr4[0];
            int32_t sum1 = input_ptr1[iw] * filter_ptr0[0] +
                           input_ptr2[iw] * filter_ptr1[0] +
                           input_ptr3[iw] * filter_ptr2[0] +
                           input_ptr4[iw] * filter_ptr3[0] +
                           input_ptr5[iw] * filter_ptr4[0];
            row0 = vext_s16(row0, zero, 1);
            row1 = vext_s16(row1, zero, 1);
            row2 = vext_s16(row2, zero, 1);
            row3 = vext_s16(row3, zero, 1);
            row4 = vext_s16(row4, zero, 1);
            row5 = vext_s16(row5, zero, 1);
            acc0 = vmull_s16(row0, _ker[0]);
            acc0 = vmlal_s16(acc0, row1, _ker[1]);
            acc0 = vmlal_s16(acc0, row2, _ker[2]);
            acc0 = vmlal_s16(acc0, row3, _ker[3]);
            acc0 = vmlal_s16(acc0, row4, _ker[4]);
            acc1 = vmull_s16(row1, _ker[0]);
            acc1 = vmlal_s16(acc1, row2, _ker[1]);
            acc1 = vmlal_s16(acc1, row3, _ker[2]);
            acc1 = vmlal_s16(acc1, row4, _ker[3]);
            acc1 = vmlal_s16(acc1, row5, _ker[4]);
            acc0 = vpaddq_s32(acc0, acc1);
            int32x2_t sum = vpadd_s32(vget_low_s32(acc0), vget_high_s32(acc0));
            sum0 += vget_lane_s32(sum, 0);
            sum1 += vget_lane_s32(sum, 1);
            *output_ptr0 = sum0;
            *output_ptr1 = sum1;
          }
          output_ptr0++;
          output_ptr1++;
        }
      }
    }
    // remain height
    int start_h = valid_h_start + (valid_h & 0xfffe);
    if (start_h < valid_h_end) {
      const int8_t *input_ptr0 = input_ptr + (start_h - padding_h) * input_w;
      const int8_t *input_ptr1 = input_ptr0 + input_w;
      const int8_t *input_ptr2 = input_ptr1 + input_w;
      const int8_t *input_ptr3 = input_ptr2 + input_w;
      const int8_t *input_ptr4 = input_ptr3 + input_w;
      int32_t *output_ptr0 = output_ptr + start_h * output_w;
      // pad left
      if (padding_w) {
        int16x4_t row0 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr0)));
        int16x4_t row1 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr1)));
        int16x4_t row2 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr2)));
        int16x4_t row3 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr3)));
        int16x4_t row4 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr4)));
        int16x4_t zero = vdup_n_s16(0);
        int32x4_t acc;
        for (int w = valid_w_start - 1; w >= 0; --w) {
          int padding = padding_w - w;
          if (padding >= 5) {
            output_ptr0[w] = 0;
          } else {
            acc = vmull_s16(row0, _ker[0]);
            acc = vmlal_s16(acc, row1, _ker[1]);
            acc = vmlal_s16(acc, row2, _ker[2]);
            acc = vmlal_s16(acc, row3, _ker[3]);
            acc = vmlal_s16(acc, row4, _ker[4]);
            int32x2_t sum = vpadd_s32(vget_low_s32(acc), vget_high_s32(acc));
            sum = vpadd_s32(sum, sum);
            vst1_lane_s32(output_ptr0 + w, sum, 0);

            row0 = vext_s16(zero, row0, 3);
            row1 = vext_s16(zero, row1, 3);
            row2 = vext_s16(zero, row2, 3);
            row3 = vext_s16(zero, row3, 3);
            row4 = vext_s16(zero, row4, 3);
          }
        }
        output_ptr0 += valid_w_start;
      }
      // valid
      int loop = output_w_tiles;
      int w_remain = output_w_remain;
      asm volatile(
          "cmp        %[loop], #0                     \n"
          "ble        start_remain4_%=                \n"
          "mov        r0, #8                          \n"
          "loop_1h8w_%=:                              \n"
          "vld1.s8    {d10-d11}, [%[input_ptr0]], r0  \n"
          "vld1.s8    {d12-d13}, [%[input_ptr1]], r0  \n"
          "vld1.s8    {d14-d15}, [%[input_ptr2]], r0  \n"
          "vmovl.s8   q8, d10                         \n"
          "vmovl.s8   q9, d11                         \n"
          "vmull.s16  q12, d16, %f[ker2][0]           \n"
          "vmull.s16  q13, d17, %f[ker2][0]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %e[ker0][0]           \n"
          "vmlal.s16  q13, d21, %e[ker0][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %e[ker0][1]           \n"
          "vmlal.s16  q13, d21, %e[ker0][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %e[ker0][2]           \n"
          "vmlal.s16  q13, d21, %e[ker0][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %e[ker0][3]           \n"
          "vmlal.s16  q13, d21, %e[ker0][3]           \n"

          "vmovl.s8   q8, d12                         \n"
          "vmovl.s8   q9, d13                         \n"
          "vmlal.s16  q12, d16, %f[ker2][1]           \n"
          "vmlal.s16  q13, d17, %f[ker2][1]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %f[ker0][0]           \n"
          "vmlal.s16  q13, d21, %f[ker0][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %f[ker0][1]           \n"
          "vmlal.s16  q13, d21, %f[ker0][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %f[ker0][2]           \n"
          "vmlal.s16  q13, d21, %f[ker0][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %f[ker0][3]           \n"
          "vmlal.s16  q13, d21, %f[ker0][3]           \n"

          "vmovl.s8   q8, d14                         \n"
          "vmovl.s8   q9, d15                         \n"
          "vmlal.s16  q12, d16, %f[ker2][2]           \n"
          "vmlal.s16  q13, d17, %f[ker2][2]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %e[ker1][0]           \n"
          "vmlal.s16  q13, d21, %e[ker1][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %e[ker1][1]           \n"
          "vmlal.s16  q13, d21, %e[ker1][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %e[ker1][2]           \n"
          "vmlal.s16  q13, d21, %e[ker1][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %e[ker1][3]           \n"
          "vmlal.s16  q13, d21, %e[ker1][3]           \n"

          "vld1.s8    {d10-d11}, [%[input_ptr3]], r0  \n"
          "vld1.s8    {d12-d13}, [%[input_ptr4]], r0  \n"
          "vmovl.s8   q8, d10                         \n"
          "vmovl.s8   q9, d11                         \n"
          "vmlal.s16  q12, d16, %f[ker2][3]           \n"
          "vmlal.s16  q13, d17, %f[ker2][3]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %f[ker1][0]           \n"
          "vmlal.s16  q13, d21, %f[ker1][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %f[ker1][1]           \n"
          "vmlal.s16  q13, d21, %f[ker1][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %f[ker1][2]           \n"
          "vmlal.s16  q13, d21, %f[ker1][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %f[ker1][3]           \n"
          "vmlal.s16  q13, d21, %f[ker1][3]           \n"

          "vmovl.s8   q8, d12                         \n"
          "vmovl.s8   q9, d13                         \n"
          "vmlal.s16  q12, d16, %e[ker3][0]           \n"
          "vmlal.s16  q13, d17, %e[ker3][0]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %e[ker2][0]           \n"
          "vmlal.s16  q13, d21, %e[ker2][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %e[ker2][1]           \n"
          "vmlal.s16  q13, d21, %e[ker2][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %e[ker2][2]           \n"
          "vmlal.s16  q13, d21, %e[ker2][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %e[ker2][3]           \n"
          "vmlal.s16  q13, d21, %e[ker2][3]           \n"

          // restore output
          "vst1.32    {q12-q13}, [%[output_ptr0]]!    \n"
          "subs       %[loop], #1                     \n"
          "bne        loop_1h8w_%=                    \n"

          "start_remain4_%=:                          \n"
          "cmp        %[remain], #4                   \n"
          "blt        start_remain_%=                 \n"
          "mov        r0, #4                          \n"
          "vld1.s8    {d10}, [%[input_ptr0]], r0      \n"
          "vld1.s8    {d12}, [%[input_ptr1]], r0      \n"
          "vld1.s8    {d14}, [%[input_ptr2]], r0      \n"
          "vmovl.s8   q8, d10                         \n"
          "vmull.s16  q12, d16, %f[ker2][0]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %e[ker0][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %e[ker0][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %e[ker0][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %e[ker0][3]           \n"

          "vmovl.s8   q8, d12                         \n"
          "vmlal.s16  q12, d16, %f[ker2][1]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %f[ker0][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %f[ker0][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %f[ker0][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %f[ker0][3]           \n"

          "vmovl.s8   q8, d14                         \n"
          "vmlal.s16  q12, d16, %f[ker2][2]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %e[ker1][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %e[ker1][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %e[ker1][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %e[ker1][3]           \n"

          "vld1.s8    {d10}, [%[input_ptr3]], r0      \n"
          "vld1.s8    {d12}, [%[input_ptr4]], r0      \n"
          "vmovl.s8   q8, d10                         \n"
          "vmlal.s16  q12, d16, %f[ker2][3]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %f[ker1][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %f[ker1][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %f[ker1][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %f[ker1][3]           \n"

          "vmovl.s8   q8, d12                         \n"
          "vmlal.s16  q12, d16, %e[ker3][0]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %e[ker2][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %e[ker2][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %e[ker2][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %e[ker2][3]           \n"

          // restore output
          "vst1.32    {d24-d25}, [%[output_ptr0]]!    \n"
          "sub        %[remain], #4                   \n"

          "start_remain_%=:                           \n"
          "cmp        %[remain], #0                   \n"
          "ble        end_%=                          \n"
          "mov        r0, %[remain]                   \n"
          "vld1.s8    {d10}, [%[input_ptr0]], r0      \n"
          "vld1.s8    {d12}, [%[input_ptr1]], r0      \n"
          "vld1.s8    {d14}, [%[input_ptr2]], r0      \n"
          "vmovl.s8   q8, d10                         \n"
          "vmull.s16  q12, d16, %f[ker2][0]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %e[ker0][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %e[ker0][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %e[ker0][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %e[ker0][3]           \n"

          "vmovl.s8   q8, d12                         \n"
          "vmlal.s16  q12, d16, %f[ker2][1]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %f[ker0][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %f[ker0][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %f[ker0][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %f[ker0][3]           \n"

          "vmovl.s8   q8, d14                         \n"
          "vmlal.s16  q12, d16, %f[ker2][2]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %e[ker1][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %e[ker1][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %e[ker1][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %e[ker1][3]           \n"

          "vld1.s8    {d10}, [%[input_ptr3]], r0      \n"
          "vld1.s8    {d12}, [%[input_ptr4]], r0      \n"
          "vmovl.s8   q8, d10                         \n"
          "vmlal.s16  q12, d16, %f[ker2][3]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %f[ker1][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %f[ker1][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %f[ker1][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %f[ker1][3]           \n"

          "vmovl.s8   q8, d12                         \n"
          "vmlal.s16  q12, d16, %e[ker3][0]           \n"
          "vext.s16   q10, q8, q9, #1                 \n"
          "vmlal.s16  q12, d20, %e[ker2][0]           \n"
          "vext.s16   q10, q8, q9, #2                 \n"
          "vmlal.s16  q12, d20, %e[ker2][1]           \n"
          "vext.s16   q10, q8, q9, #3                 \n"
          "vmlal.s16  q12, d20, %e[ker2][2]           \n"
          "vext.s16   q10, q8, q9, #4                 \n"
          "vmlal.s16  q12, d20, %e[ker2][3]           \n"

          "cmp        %[remain], #2                   \n"
          "blt        store_1h1w_%=                   \n"
          "vst1.32    {d24}, [%[output_ptr0]]!        \n"
          "cmp        %[remain], #3                   \n"
          "blt        end_%=                          \n"
          "vst1.32    {d25[0]}, [%[output_ptr0]]!     \n"
          "b          end_%=                          \n"

          "store_1h1w_%=:                             \n"
          "vst1.32    {d24[0]}, [%[output_ptr0]]!     \n"
          "end_%=:                                    \n"
          : [input_ptr0] "+r"(input_ptr0), [input_ptr1] "+r"(input_ptr1),
            [input_ptr2] "+r"(input_ptr2), [input_ptr3] "+r"(input_ptr3),
            [input_ptr4] "+r"(input_ptr4), [output_ptr0] "+r"(output_ptr0),
            [loop] "+r"(loop), [remain] "+r"(w_remain)
          : [ker0] "w"(_ker0), [ker1] "w"(_ker1), [ker2] "w"(_ker2),
            [ker3] "w"(_ker3)
          : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
            "q12", "q13", "q14", "q15", "r0");
      // pad right
      if (padding_w) {
        int16x4_t row0 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr0)));
        int16x4_t row1 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr1)));
        int16x4_t row2 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr2)));
        int16x4_t row3 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr3)));
        int16x4_t row4 = vget_low_s16(vmovl_s8(vld1_s8(input_ptr4)));
        int16x4_t zero = vdup_n_s16(0);
        int32x4_t acc;
        for (int w = valid_w_end; w < output_w; ++w) {
          int padding = w + 5 - (padding_w + input_w);
          if (padding >= 5) {
            *output_ptr0 = 0;
          } else {
            int iw = w - valid_w_end;
            int32_t sum0 = input_ptr0[iw] * filter_ptr0[0] +
                           input_ptr1[iw] * filter_ptr1[0] +
                           input_ptr2[iw] * filter_ptr2[0] +
                           input_ptr3[iw] * filter_ptr3[0] +
                           input_ptr4[iw] * filter_ptr4[0];
            row0 = vext_s16(row0, zero, 1);
            row1 = vext_s16(row1, zero, 1);
            row2 = vext_s16(row2, zero, 1);
            row3 = vext_s16(row3, zero, 1);
            row4 = vext_s16(row4, zero, 1);
            acc = vmull_s16(row0, _ker[0]);
            acc = vmlal_s16(acc, row1, _ker[1]);
            acc = vmlal_s16(acc, row2, _ker[2]);
            acc = vmlal_s16(acc, row3, _ker[3]);
            acc = vmlal_s16(acc, row4, _ker[4]);
            int32x2_t sum = vpadd_s32(vget_low_s32(acc), vget_high_s32(acc));
            sum = vpadd_s32(sum, sum);
            sum0 += vget_lane_s32(sum, 0);
            *output_ptr0 = sum0;
          }
          output_ptr0++;
        }
      }
    }
    // pad bottom
    for (int h = valid_h_end; h < output_h; ++h) {
      DepthwiseConv5x5NormalRow<1, 1>(input_ptr, filter_ptr, h, input_h,
                                      input_w, padding_h, padding_w, output_w,
                                      output_ptr, _ker, kernel);
    }
  }
}

template <>
void DepthwiseConv5x5S2<int8_t, int32_t>(const framework::Tensor &input,
                                         const framework::Tensor &filter,
                                         const std::vector<int> &paddings,
                                         framework::Tensor *output) {}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif  // __ARM_NEON__
