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

#include "operators/math/depthwise_conv5x5.h"
#include <arm_neon.h>

namespace paddle_mobile {
namespace operators {
namespace math {

#ifndef __aarch64__
inline float32x4_t vpaddq_f32(float32x4_t r0, float32x4_t r1) {
  float32x2_t sum0 = vpadd_f32(vget_low_f32(r0), vget_high_f32(r0));
  float32x2_t sum1 = vpadd_f32(vget_low_f32(r1), vget_high_f32(r1));
  return vcombine_f32(sum0, sum1);
}
#endif

template <int Stride = 1>
inline void Depth5x5NormalRowLoadInput(const float *input, float32x4_t *y) {
  y[0] = vld1q_f32(input);
  y[4] = vld1q_f32(input + 4);
  y[1] = vextq_f32(y[0], y[4], 1);
  y[2] = vextq_f32(y[0], y[4], 2);
  y[3] = vextq_f32(y[0], y[4], 3);
}

template <>
inline void Depth5x5NormalRowLoadInput<2>(const float *input, float32x4_t *y) {
  float32x4x2_t x = vld2q_f32(input);
  y[0] = x.val[0];
  y[1] = x.val[1];
  y[2] = vextq_f32(y[0], y[0], 1);
  y[3] = vextq_f32(y[1], y[1], 1);
  y[4] = vextq_f32(y[0], y[0], 2);
}

#define DEPTHWISE_CONV_NORMAL_BORDER(start, end)                         \
  for (int w = start; w < end; ++w) {                                    \
    const int w_in_start = -padding_w + w * Stride_w;                    \
    const int w_in_end = w_in_start + 5;                                 \
    const int w_start = w_in_start > 0 ? w_in_start : 0;                 \
    const int w_end = w_in_end < input_w ? w_in_end : input_w;           \
    float value = 0;                                                     \
    for (int h_in = h_start; h_in < h_end; ++h_in) {                     \
      for (int w_in = w_start; w_in < w_end; ++w_in) {                   \
        value += filter[(h_in - h_in_start) * 5 + (w_in - w_in_start)] * \
                 input[h_in * input_w + w_in];                           \
      }                                                                  \
    }                                                                    \
    output_ptr[w] = value;                                               \
  }

template <int Stride_h, int Stride_w>
inline void DepthwiseConv5x5NormalRow(const float *input, const float *filter,
                                      const int h_output, const int input_h,
                                      const int input_w, const int padding_h,
                                      const int padding_w, const int output_w,
                                      float *output, float32x4_t *ker,
                                      float32_t *ker1) {
  const int h_in_start = -padding_h + h_output * Stride_h;
  const int h_in_end = h_in_start + 5;
  const int h_start = h_in_start > 0 ? h_in_start : 0;
  const int h_end = h_in_end < input_h ? h_in_end : input_h;

  int valid_w_start = (padding_w + Stride_w - 1) / Stride_w;
  int valid_w_end = output_w - valid_w_start;
  float *output_ptr = output + h_output * output_w;
  // border left
  DEPTHWISE_CONV_NORMAL_BORDER(0, valid_w_start)
  // middle
  int output_tiles = (valid_w_end - valid_w_start) >> 2;
  float32x4_t _sum, _x[5];
  // valid w
  for (int w = 0; w < output_tiles * 4; w += 4) {
    _sum = vdupq_n_f32(0.f);
    int output_offset = valid_w_start + w;
    int input_w_offset = output_offset * Stride_w - padding_w;
    for (int h_in = h_start; h_in < h_end; ++h_in) {
      int index = h_in - h_in_start;
      Depth5x5NormalRowLoadInput<Stride_w>(
          input + h_in * input_w + input_w_offset, _x);
      _sum = vmlaq_n_f32(_sum, _x[0], ker1[index]);
      _sum = vmlaq_lane_f32(_sum, _x[1], vget_low_f32(ker[index]), 0);
      _sum = vmlaq_lane_f32(_sum, _x[2], vget_low_f32(ker[index]), 1);
      _sum = vmlaq_lane_f32(_sum, _x[3], vget_high_f32(ker[index]), 0);
      _sum = vmlaq_lane_f32(_sum, _x[4], vget_high_f32(ker[index]), 1);
    }
    vst1q_f32(output_ptr + output_offset, _sum);
  }
  // remain valid w
  int remain = (valid_w_end - valid_w_start) & 0x3;
  if (remain > 0) {
    _sum = vdupq_n_f32(0.f);
    int remain_start = valid_w_start + (output_tiles << 2);
    int input_w_offset = remain_start * Stride_w - padding_w;
    float *output_ptr0 = output_ptr + remain_start;

    for (int h_in = h_start; h_in < h_end; ++h_in) {
      int index = h_in - h_in_start;
      Depth5x5NormalRowLoadInput<Stride_w>(
          input + h_in * input_w + input_w_offset, _x);
      _sum = vmlaq_n_f32(_sum, _x[0], ker1[index]);
      _sum = vmlaq_lane_f32(_sum, _x[1], vget_low_f32(ker[index]), 0);
      _sum = vmlaq_lane_f32(_sum, _x[2], vget_low_f32(ker[index]), 1);
      _sum = vmlaq_lane_f32(_sum, _x[3], vget_high_f32(ker[index]), 0);
      _sum = vmlaq_lane_f32(_sum, _x[4], vget_high_f32(ker[index]), 1);
    }
    switch (remain) {
      case 1:
        vst1_lane_f32(output_ptr0, vget_low_f32(_sum), 0);
        break;
      case 2:
        vst1_f32(output_ptr0, vget_low_f32(_sum));
        break;
      case 3:
        vst1_f32(output_ptr0, vget_low_f32(_sum));
        vst1_lane_f32(output_ptr0 + 2, vget_high_f32(_sum), 0);
        break;
    }
  }
  // border right
  DEPTHWISE_CONV_NORMAL_BORDER(valid_w_end, output_w)
}

template <>
void DepthwiseConv5x5S1<float, float>(const framework::Tensor &input,
                                      const framework::Tensor &filter,
                                      const std::vector<int> &paddings,
                                      framework::Tensor *output) {
  const float *input_data = input.data<float>();
  const float *filter_data = filter.data<float>();
  float *out_data = output->mutable_data<float>();
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
    const float *input_ptr = input_data + g * image_size;
    const float *filter_ptr = filter_data + g * 25;
    float *output_ptr = out_data + g * out_image_size;

    const float *filter_ptr0 = filter_ptr;
    const float *filter_ptr1 = filter_ptr0 + 5;
    const float *filter_ptr2 = filter_ptr1 + 5;
    const float *filter_ptr3 = filter_ptr2 + 5;
    const float *filter_ptr4 = filter_ptr3 + 5;
    float32x4_t _ker[7];
    float32_t _ker1[5] = {*filter_ptr0, *filter_ptr1, *filter_ptr2,
                          *filter_ptr3, *filter_ptr4};
    _ker[0] = vld1q_f32(filter_ptr0 + 1);
    _ker[1] = vld1q_f32(filter_ptr1 + 1);
    _ker[2] = vld1q_f32(filter_ptr2 + 1);
    _ker[3] = vld1q_f32(filter_ptr3 + 1);
    _ker[4] = vld1q_f32(filter_ptr4 + 1);
    _ker[5] = vld1q_f32(_ker1);
    _ker[6] = vld1q_f32(_ker1 + 4);

    // pad top
    for (int h = 0; h < valid_h_start; ++h) {
      DepthwiseConv5x5NormalRow<1, 1>(input_ptr, filter_ptr, h, input_h,
                                      input_w, padding_h, padding_w, output_w,
                                      output_ptr, _ker, _ker1);
    }

    // output 4x4
    int output_w_tiles = valid_w / 4;
    int output_w_remain = valid_w - output_w_tiles * 4;
    for (int h = valid_h_start; h < valid_h_end - 1; h += 2) {
      const float *input_ptr0 = input_ptr + (h - padding_h) * input_w;
      const float *input_ptr1 = input_ptr0 + input_w;
      const float *input_ptr2 = input_ptr1 + input_w;
      const float *input_ptr3 = input_ptr2 + input_w;
      const float *input_ptr4 = input_ptr3 + input_w;
      const float *input_ptr5 = input_ptr4 + input_w;
      float *output_ptr0 = output_ptr + h * output_w;
      float *output_ptr1 = output_ptr0 + output_w;
      // pad left
      if (padding_w) {
        float32x4_t row0 = vld1q_f32(input_ptr0);
        float32x4_t row1 = vld1q_f32(input_ptr1);
        float32x4_t row2 = vld1q_f32(input_ptr2);
        float32x4_t row3 = vld1q_f32(input_ptr3);
        float32x4_t row4 = vld1q_f32(input_ptr4);
        float32x4_t row5 = vld1q_f32(input_ptr5);
        float32x4_t zero = vdupq_n_f32(0.f);
        float32x4_t acc0, acc1;
        for (int w = valid_w_start - 1; w >= 0; --w) {
          int padding = padding_w - w;
          if (padding >= 5) {
            output_ptr0[w] = 0.f;
            output_ptr1[w] = 0.f;
          } else {
            acc0 = vmulq_f32(row0, _ker[0]);
            acc0 = vmlaq_f32(acc0, row1, _ker[1]);
            acc0 = vmlaq_f32(acc0, row2, _ker[2]);
            acc0 = vmlaq_f32(acc0, row3, _ker[3]);
            acc0 = vmlaq_f32(acc0, row4, _ker[4]);
            acc1 = vmulq_f32(row1, _ker[0]);
            acc1 = vmlaq_f32(acc1, row2, _ker[1]);
            acc1 = vmlaq_f32(acc1, row3, _ker[2]);
            acc1 = vmlaq_f32(acc1, row4, _ker[3]);
            acc1 = vmlaq_f32(acc1, row5, _ker[4]);
            acc0 = vpaddq_f32(acc0, acc1);
            float32x2_t sum =
                vpadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
            vst1_lane_f32(output_ptr0 + w, sum, 0);
            vst1_lane_f32(output_ptr1 + w, sum, 1);

            row0 = vextq_f32(zero, row0, 3);
            row1 = vextq_f32(zero, row1, 3);
            row2 = vextq_f32(zero, row2, 3);
            row3 = vextq_f32(zero, row3, 3);
            row4 = vextq_f32(zero, row4, 3);
            row5 = vextq_f32(zero, row5, 3);
          }
        }
        output_ptr0 += valid_w_start;
        output_ptr1 += valid_w_start;
      }
        // valid
// #if __aarch64__
#if 0
      float32x4_t _q14, _q15;
      for (int loop = 0; loop = output_w_tiles; ++loop) {
        float32x4_t _q7 = vld1q_f32(input_ptr0);
        float32x4_t _q8 = vld1q_f32(input_ptr0 + 4);
        float32x4_t _q9 = vld1q_f32(input_ptr1);
        float32x4_t _q10 = vld1q_f32(input_ptr1 + 4);
        float32x4_t _q11 = vld1q_f32(input_ptr2);
        float32x4_t _q12 = vld1q_f32(input_ptr2 + 4);

        _q14 = vmulq_lane_f32(_q7, vget_low_f32(_ker[5]), 0);
        float32x4_t _q13 = vextq_f32(_q7, _q8, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[0]), 0);
        _q13 = vextq_f32(_q7, _q8, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[0]), 1);
        _q13 = vextq_f32(_q7, _q8, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[0]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q8, vget_high_f32(_ker[0]), 1);

        _q14 = vmlaq_lane_f32(_q14, _q9, vget_low_f32(_ker[5]), 1);
        _q15 = vmulq_lane_f32(_q9, vget_low_f32(_ker[5]), 0);
        _q13 = vextq_f32(_q9, _q10, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[1]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[0]), 0);
        _q13 = vextq_f32(_q9, _q10, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[1]), 1);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[0]), 1);
        _q13 = vextq_f32(_q9, _q10, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[1]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_high_f32(_ker[0]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q10, vget_high_f32(_ker[1]), 1);
        _q15 = vmlaq_lane_f32(_q15, _q10, vget_high_f32(_ker[0]), 1);

        _q14 = vmlaq_lane_f32(_q14, _q11, vget_high_f32(_ker[5]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q11, vget_low_f32(_ker[5]), 1);
        _q13 = vextq_f32(_q11, _q12, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[2]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[1]), 0);
        _q13 = vextq_f32(_q11, _q12, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[2]), 1);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[1]), 1);
        _q13 = vextq_f32(_q11, _q12, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[2]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_high_f32(_ker[1]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q12, vget_high_f32(_ker[2]), 1);
        _q15 = vmlaq_lane_f32(_q15, _q12, vget_high_f32(_ker[1]), 1);

        _q7 = vld1q_f32(input_ptr3);
        _q8 = vld1q_f32(input_ptr3 + 4);
        _q9 = vld1q_f32(input_ptr4);
        _q10 = vld1q_f32(input_ptr4 + 4);
        _q11 = vld1q_f32(input_ptr5);
        _q12 = vld1q_f32(input_ptr5 + 4);

        _q14 = vmlaq_lane_f32(_q14, _q7, vget_high_f32(_ker[5]), 1);
        _q15 = vmlaq_lane_f32(_q15, _q7, vget_high_f32(_ker[5]), 0);
        _q13 = vextq_f32(_q7, _q8, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[3]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[2]), 0);
        _q13 = vextq_f32(_q7, _q8, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[3]), 1);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[2]), 1);
        _q13 = vextq_f32(_q7, _q8, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[3]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_high_f32(_ker[2]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q8, vget_high_f32(_ker[3]), 1);
        _q15 = vmlaq_lane_f32(_q15, _q8, vget_high_f32(_ker[2]), 1);

        _q14 = vmlaq_lane_f32(_q14, _q9, vget_low_f32(_ker[6]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q9, vget_high_f32(_ker[5]), 1);
        _q13 = vextq_f32(_q9, _q10, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[4]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[3]), 0);
        _q13 = vextq_f32(_q9, _q10, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[4]), 1);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[3]), 1);
        _q13 = vextq_f32(_q9, _q10, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[4]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_high_f32(_ker[3]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q10, vget_high_f32(_ker[4]), 1);
        _q15 = vmlaq_lane_f32(_q15, _q10, vget_high_f32(_ker[3]), 1);

        _q15 = vmlaq_lane_f32(_q15, _q11, vget_low_f32(_ker[6]), 0);
        _q13 = vextq_f32(_q11, _q12, 1);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[4]), 0);
        _q13 = vextq_f32(_q11, _q12, 2);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[4]), 1);
        _q13 = vextq_f32(_q11, _q12, 3);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_high_f32(_ker[4]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q12, vget_high_f32(_ker[4]), 1);

        vst1q_f32(output_ptr0, _q14);
        vst1q_f32(output_ptr1, _q15);

        input_ptr0 += 4;
        input_ptr1 += 4;
        input_ptr2 += 4;
        input_ptr3 += 4;
        input_ptr4 += 4;
        input_ptr5 += 4;
        output_ptr0 += 4;
        output_ptr1 += 4;
      }
      // remain w
      if (output_w_remain > 0) {
        float32x4_t _q7 = vld1q_f32(input_ptr0);
        float32x4_t _q8 = vld1q_f32(input_ptr0 + 4);
        float32x4_t _q9 = vld1q_f32(input_ptr1);
        float32x4_t _q10 = vld1q_f32(input_ptr1 + 4);
        float32x4_t _q11 = vld1q_f32(input_ptr2);
        float32x4_t _q12 = vld1q_f32(input_ptr2 + 4);

        _q14 = vmulq_lane_f32(_q7, vget_low_f32(_ker[5]), 0);
        float32x4_t _q13 = vextq_f32(_q7, _q8, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[0]), 0);
        _q13 = vextq_f32(_q7, _q8, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[0]), 1);
        _q13 = vextq_f32(_q7, _q8, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[0]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q8, vget_high_f32(_ker[0]), 1);

        _q14 = vmlaq_lane_f32(_q14, _q9, vget_low_f32(_ker[5]), 1);
        _q15 = vmulq_lane_f32(_q9, vget_low_f32(_ker[5]), 0);
        _q13 = vextq_f32(_q9, _q10, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[1]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[0]), 0);
        _q13 = vextq_f32(_q9, _q10, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[1]), 1);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[0]), 1);
        _q13 = vextq_f32(_q9, _q10, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[1]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_high_f32(_ker[0]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q10, vget_high_f32(_ker[1]), 1);
        _q15 = vmlaq_lane_f32(_q15, _q10, vget_high_f32(_ker[0]), 1);

        _q14 = vmlaq_lane_f32(_q14, _q11, vget_high_f32(_ker[5]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q11, vget_low_f32(_ker[5]), 1);
        _q13 = vextq_f32(_q11, _q12, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[2]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[1]), 0);
        _q13 = vextq_f32(_q11, _q12, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[2]), 1);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[1]), 1);
        _q13 = vextq_f32(_q11, _q12, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[2]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_high_f32(_ker[1]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q12, vget_high_f32(_ker[2]), 1);
        _q15 = vmlaq_lane_f32(_q15, _q12, vget_high_f32(_ker[1]), 1);

        _q7 = vld1q_f32(input_ptr3);
        _q8 = vld1q_f32(input_ptr3 + 4);
        _q9 = vld1q_f32(input_ptr4);
        _q10 = vld1q_f32(input_ptr4 + 4);
        _q11 = vld1q_f32(input_ptr5);
        _q12 = vld1q_f32(input_ptr5 + 4);

        _q14 = vmlaq_lane_f32(_q14, _q7, vget_high_f32(_ker[5]), 1);
        _q15 = vmlaq_lane_f32(_q15, _q7, vget_high_f32(_ker[5]), 0);
        _q13 = vextq_f32(_q7, _q8, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[3]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[2]), 0);
        _q13 = vextq_f32(_q7, _q8, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[3]), 1);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[2]), 1);
        _q13 = vextq_f32(_q7, _q8, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[3]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_high_f32(_ker[2]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q8, vget_high_f32(_ker[3]), 1);
        _q15 = vmlaq_lane_f32(_q15, _q8, vget_high_f32(_ker[2]), 1);

        _q14 = vmlaq_lane_f32(_q14, _q9, vget_low_f32(_ker[6]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q9, vget_high_f32(_ker[5]), 1);
        _q13 = vextq_f32(_q9, _q10, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[4]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[3]), 0);
        _q13 = vextq_f32(_q9, _q10, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[4]), 1);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[3]), 1);
        _q13 = vextq_f32(_q9, _q10, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[4]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_high_f32(_ker[3]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q10, vget_high_f32(_ker[4]), 1);
        _q15 = vmlaq_lane_f32(_q15, _q10, vget_high_f32(_ker[3]), 1);

        _q15 = vmlaq_lane_f32(_q15, _q11, vget_low_f32(_ker[6]), 0);
        _q13 = vextq_f32(_q11, _q12, 1);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[4]), 0);
        _q13 = vextq_f32(_q11, _q12, 2);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_low_f32(_ker[4]), 1);
        _q13 = vextq_f32(_q11, _q12, 3);
        _q15 = vmlaq_lane_f32(_q15, _q13, vget_high_f32(_ker[4]), 0);
        _q15 = vmlaq_lane_f32(_q15, _q12, vget_high_f32(_ker[4]), 1);

        switch (output_w_remain) {
          case 3:
            vst1q_lane_f32(output_ptr0 + 2, _q14, 2);
            vst1q_lane_f32(output_ptr1 + 2, _q15, 2);
          case 2:
            vst1_f32(output_ptr0, vget_low_f32(_q14));
            vst1_f32(output_ptr1, vget_low_f32(_q15));
            break;
          case 1:
            vst1q_lane_f32(output_ptr0, _q14, 0);
            vst1q_lane_f32(output_ptr1, _q15, 0);
            break;
        }
        input_ptr0 += output_w_remain;
        input_ptr1 += output_w_remain;
        input_ptr2 += output_w_remain;
        input_ptr3 += output_w_remain;
        input_ptr4 += output_w_remain;
        input_ptr5 += output_w_remain;
        output_ptr0 += output_w_remain;
        output_ptr1 += output_w_remain;
      }
#else
      int loop = output_w_tiles;
      asm volatile(
          "cmp        %[loop], #0                     \n"
          "ble        start_remain_%=                 \n"
          "mov        r0, #16                         \n"
          "loop_2h4w_%=:                              \n"
          "vld1.32    {d14-d17}, [%[input_ptr0]], r0  \n"
          "vld1.32    {d18-d21}, [%[input_ptr1]], r0  \n"
          "vld1.32    {d22-d25}, [%[input_ptr2]], r0  \n"
          "vmul.f32   q14, q7, %e[ker0][0]            \n"
          "vext.32    q13, q7, q8, #1                 \n"
          "vmla.f32   q14, q13, %e[kr0][0]            \n"
          "vext.32    q13, q7, q8, #2                 \n"
          "vmla.f32   q14, q13, %e[kr0][1]            \n"
          "vext.32    q13, q7, q8, #3                 \n"
          "vmla.f32   q14, q13, %f[kr0][0]            \n"
          "vmla.f32   q14, q8, %f[kr0][1]             \n"

          "vmla.f32   q14, q9, %e[ker0][1]            \n"
          "vmul.f32   q15, q9, %e[ker0][0]            \n"
          "vext.32    q13, q9, q10, #1                \n"
          "vmla.f32   q14, q13, %e[kr1][0]            \n"
          "vmla.f32   q15, q13, %e[kr0][0]            \n"
          "vext.32    q13, q9, q10, #2                \n"
          "vmla.f32   q14, q13, %e[kr1][1]            \n"
          "vmla.f32   q15, q13, %e[kr0][1]            \n"
          "vext.32    q13, q9, q10, #3                \n"
          "vmla.f32   q14, q13, %f[kr1][0]            \n"
          "vmla.f32   q15, q13, %f[kr0][0]            \n"
          "vmla.f32   q14, q10, %f[kr1][1]            \n"
          "vmla.f32   q15, q10, %f[kr0][1]            \n"

          "vmla.f32   q14, q11, %f[ker0][0]           \n"
          "vmla.f32   q15, q11, %e[ker0][1]           \n"
          "vext.32    q13, q11, q12, #1               \n"
          "vmla.f32   q14, q13, %e[kr2][0]            \n"
          "vmla.f32   q15, q13, %e[kr1][0]            \n"
          "vext.32    q13, q11, q12, #2               \n"
          "vmla.f32   q14, q13, %e[kr2][1]            \n"
          "vmla.f32   q15, q13, %e[kr1][1]            \n"
          "vext.32    q13, q11, q12, #3               \n"
          "vmla.f32   q14, q13, %f[kr2][0]            \n"
          "vmla.f32   q15, q13, %f[kr1][0]            \n"
          "vmla.f32   q14, q12, %f[kr2][1]            \n"
          "vmla.f32   q15, q12, %f[kr1][1]            \n"

          "vld1.32    {d14-d17}, [%[input_ptr3]], r0  \n"
          "vld1.32    {d18-d21}, [%[input_ptr4]], r0  \n"
          "vld1.32    {d22-d25}, [%[input_ptr5]], r0  \n"
          "vmla.f32   q14, q7, %f[ker0][1]            \n"
          "vmla.f32   q15, q7, %f[ker0][0]            \n"
          "vext.32    q13, q7, q8, #1                 \n"
          "vmla.f32   q14, q13, %e[kr3][0]            \n"
          "vmla.f32   q15, q13, %e[kr2][0]            \n"
          "vext.32    q13, q7, q8, #2                 \n"
          "vmla.f32   q14, q13, %e[kr3][1]            \n"
          "vmla.f32   q15, q13, %e[kr2][1]            \n"
          "vext.32    q13, q7, q8, #3                 \n"
          "vmla.f32   q14, q13, %f[kr3][0]            \n"
          "vmla.f32   q15, q13, %f[kr2][0]            \n"
          "vmla.f32   q14, q8, %f[kr3][1]             \n"
          "vmla.f32   q15, q8, %f[kr2][1]             \n"

          "vmla.f32   q14, q9, %e[ker1][0]            \n"
          "vmla.f32   q15, q9, %f[ker0][1]            \n"
          "vext.32    q13, q9, q10, #1                \n"
          "vmla.f32   q14, q13, %e[kr4][0]            \n"
          "vmla.f32   q15, q13, %e[kr3][0]            \n"
          "vext.32    q13, q9, q10, #2                \n"
          "vmla.f32   q14, q13, %e[kr4][1]            \n"
          "vmla.f32   q15, q13, %e[kr3][1]            \n"
          "vext.32    q13, q9, q10, #3                \n"
          "vmla.f32   q14, q13, %f[kr4][0]            \n"
          "vmla.f32   q15, q13, %f[kr3][0]            \n"
          "vmla.f32   q14, q10, %f[kr4][1]            \n"
          "vmla.f32   q15, q10, %f[kr3][1]            \n"

          "vmla.f32   q15, q11, %e[ker1][0]           \n"
          "vext.32    q13, q11, q12, #1               \n"
          "vmla.f32   q15, q13, %e[kr4][0]            \n"
          "vext.32    q13, q11, q12, #2               \n"
          "vmla.f32   q15, q13, %e[kr4][1]            \n"
          "vext.32    q13, q11, q12, #3               \n"
          "vmla.f32   q15, q13, %f[kr4][0]            \n"
          "vmla.f32   q15, q12, %f[kr4][1]            \n"
          // restore output
          "vst1.32    {q14}, [%[output_ptr0]]!        \n"
          "vst1.32    {q15}, [%[output_ptr1]]!        \n"
          "subs       %[loop], #1                     \n"
          "bne        loop_2h4w_%=                    \n"

          "start_remain_%=:                           \n"
          "cmp        %[remain], #0                   \n"
          "ble        end_%=                          \n"
          "mov        r0, %[remain], lsl #2           \n"
          "vld1.32    {d14-d17}, [%[input_ptr0]], r0  \n"
          "vld1.32    {d18-d21}, [%[input_ptr1]], r0  \n"
          "vld1.32    {d22-d25}, [%[input_ptr2]], r0  \n"
          "vmul.f32   q14, q7, %e[ker0][0]            \n"
          "vext.32    q13, q7, q8, #1                 \n"
          "vmla.f32   q14, q13, %e[kr0][0]            \n"
          "vext.32    q13, q7, q8, #2                 \n"
          "vmla.f32   q14, q13, %e[kr0][1]            \n"
          "vext.32    q13, q7, q8, #3                 \n"
          "vmla.f32   q14, q13, %f[kr0][0]            \n"
          "vmla.f32   q14, q8, %f[kr0][1]             \n"

          "vmla.f32   q14, q9, %e[ker0][1]            \n"
          "vmul.f32   q15, q9, %e[ker0][0]            \n"
          "vext.32    q13, q9, q10, #1                \n"
          "vmla.f32   q14, q13, %e[kr1][0]            \n"
          "vmla.f32   q15, q13, %e[kr0][0]            \n"
          "vext.32    q13, q9, q10, #2                \n"
          "vmla.f32   q14, q13, %e[kr1][1]            \n"
          "vmla.f32   q15, q13, %e[kr0][1]            \n"
          "vext.32    q13, q9, q10, #3                \n"
          "vmla.f32   q14, q13, %f[kr1][0]            \n"
          "vmla.f32   q15, q13, %f[kr0][0]            \n"
          "vmla.f32   q14, q10, %f[kr1][1]            \n"
          "vmla.f32   q15, q10, %f[kr0][1]            \n"

          "vmla.f32   q14, q11, %f[ker0][0]           \n"
          "vmla.f32   q15, q11, %e[ker0][1]           \n"
          "vext.32    q13, q11, q12, #1               \n"
          "vmla.f32   q14, q13, %e[kr2][0]            \n"
          "vmla.f32   q15, q13, %e[kr1][0]            \n"
          "vext.32    q13, q11, q12, #2               \n"
          "vmla.f32   q14, q13, %e[kr2][1]            \n"
          "vmla.f32   q15, q13, %e[kr1][1]            \n"
          "vext.32    q13, q11, q12, #3               \n"
          "vmla.f32   q14, q13, %f[kr2][0]            \n"
          "vmla.f32   q15, q13, %f[kr1][0]            \n"
          "vmla.f32   q14, q12, %f[kr2][1]            \n"
          "vmla.f32   q15, q12, %f[kr1][1]            \n"

          "vld1.32    {d14-d17}, [%[input_ptr3]], r0  \n"
          "vld1.32    {d18-d21}, [%[input_ptr4]], r0  \n"
          "vld1.32    {d22-d25}, [%[input_ptr5]], r0  \n"
          "vmla.f32   q14, q7, %f[ker0][1]            \n"
          "vmla.f32   q15, q7, %f[ker0][0]            \n"
          "vext.32    q13, q7, q8, #1                 \n"
          "vmla.f32   q14, q13, %e[kr3][0]            \n"
          "vmla.f32   q15, q13, %e[kr2][0]            \n"
          "vext.32    q13, q7, q8, #2                 \n"
          "vmla.f32   q14, q13, %e[kr3][1]            \n"
          "vmla.f32   q15, q13, %e[kr2][1]            \n"
          "vext.32    q13, q7, q8, #3                 \n"
          "vmla.f32   q14, q13, %f[kr3][0]            \n"
          "vmla.f32   q15, q13, %f[kr2][0]            \n"
          "vmla.f32   q14, q8, %f[kr3][1]             \n"
          "vmla.f32   q15, q8, %f[kr2][1]             \n"

          "vmla.f32   q14, q9, %e[ker1][0]            \n"
          "vmla.f32   q15, q9, %f[ker0][1]            \n"
          "vext.32    q13, q9, q10, #1                \n"
          "vmla.f32   q14, q13, %e[kr4][0]            \n"
          "vmla.f32   q15, q13, %e[kr3][0]            \n"
          "vext.32    q13, q9, q10, #2                \n"
          "vmla.f32   q14, q13, %e[kr4][1]            \n"
          "vmla.f32   q15, q13, %e[kr3][1]            \n"
          "vext.32    q13, q9, q10, #3                \n"
          "vmla.f32   q14, q13, %f[kr4][0]            \n"
          "vmla.f32   q15, q13, %f[kr3][0]            \n"
          "vmla.f32   q14, q10, %f[kr4][1]            \n"
          "vmla.f32   q15, q10, %f[kr3][1]            \n"

          "vmla.f32   q15, q11, %e[ker1][0]           \n"
          "vext.32    q13, q11, q12, #1               \n"
          "vmla.f32   q15, q13, %e[kr4][0]            \n"
          "vext.32    q13, q11, q12, #2               \n"
          "vmla.f32   q15, q13, %e[kr4][1]            \n"
          "vext.32    q13, q11, q12, #3               \n"
          "vmla.f32   q15, q13, %f[kr4][0]            \n"
          "vmla.f32   q15, q12, %f[kr4][1]            \n"

          "cmp        %[remain], #2                   \n"
          "blt        store_2h1w_%=                   \n"
          "vst1.32    {d28}, [%[output_ptr0]]!        \n"
          "vst1.32    {d30}, [%[output_ptr1]]!        \n"
          "cmp        %[remain], #3                   \n"
          "blt        end_%=                          \n"
          "vst1.32    {d29[0]}, [%[output_ptr0]]!     \n"
          "vst1.32    {d31[0]}, [%[output_ptr1]]!     \n"
          "b          end_%=                          \n"

          "store_2h1w_%=:                             \n"
          "vst1.32    {d28[0]}, [%[output_ptr0]]!     \n"
          "vst1.32    {d30[0]}, [%[output_ptr1]]!     \n"
          "end_%=:                                    \n"
          : [input_ptr0] "+r"(input_ptr0), [input_ptr1] "+r"(input_ptr1),
            [input_ptr2] "+r"(input_ptr2), [input_ptr3] "+r"(input_ptr3),
            [input_ptr4] "+r"(input_ptr4), [input_ptr5] "+r"(input_ptr5),
            [output_ptr0] "+r"(output_ptr0), [output_ptr1] "+r"(output_ptr1),
            [loop] "+r"(loop)
          : [remain] "r"(output_w_remain), [kr0] "w"(_ker[0]),
            [kr1] "w"(_ker[1]), [kr2] "w"(_ker[2]), [kr3] "w"(_ker[3]),
            [kr4] "w"(_ker[4]), [ker0] "w"(_ker[5]), [ker1] "w"(_ker[6])
          : "cc", "memory", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14",
            "q15", "r0");
#endif  // __aarch64__
      // pad right
      if (padding_w) {
        float32x4_t row0 = vld1q_f32(input_ptr0);
        float32x4_t row1 = vld1q_f32(input_ptr1);
        float32x4_t row2 = vld1q_f32(input_ptr2);
        float32x4_t row3 = vld1q_f32(input_ptr3);
        float32x4_t row4 = vld1q_f32(input_ptr4);
        float32x4_t row5 = vld1q_f32(input_ptr5);
        float32x4_t zero = vdupq_n_f32(0.f);
        float32x4_t acc0, acc1;
        for (int w = valid_w_end; w < output_w; ++w) {
          int padding = w + 5 - (padding_w + input_w);
          if (padding >= 5) {
            *output_ptr0 = 0.f;
            *output_ptr1 = 0.f;
          } else {
            int iw = w - valid_w_end;
            float sum0 = input_ptr0[iw] * filter_ptr0[0] +
                         input_ptr1[iw] * filter_ptr1[0] +
                         input_ptr2[iw] * filter_ptr2[0] +
                         input_ptr3[iw] * filter_ptr3[0] +
                         input_ptr4[iw] * filter_ptr4[0];
            float sum1 = input_ptr1[iw] * filter_ptr0[0] +
                         input_ptr2[iw] * filter_ptr1[0] +
                         input_ptr3[iw] * filter_ptr2[0] +
                         input_ptr4[iw] * filter_ptr3[0] +
                         input_ptr5[iw] * filter_ptr4[0];
            row0 = vextq_f32(row0, zero, 1);
            row1 = vextq_f32(row1, zero, 1);
            row2 = vextq_f32(row2, zero, 1);
            row3 = vextq_f32(row3, zero, 1);
            row4 = vextq_f32(row4, zero, 1);
            row5 = vextq_f32(row5, zero, 1);
            acc0 = vmulq_f32(row0, _ker[0]);
            acc0 = vmlaq_f32(acc0, row1, _ker[1]);
            acc0 = vmlaq_f32(acc0, row2, _ker[2]);
            acc0 = vmlaq_f32(acc0, row3, _ker[3]);
            acc0 = vmlaq_f32(acc0, row4, _ker[4]);
            acc1 = vmulq_f32(row1, _ker[0]);
            acc1 = vmlaq_f32(acc1, row2, _ker[1]);
            acc1 = vmlaq_f32(acc1, row3, _ker[2]);
            acc1 = vmlaq_f32(acc1, row4, _ker[3]);
            acc1 = vmlaq_f32(acc1, row5, _ker[4]);
            acc0 = vpaddq_f32(acc0, acc1);
            float32x2_t sum =
                vpadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
            sum0 += vget_lane_f32(sum, 0);
            sum1 += vget_lane_f32(sum, 1);
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
      const float *input_ptr0 = input_ptr + (start_h - padding_h) * input_w;
      const float *input_ptr1 = input_ptr0 + input_w;
      const float *input_ptr2 = input_ptr1 + input_w;
      const float *input_ptr3 = input_ptr2 + input_w;
      const float *input_ptr4 = input_ptr3 + input_w;
      float *output_ptr0 = output_ptr + start_h * output_w;
      // pad left
      if (padding_w) {
        float32x4_t row0 = vld1q_f32(input_ptr0);
        float32x4_t row1 = vld1q_f32(input_ptr1);
        float32x4_t row2 = vld1q_f32(input_ptr2);
        float32x4_t row3 = vld1q_f32(input_ptr3);
        float32x4_t row4 = vld1q_f32(input_ptr4);
        float32x4_t zero = vdupq_n_f32(0.f);
        float32x4_t acc;
        for (int w = valid_w_start - 1; w >= 0; --w) {
          int padding = padding_w - w;
          if (padding >= 5) {
            output_ptr0[w] = 0.f;
          } else {
            acc = vmulq_f32(row0, _ker[0]);
            acc = vmlaq_f32(acc, row1, _ker[1]);
            acc = vmlaq_f32(acc, row2, _ker[2]);
            acc = vmlaq_f32(acc, row3, _ker[3]);
            acc = vmlaq_f32(acc, row4, _ker[4]);
            float32x2_t sum = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
            sum = vpadd_f32(sum, sum);
            vst1_lane_f32(output_ptr0 + w, sum, 0);

            row0 = vextq_f32(zero, row0, 3);
            row1 = vextq_f32(zero, row1, 3);
            row2 = vextq_f32(zero, row2, 3);
            row3 = vextq_f32(zero, row3, 3);
            row4 = vextq_f32(zero, row4, 3);
          }
        }
        output_ptr0 += valid_w_start;
      }
        // valid
// #if __aarch64__
#if 0
      float32x4_t _q14;
      for (int loop = 0; loop = output_w_tiles; ++loop) {
        float32x4_t _q7 = vld1q_f32(input_ptr0);
        float32x4_t _q8 = vld1q_f32(input_ptr0 + 4);
        float32x4_t _q9 = vld1q_f32(input_ptr1);
        float32x4_t _q10 = vld1q_f32(input_ptr1 + 4);
        float32x4_t _q11 = vld1q_f32(input_ptr2);
        float32x4_t _q12 = vld1q_f32(input_ptr2 + 4);

        _q14 = vmulq_lane_f32(_q7, vget_low_f32(_ker[5]), 0);
        float32x4_t _q13 = vextq_f32(_q7, _q8, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[0]), 0);
        _q13 = vextq_f32(_q7, _q8, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[0]), 1);
        _q13 = vextq_f32(_q7, _q8, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[0]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q8, vget_high_f32(_ker[0]), 1);

        _q14 = vmlaq_lane_f32(_q14, _q9, vget_low_f32(_ker[5]), 1);
        _q13 = vextq_f32(_q9, _q10, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[1]), 0);
        _q13 = vextq_f32(_q9, _q10, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[1]), 1);
        _q13 = vextq_f32(_q9, _q10, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[1]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q10, vget_high_f32(_ker[1]), 1);

        _q14 = vmlaq_lane_f32(_q14, _q11, vget_high_f32(_ker[5]), 0);
        _q13 = vextq_f32(_q11, _q12, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[2]), 0);
        _q13 = vextq_f32(_q11, _q12, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[2]), 1);
        _q13 = vextq_f32(_q11, _q12, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[2]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q12, vget_high_f32(_ker[2]), 1);

        _q7 = vld1q_f32(input_ptr3);
        _q8 = vld1q_f32(input_ptr3 + 4);
        _q9 = vld1q_f32(input_ptr4);
        _q10 = vld1q_f32(input_ptr4 + 4);

        _q14 = vmlaq_lane_f32(_q14, _q7, vget_high_f32(_ker[5]), 1);
        _q13 = vextq_f32(_q7, _q8, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[3]), 0);
        _q13 = vextq_f32(_q7, _q8, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[3]), 1);
        _q13 = vextq_f32(_q7, _q8, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[3]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q8, vget_high_f32(_ker[3]), 1);

        _q14 = vmlaq_lane_f32(_q14, _q9, vget_low_f32(_ker[6]), 0);
        _q13 = vextq_f32(_q9, _q10, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[4]), 0);
        _q13 = vextq_f32(_q9, _q10, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[4]), 1);
        _q13 = vextq_f32(_q9, _q10, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[4]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q10, vget_high_f32(_ker[4]), 1);

        vst1q_f32(output_ptr0, _q14);

        input_ptr0 += 4;
        input_ptr1 += 4;
        input_ptr2 += 4;
        input_ptr3 += 4;
        input_ptr4 += 4;
        output_ptr0 += 4;
      }
      // remain w
      if (output_w_remain > 0) {
        float32x4_t _q7 = vld1q_f32(input_ptr0);
        float32x4_t _q8 = vld1q_f32(input_ptr0 + 4);
        float32x4_t _q9 = vld1q_f32(input_ptr1);
        float32x4_t _q10 = vld1q_f32(input_ptr1 + 4);
        float32x4_t _q11 = vld1q_f32(input_ptr2);
        float32x4_t _q12 = vld1q_f32(input_ptr2 + 4);

        _q14 = vmulq_lane_f32(_q7, vget_low_f32(_ker[5]), 0);
        float32x4_t _q13 = vextq_f32(_q7, _q8, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[0]), 0);
        _q13 = vextq_f32(_q7, _q8, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[0]), 1);
        _q13 = vextq_f32(_q7, _q8, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[0]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q8, vget_high_f32(_ker[0]), 1);

        _q14 = vmlaq_lane_f32(_q14, _q9, vget_low_f32(_ker[5]), 1);
        _q13 = vextq_f32(_q9, _q10, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[1]), 0);
        _q13 = vextq_f32(_q9, _q10, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[1]), 1);
        _q13 = vextq_f32(_q9, _q10, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[1]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q10, vget_high_f32(_ker[1]), 1);

        _q14 = vmlaq_lane_f32(_q14, _q11, vget_high_f32(_ker[5]), 0);
        _q13 = vextq_f32(_q11, _q12, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[2]), 0);
        _q13 = vextq_f32(_q11, _q12, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[2]), 1);
        _q13 = vextq_f32(_q11, _q12, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[2]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q12, vget_high_f32(_ker[2]), 1);

        _q7 = vld1q_f32(input_ptr3);
        _q8 = vld1q_f32(input_ptr3 + 4);
        _q9 = vld1q_f32(input_ptr4);
        _q10 = vld1q_f32(input_ptr4 + 4);

        _q14 = vmlaq_lane_f32(_q14, _q7, vget_high_f32(_ker[5]), 1);
        _q13 = vextq_f32(_q7, _q8, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[3]), 0);
        _q13 = vextq_f32(_q7, _q8, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[3]), 1);
        _q13 = vextq_f32(_q7, _q8, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[3]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q8, vget_high_f32(_ker[3]), 1);

        _q14 = vmlaq_lane_f32(_q14, _q9, vget_low_f32(_ker[6]), 0);
        _q13 = vextq_f32(_q9, _q10, 1);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[4]), 0);
        _q13 = vextq_f32(_q9, _q10, 2);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_low_f32(_ker[4]), 1);
        _q13 = vextq_f32(_q9, _q10, 3);
        _q14 = vmlaq_lane_f32(_q14, _q13, vget_high_f32(_ker[4]), 0);
        _q14 = vmlaq_lane_f32(_q14, _q10, vget_high_f32(_ker[4]), 1);

        switch (output_w_remain) {
          case 3:
            vst1q_lane_f32(output_ptr0 + 2, _q14, 2);
          case 2:
            vst1_f32(output_ptr0, vget_low_f32(_q14));
            break;
          case 1:
            vst1q_lane_f32(output_ptr0, _q14, 0);
            break;
        }

        input_ptr0 += output_w_remain;
        input_ptr1 += output_w_remain;
        input_ptr2 += output_w_remain;
        input_ptr3 += output_w_remain;
        input_ptr4 += output_w_remain;
        output_ptr0 += output_w_remain;
      }
#else
      int loop = output_w_tiles;
      asm volatile(
          "cmp        %[loop], #0                     \n"
          "ble        start_remain_%=                 \n"
          "mov        r0, #16                         \n"
          "loop_1h4w_%=:                              \n"
          "vld1.32    {d14-d17}, [%[input_ptr0]], r0  \n"
          "vld1.32    {d18-d21}, [%[input_ptr1]], r0  \n"
          "vld1.32    {d22-d25}, [%[input_ptr2]], r0  \n"
          "vmul.f32   q14, q7, %e[ker0][0]            \n"
          "vext.32    q13, q7, q8, #1                 \n"
          "vmla.f32   q14, q13, %e[kr0][0]            \n"
          "vext.32    q13, q7, q8, #2                 \n"
          "vmla.f32   q14, q13, %e[kr0][1]            \n"
          "vext.32    q13, q7, q8, #3                 \n"
          "vmla.f32   q14, q13, %f[kr0][0]            \n"
          "vmla.f32   q14, q8, %f[kr0][1]             \n"

          "vmla.f32   q14, q9, %e[ker0][1]            \n"
          "vext.32    q13, q9, q10, #1                \n"
          "vmla.f32   q14, q13, %e[kr1][0]            \n"
          "vext.32    q13, q9, q10, #2                \n"
          "vmla.f32   q14, q13, %e[kr1][1]            \n"
          "vext.32    q13, q9, q10, #3                \n"
          "vmla.f32   q14, q13, %f[kr1][0]            \n"
          "vmla.f32   q14, q10, %f[kr1][1]            \n"

          "vmla.f32   q14, q11, %f[ker0][0]           \n"
          "vext.32    q13, q11, q12, #1               \n"
          "vmla.f32   q14, q13, %e[kr2][0]            \n"
          "vext.32    q13, q11, q12, #2               \n"
          "vmla.f32   q14, q13, %e[kr2][1]            \n"
          "vext.32    q13, q11, q12, #3               \n"
          "vmla.f32   q14, q13, %f[kr2][0]            \n"
          "vmla.f32   q14, q12, %f[kr2][1]            \n"

          "vld1.32    {d14-d17}, [%[input_ptr3]], r0  \n"
          "vld1.32    {d18-d21}, [%[input_ptr4]], r0  \n"
          "vmla.f32   q14, q7, %f[ker0][1]            \n"
          "vext.32    q13, q7, q8, #1                 \n"
          "vmla.f32   q14, q13, %e[kr3][0]            \n"
          "vext.32    q13, q7, q8, #2                 \n"
          "vmla.f32   q14, q13, %e[kr3][1]            \n"
          "vext.32    q13, q7, q8, #3                 \n"
          "vmla.f32   q14, q13, %f[kr3][0]            \n"
          "vmla.f32   q14, q8, %f[kr3][1]             \n"

          "vmla.f32   q14, q9, %e[ker1][0]            \n"
          "vext.32    q13, q9, q10, #1                \n"
          "vmla.f32   q14, q13, %e[kr4][0]            \n"
          "vext.32    q13, q9, q10, #2                \n"
          "vmla.f32   q14, q13, %e[kr4][1]            \n"
          "vext.32    q13, q9, q10, #3                \n"
          "vmla.f32   q14, q13, %f[kr4][0]            \n"
          "vmla.f32   q14, q10, %f[kr4][1]            \n"

          // restore output
          "vst1.32    {q14}, [%[output_ptr0]]!        \n"
          "subs       %[loop], #1                     \n"
          "bne        loop_1h4w_%=                    \n"

          "start_remain_%=:                           \n"
          "cmp        %[remain], #0                   \n"
          "ble        end_%=                          \n"
          "mov        r0, %[remain], lsl #2           \n"
          "vld1.32    {d14-d17}, [%[input_ptr0]], r0  \n"
          "vld1.32    {d18-d21}, [%[input_ptr1]], r0  \n"
          "vld1.32    {d22-d25}, [%[input_ptr2]], r0  \n"
          "vmul.f32   q14, q7, %e[ker0][0]            \n"
          "vext.32    q13, q7, q8, #1                 \n"
          "vmla.f32   q14, q13, %e[kr0][0]            \n"
          "vext.32    q13, q7, q8, #2                 \n"
          "vmla.f32   q14, q13, %e[kr0][1]            \n"
          "vext.32    q13, q7, q8, #3                 \n"
          "vmla.f32   q14, q13, %f[kr0][0]            \n"
          "vmla.f32   q14, q8, %f[kr0][1]             \n"

          "vmla.f32   q14, q9, %e[ker0][1]            \n"
          "vext.32    q13, q9, q10, #1                \n"
          "vmla.f32   q14, q13, %e[kr1][0]            \n"
          "vext.32    q13, q9, q10, #2                \n"
          "vmla.f32   q14, q13, %e[kr1][1]            \n"
          "vext.32    q13, q9, q10, #3                \n"
          "vmla.f32   q14, q13, %f[kr1][0]            \n"
          "vmla.f32   q14, q10, %f[kr1][1]            \n"

          "vmla.f32   q14, q11, %f[ker0][0]           \n"
          "vext.32    q13, q11, q12, #1               \n"
          "vmla.f32   q14, q13, %e[kr2][0]            \n"
          "vext.32    q13, q11, q12, #2               \n"
          "vmla.f32   q14, q13, %e[kr2][1]            \n"
          "vext.32    q13, q11, q12, #3               \n"
          "vmla.f32   q14, q13, %f[kr2][0]            \n"
          "vmla.f32   q14, q12, %f[kr2][1]            \n"

          "vld1.32    {d14-d17}, [%[input_ptr3]], r0  \n"
          "vld1.32    {d18-d21}, [%[input_ptr4]], r0  \n"
          "vmla.f32   q14, q7, %f[ker0][1]            \n"
          "vext.32    q13, q7, q8, #1                 \n"
          "vmla.f32   q14, q13, %e[kr3][0]            \n"
          "vext.32    q13, q7, q8, #2                 \n"
          "vmla.f32   q14, q13, %e[kr3][1]            \n"
          "vext.32    q13, q7, q8, #3                 \n"
          "vmla.f32   q14, q13, %f[kr3][0]            \n"
          "vmla.f32   q14, q8, %f[kr3][1]             \n"

          "vmla.f32   q14, q9, %e[ker1][0]            \n"
          "vext.32    q13, q9, q10, #1                \n"
          "vmla.f32   q14, q13, %e[kr4][0]            \n"
          "vext.32    q13, q9, q10, #2                \n"
          "vmla.f32   q14, q13, %e[kr4][1]            \n"
          "vext.32    q13, q9, q10, #3                \n"
          "vmla.f32   q14, q13, %f[kr4][0]            \n"
          "vmla.f32   q14, q10, %f[kr4][1]            \n"

          "cmp        %[remain], #2                   \n"
          "blt        store_1h1w_%=                   \n"
          "vst1.32    {d28}, [%[output_ptr0]]!        \n"
          "cmp        %[remain], #3                   \n"
          "blt        end_%=                          \n"
          "vst1.32    {d29[0]}, [%[output_ptr0]]!     \n"
          "b          end_%=                          \n"

          "store_1h1w_%=:                             \n"
          "vst1.32    {d28[0]}, [%[output_ptr0]]!     \n"
          "end_%=:                                    \n"
          : [input_ptr0] "+r"(input_ptr0), [input_ptr1] "+r"(input_ptr1),
            [input_ptr2] "+r"(input_ptr2), [input_ptr3] "+r"(input_ptr3),
            [input_ptr4] "+r"(input_ptr4), [output_ptr0] "+r"(output_ptr0),
            [loop] "+r"(loop)
          : [remain] "r"(output_w_remain), [kr0] "w"(_ker[0]),
            [kr1] "w"(_ker[1]), [kr2] "w"(_ker[2]), [kr3] "w"(_ker[3]),
            [kr4] "w"(_ker[4]), [ker0] "w"(_ker[5]), [ker1] "w"(_ker[6])
          : "cc", "memory", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14",
            "q15", "r0");
#endif  // __aarch64__
      // pad right
      if (padding_w) {
        float32x4_t row0 = vld1q_f32(input_ptr0);
        float32x4_t row1 = vld1q_f32(input_ptr1);
        float32x4_t row2 = vld1q_f32(input_ptr2);
        float32x4_t row3 = vld1q_f32(input_ptr3);
        float32x4_t row4 = vld1q_f32(input_ptr4);
        float32x4_t zero = vdupq_n_f32(0.f);
        float32x4_t acc;
        for (int w = valid_w_end; w < output_w; ++w) {
          int padding = w + 5 - (padding_w + input_w);
          if (padding >= 5) {
            *output_ptr0 = 0.f;
          } else {
            int iw = w - valid_w_end;
            float sum0 = input_ptr0[iw] * filter_ptr0[0] +
                         input_ptr1[iw] * filter_ptr1[0] +
                         input_ptr2[iw] * filter_ptr2[0] +
                         input_ptr3[iw] * filter_ptr3[0] +
                         input_ptr4[iw] * filter_ptr4[0];
            row0 = vextq_f32(row0, zero, 1);
            row1 = vextq_f32(row1, zero, 1);
            row2 = vextq_f32(row2, zero, 1);
            row3 = vextq_f32(row3, zero, 1);
            row4 = vextq_f32(row4, zero, 1);
            acc = vmulq_f32(row0, _ker[0]);
            acc = vmlaq_f32(acc, row1, _ker[1]);
            acc = vmlaq_f32(acc, row2, _ker[2]);
            acc = vmlaq_f32(acc, row3, _ker[3]);
            acc = vmlaq_f32(acc, row4, _ker[4]);
            float32x2_t sum = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
            sum = vpadd_f32(sum, sum);
            sum0 += vget_lane_f32(sum, 0);
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
                                      output_ptr, _ker, _ker1);
    }
  }
}

template <>
void DepthwiseConv5x5S2<float, float>(const framework::Tensor &input,
                                      const framework::Tensor &filter,
                                      const std::vector<int> &paddings,
                                      framework::Tensor *output) {}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif  // __ARM_NEON__
