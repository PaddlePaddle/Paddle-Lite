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

#include "operators/math/depthwise_conv3x3.h"
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
inline void Depth3x3NormalRowLoadInput(const float *input, float32x4_t *y) {
  y[0] = vld1q_f32(input);
  y[2] = vld1q_f32(input + 4);
  y[1] = vextq_f32(y[0], y[2], 1);
  y[2] = vextq_f32(y[0], y[2], 2);
}

template <>
inline void Depth3x3NormalRowLoadInput<2>(const float *input, float32x4_t *y) {
  float32x4x2_t x = vld2q_f32(input);
  y[0] = x.val[0];
  y[1] = x.val[1];
  y[2] = vextq_f32(y[0], y[0], 1);
  y[2] = vsetq_lane_f32(input[8], y[2], 3);
}

#define DEPTHWISE_CONV3X3_NORMAL_BORDER(start, end)                      \
  for (int w = start; w < end; ++w) {                                    \
    const int w_in_start = -padding_w + w * Stride_w;                    \
    const int w_in_end = w_in_start + 3;                                 \
    const int w_start = w_in_start > 0 ? w_in_start : 0;                 \
    const int w_end = w_in_end < input_w ? w_in_end : input_w;           \
    float value = 0;                                                     \
    for (int h_in = h_start; h_in < h_end; ++h_in) {                     \
      for (int w_in = w_start; w_in < w_end; ++w_in) {                   \
        value += filter[(h_in - h_in_start) * 3 + (w_in - w_in_start)] * \
                 input[h_in * input_w + w_in];                           \
      }                                                                  \
    }                                                                    \
    output_ptr[w] = value;                                               \
  }

template <int Stride_h, int Stride_w>
inline void DepthwiseConv3x3NormalRow(const float *input, const float *filter,
                                      const int h_output, const int input_h,
                                      const int input_w, const int padding_h,
                                      const int padding_w, const int output_w,
                                      float *output, float32x4_t *ker) {
  const int h_in_start = -padding_h + h_output * Stride_h;
  const int h_in_end = h_in_start + 3;
  const int h_start = h_in_start > 0 ? h_in_start : 0;
  const int h_end = h_in_end < input_h ? h_in_end : input_h;

  int valid_w_start = (padding_w + Stride_w - 1) / Stride_w;
  int valid_w_end = (input_w + padding_w - 3) / Stride_w + 1;
  if (valid_w_end < valid_w_start) {
    valid_w_end = valid_w_start;
  }
  // const int valid_w_end = output_w - valid_w_start;
  float *output_ptr = output + h_output * output_w;
  // border left
  DEPTHWISE_CONV3X3_NORMAL_BORDER(0, valid_w_start)
  // middle
  int output_tiles = (valid_w_end - valid_w_start) >> 2;
  float32x4_t _sum, _x[3];
  // valid w
  for (int w = 0; w < output_tiles * 4; w += 4) {
    _sum = vdupq_n_f32(0.f);
    int output_offset = valid_w_start + w;
    int input_w_offset = output_offset * Stride_w - padding_w;
    for (int h_in = h_start; h_in < h_end; ++h_in) {
      int index = h_in - h_in_start;
      Depth3x3NormalRowLoadInput<Stride_w>(
          input + h_in * input_w + input_w_offset, _x);
      _sum = vmlaq_lane_f32(_sum, _x[0], vget_low_f32(ker[index]), 0);
      _sum = vmlaq_lane_f32(_sum, _x[1], vget_low_f32(ker[index]), 1);
      _sum = vmlaq_lane_f32(_sum, _x[2], vget_high_f32(ker[index]), 0);
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
      Depth3x3NormalRowLoadInput<Stride_w>(
          input + h_in * input_w + input_w_offset, _x);
      _sum = vmlaq_lane_f32(_sum, _x[0], vget_low_f32(ker[index]), 0);
      _sum = vmlaq_lane_f32(_sum, _x[1], vget_low_f32(ker[index]), 1);
      _sum = vmlaq_lane_f32(_sum, _x[2], vget_high_f32(ker[index]), 0);
    }
    switch (remain) {
      case 3:
        vst1q_lane_f32(output_ptr0 + 2, _sum, 2);
      case 2:
        vst1_f32(output_ptr0, vget_low_f32(_sum));
        break;
      case 1:
        vst1q_lane_f32(output_ptr0, _sum, 0);
        break;
    }
  }
  // border right
  DEPTHWISE_CONV3X3_NORMAL_BORDER(valid_w_end, output_w)
}

template <>
void DepthwiseConv3x3S1<float, float>(const framework::Tensor &input,
                                      const framework::Tensor &filter,
                                      const std::vector<int> &paddings,
                                      framework::Tensor *output) {
  const float *input_data = input.data<float>();
  const float *filter_data = filter.data<float>();
  float *out_data = output->mutable_data<float>();

  const int input_h = input.dims()[2];
  const int input_w = input.dims()[3];
  const int output_h = output->dims()[2];
  const int output_w = output->dims()[3];
  const int padding_h = paddings[0];
  const int padding_w = paddings[1];
  const int image_size = input_h * input_w;
  const int out_image_size = output_h * output_w;
  const int valid_h_start = padding_h;
  const int valid_h_end = output_h - valid_h_start;
  const int valid_h = valid_h_end - valid_h_start;
  const int valid_w_start = padding_w;
  const int valid_w_end = output_w - valid_w_start;
  const int valid_w = valid_w_end - valid_w_start;

  #pragma omp parallel for
  for (int g = 0; g < input.dims()[1]; ++g) {
    const float *input_ptr = input_data + g * image_size;
    const float *filter_ptr = filter_data + g * 9;
    float *output_ptr = out_data + g * out_image_size;

    const float *filter_ptr0 = filter_ptr;
    const float *filter_ptr1 = filter_ptr0 + 3;
    const float *filter_ptr2 = filter_ptr1 + 3;
    float32x4_t _ker[3];
    _ker[0] = vld1q_f32(filter_ptr0);
    _ker[1] = vld1q_f32(filter_ptr1);
    _ker[2] = vld1q_f32(filter_ptr2);

    // pad top
    for (int h = 0; h < valid_h_start; ++h) {
      DepthwiseConv3x3NormalRow<1, 1>(input_ptr, filter_ptr, h, input_h,
                                      input_w, padding_h, padding_w, output_w,
                                      output_ptr, _ker);
    }

    // output 2x6
    int output_w_tiles = valid_w / 6;
    int output_w_remain = valid_w - output_w_tiles * 6;
    for (int h = valid_h_start; h < valid_h_end - 1; h += 2) {
      const float *input_ptr0 = input_ptr + (h - padding_h) * input_w;
      const float *input_ptr1 = input_ptr0 + input_w;
      const float *input_ptr2 = input_ptr1 + input_w;
      const float *input_ptr3 = input_ptr2 + input_w;
      float *output_ptr0 = output_ptr + h * output_w;
      float *output_ptr1 = output_ptr0 + output_w;
      // pad left
      if (padding_w) {
        float32x4_t row0 = vld1q_f32(input_ptr0);
        float32x4_t row1 = vld1q_f32(input_ptr1);
        float32x4_t row2 = vld1q_f32(input_ptr2);
        float32x4_t row3 = vld1q_f32(input_ptr3);
        float32x4_t zero = vdupq_n_f32(0.f);
        row0 = vextq_f32(zero, row0, 3);
        row1 = vextq_f32(zero, row1, 3);
        row2 = vextq_f32(zero, row2, 3);
        row3 = vextq_f32(zero, row3, 3);
        float32x4_t acc0, acc1;
        for (int w = valid_w_start - 1; w >= 0; --w) {
          int padding = padding_w - w;
          if (padding >= 3) {
            output_ptr0[w] = 0.f;
            output_ptr1[w] = 0.f;
          } else {
            acc0 = vmulq_f32(row0, _ker[0]);
            acc0 = vmlaq_f32(acc0, row1, _ker[1]);
            acc0 = vmlaq_f32(acc0, row2, _ker[2]);
            acc0 = vextq_f32(acc0, acc0, 1);
            acc1 = vmulq_f32(row1, _ker[0]);
            acc1 = vmlaq_f32(acc1, row2, _ker[1]);
            acc1 = vmlaq_f32(acc1, row3, _ker[2]);
            acc1 = vextq_f32(acc1, acc1, 1);
            float32x2_t sum = vpadd_f32(vget_low_f32(acc0), vget_low_f32(acc1));
            vst1_lane_f32(output_ptr0 + w, sum, 0);
            vst1_lane_f32(output_ptr1 + w, sum, 1);

            row0 = vextq_f32(zero, row0, 3);
            row1 = vextq_f32(zero, row1, 3);
            row2 = vextq_f32(zero, row2, 3);
            row3 = vextq_f32(zero, row3, 3);
          }
        }
        output_ptr0 += valid_w_start;
        output_ptr1 += valid_w_start;
      }
      // valid
      float32x4_t _result0, _result1, _result2, _result3;
      for (int loop = 0; loop < output_w_tiles; ++loop) {
        float32x4_t _row00 = vld1q_f32(input_ptr0);
        float32x4_t _row01 = vld1q_f32(input_ptr0 + 4);
        float32x4_t _row10 = vld1q_f32(input_ptr1);
        float32x4_t _row11 = vld1q_f32(input_ptr1 + 4);

        float32x4_t _ext01 = vextq_f32(_row00, _row01, 1);
        float32x4_t _ext02 = vextq_f32(_row00, _row01, 2);
        float32x4_t _ext03 = vextq_f32(_row01, _row01, 1);
        float32x4_t _ext04 = vextq_f32(_row01, _row01, 2);

        _result0 = vmulq_lane_f32(_row00, vget_low_f32(_ker[0]), 0);
        _result0 = vmlaq_lane_f32(_result0, _ext01, vget_low_f32(_ker[0]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext02, vget_high_f32(_ker[0]), 0);
        _result1 = vmulq_lane_f32(_row01, vget_low_f32(_ker[0]), 0);
        _result1 = vmlaq_lane_f32(_result1, _ext03, vget_low_f32(_ker[0]), 1);
        _result1 = vmlaq_lane_f32(_result1, _ext04, vget_high_f32(_ker[0]), 0);

        _ext01 = vextq_f32(_row10, _row11, 1);
        _ext02 = vextq_f32(_row10, _row11, 2);
        _ext03 = vextq_f32(_row11, _row11, 1);
        _ext04 = vextq_f32(_row11, _row11, 2);

        _result0 = vmlaq_lane_f32(_result0, _row10, vget_low_f32(_ker[1]), 0);
        _result0 = vmlaq_lane_f32(_result0, _ext01, vget_low_f32(_ker[1]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext02, vget_high_f32(_ker[1]), 0);
        _result1 = vmlaq_lane_f32(_result1, _row11, vget_low_f32(_ker[1]), 0);
        _result1 = vmlaq_lane_f32(_result1, _ext03, vget_low_f32(_ker[1]), 1);
        _result1 = vmlaq_lane_f32(_result1, _ext04, vget_high_f32(_ker[1]), 0);

        _result2 = vmulq_lane_f32(_row10, vget_low_f32(_ker[0]), 0);
        _result2 = vmlaq_lane_f32(_result2, _ext01, vget_low_f32(_ker[0]), 1);
        _result2 = vmlaq_lane_f32(_result2, _ext02, vget_high_f32(_ker[0]), 0);
        _result3 = vmulq_lane_f32(_row11, vget_low_f32(_ker[0]), 0);
        _result3 = vmlaq_lane_f32(_result3, _ext03, vget_low_f32(_ker[0]), 1);
        _result3 = vmlaq_lane_f32(_result3, _ext04, vget_high_f32(_ker[0]), 0);

        _row00 = vld1q_f32(input_ptr2);
        _row01 = vld1q_f32(input_ptr2 + 4);
        _row10 = vld1q_f32(input_ptr3);
        _row11 = vld1q_f32(input_ptr3 + 4);

        _ext01 = vextq_f32(_row00, _row01, 1);
        _ext02 = vextq_f32(_row00, _row01, 2);
        _ext03 = vextq_f32(_row01, _row01, 1);
        _ext04 = vextq_f32(_row01, _row01, 2);

        _result0 = vmlaq_lane_f32(_result0, _row00, vget_low_f32(_ker[2]), 0);
        _result0 = vmlaq_lane_f32(_result0, _ext01, vget_low_f32(_ker[2]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext02, vget_high_f32(_ker[2]), 0);
        _result1 = vmlaq_lane_f32(_result1, _row01, vget_low_f32(_ker[2]), 0);
        _result1 = vmlaq_lane_f32(_result1, _ext03, vget_low_f32(_ker[2]), 1);
        _result1 = vmlaq_lane_f32(_result1, _ext04, vget_high_f32(_ker[2]), 0);

        _result2 = vmlaq_lane_f32(_result2, _row00, vget_low_f32(_ker[1]), 0);
        _result2 = vmlaq_lane_f32(_result2, _ext01, vget_low_f32(_ker[1]), 1);
        _result2 = vmlaq_lane_f32(_result2, _ext02, vget_high_f32(_ker[1]), 0);
        _result3 = vmlaq_lane_f32(_result3, _row01, vget_low_f32(_ker[1]), 0);
        _result3 = vmlaq_lane_f32(_result3, _ext03, vget_low_f32(_ker[1]), 1);
        _result3 = vmlaq_lane_f32(_result3, _ext04, vget_high_f32(_ker[1]), 0);

        _ext01 = vextq_f32(_row10, _row11, 1);
        _ext02 = vextq_f32(_row10, _row11, 2);
        _ext03 = vextq_f32(_row11, _row11, 1);
        _ext04 = vextq_f32(_row11, _row11, 2);

        _result2 = vmlaq_lane_f32(_result2, _row10, vget_low_f32(_ker[2]), 0);
        _result2 = vmlaq_lane_f32(_result2, _ext01, vget_low_f32(_ker[2]), 1);
        _result2 = vmlaq_lane_f32(_result2, _ext02, vget_high_f32(_ker[2]), 0);
        _result3 = vmlaq_lane_f32(_result3, _row11, vget_low_f32(_ker[2]), 0);
        _result3 = vmlaq_lane_f32(_result3, _ext03, vget_low_f32(_ker[2]), 1);
        _result3 = vmlaq_lane_f32(_result3, _ext04, vget_high_f32(_ker[2]), 0);

        vst1q_f32(output_ptr0, _result0);
        vst1_f32(output_ptr0 + 4, vget_low_f32(_result1));
        vst1q_f32(output_ptr1, _result2);
        vst1_f32(output_ptr1 + 4, vget_low_f32(_result3));

        input_ptr0 += 6;
        input_ptr1 += 6;
        input_ptr2 += 6;
        input_ptr3 += 6;
        output_ptr0 += 6;
        output_ptr1 += 6;
      }
      // remain w
      if (output_w_remain > 0) {
        float32x4_t _row00 = vld1q_f32(input_ptr0);
        float32x4_t _row01 = vld1q_f32(input_ptr0 + 4);
        float32x4_t _row10 = vld1q_f32(input_ptr1);
        float32x4_t _row11 = vld1q_f32(input_ptr1 + 4);

        float32x4_t _ext01 = vextq_f32(_row00, _row01, 1);
        float32x4_t _ext02 = vextq_f32(_row00, _row01, 2);
        float32x4_t _ext03 = vextq_f32(_row01, _row01, 1);
        float32x4_t _ext04 = vextq_f32(_row01, _row01, 2);

        _result0 = vmulq_lane_f32(_row00, vget_low_f32(_ker[0]), 0);
        _result0 = vmlaq_lane_f32(_result0, _ext01, vget_low_f32(_ker[0]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext02, vget_high_f32(_ker[0]), 0);
        _result1 = vmulq_lane_f32(_row01, vget_low_f32(_ker[0]), 0);
        _result1 = vmlaq_lane_f32(_result1, _ext03, vget_low_f32(_ker[0]), 1);
        _result1 = vmlaq_lane_f32(_result1, _ext04, vget_high_f32(_ker[0]), 0);

        _ext01 = vextq_f32(_row10, _row11, 1);
        _ext02 = vextq_f32(_row10, _row11, 2);
        _ext03 = vextq_f32(_row11, _row11, 1);
        _ext04 = vextq_f32(_row11, _row11, 2);

        _result0 = vmlaq_lane_f32(_result0, _row10, vget_low_f32(_ker[1]), 0);
        _result0 = vmlaq_lane_f32(_result0, _ext01, vget_low_f32(_ker[1]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext02, vget_high_f32(_ker[1]), 0);
        _result1 = vmlaq_lane_f32(_result1, _row11, vget_low_f32(_ker[1]), 0);
        _result1 = vmlaq_lane_f32(_result1, _ext03, vget_low_f32(_ker[1]), 1);
        _result1 = vmlaq_lane_f32(_result1, _ext04, vget_high_f32(_ker[1]), 0);

        _result2 = vmulq_lane_f32(_row10, vget_low_f32(_ker[0]), 0);
        _result2 = vmlaq_lane_f32(_result2, _ext01, vget_low_f32(_ker[0]), 1);
        _result2 = vmlaq_lane_f32(_result2, _ext02, vget_high_f32(_ker[0]), 0);
        _result3 = vmulq_lane_f32(_row11, vget_low_f32(_ker[0]), 0);
        _result3 = vmlaq_lane_f32(_result3, _ext03, vget_low_f32(_ker[0]), 1);
        _result3 = vmlaq_lane_f32(_result3, _ext04, vget_high_f32(_ker[0]), 0);

        _row00 = vld1q_f32(input_ptr2);
        _row01 = vld1q_f32(input_ptr2 + 4);
        _row10 = vld1q_f32(input_ptr3);
        _row11 = vld1q_f32(input_ptr3 + 4);

        _ext01 = vextq_f32(_row00, _row01, 1);
        _ext02 = vextq_f32(_row00, _row01, 2);
        _ext03 = vextq_f32(_row01, _row01, 1);
        _ext04 = vextq_f32(_row01, _row01, 2);

        _result0 = vmlaq_lane_f32(_result0, _row00, vget_low_f32(_ker[2]), 0);
        _result0 = vmlaq_lane_f32(_result0, _ext01, vget_low_f32(_ker[2]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext02, vget_high_f32(_ker[2]), 0);
        _result1 = vmlaq_lane_f32(_result1, _row01, vget_low_f32(_ker[2]), 0);
        _result1 = vmlaq_lane_f32(_result1, _ext03, vget_low_f32(_ker[2]), 1);
        _result1 = vmlaq_lane_f32(_result1, _ext04, vget_high_f32(_ker[2]), 0);

        _result2 = vmlaq_lane_f32(_result2, _row00, vget_low_f32(_ker[1]), 0);
        _result2 = vmlaq_lane_f32(_result2, _ext01, vget_low_f32(_ker[1]), 1);
        _result2 = vmlaq_lane_f32(_result2, _ext02, vget_high_f32(_ker[1]), 0);
        _result3 = vmlaq_lane_f32(_result3, _row01, vget_low_f32(_ker[1]), 0);
        _result3 = vmlaq_lane_f32(_result3, _ext03, vget_low_f32(_ker[1]), 1);
        _result3 = vmlaq_lane_f32(_result3, _ext04, vget_high_f32(_ker[1]), 0);

        _ext01 = vextq_f32(_row10, _row11, 1);
        _ext02 = vextq_f32(_row10, _row11, 2);
        _ext03 = vextq_f32(_row11, _row11, 1);
        _ext04 = vextq_f32(_row11, _row11, 2);

        _result2 = vmlaq_lane_f32(_result2, _row10, vget_low_f32(_ker[2]), 0);
        _result2 = vmlaq_lane_f32(_result2, _ext01, vget_low_f32(_ker[2]), 1);
        _result2 = vmlaq_lane_f32(_result2, _ext02, vget_high_f32(_ker[2]), 0);
        _result3 = vmlaq_lane_f32(_result3, _row11, vget_low_f32(_ker[2]), 0);
        _result3 = vmlaq_lane_f32(_result3, _ext03, vget_low_f32(_ker[2]), 1);
        _result3 = vmlaq_lane_f32(_result3, _ext04, vget_high_f32(_ker[2]), 0);

        switch (output_w_remain) {
          case 5:
            vst1q_lane_f32(output_ptr0 + 4, _result1, 0);
            vst1q_lane_f32(output_ptr1 + 4, _result3, 0);
          case 4:
            vst1q_f32(output_ptr0, _result0);
            vst1q_f32(output_ptr1, _result2);
            break;
          case 3:
            vst1q_lane_f32(output_ptr0 + 2, _result0, 2);
            vst1q_lane_f32(output_ptr1 + 2, _result2, 2);
          case 2:
            vst1_f32(output_ptr0, vget_low_f32(_result0));
            vst1_f32(output_ptr1, vget_low_f32(_result2));
            break;
          case 1:
            vst1q_lane_f32(output_ptr0, _result0, 0);
            vst1q_lane_f32(output_ptr1, _result2, 0);
            break;
        }

        input_ptr0 += output_w_remain;
        input_ptr1 += output_w_remain;
        input_ptr2 += output_w_remain;
        input_ptr3 += output_w_remain;
        output_ptr0 += output_w_remain;
        output_ptr1 += output_w_remain;
      }
      // pad right
      if (padding_w) {
        float32x2_t row0 = vld1_f32(input_ptr0);
        float32x2_t row1 = vld1_f32(input_ptr1);
        float32x2_t row2 = vld1_f32(input_ptr2);
        float32x2_t row3 = vld1_f32(input_ptr3);
        float32x2_t zero = vdup_n_f32(0.f);
        float32x2_t acc0, acc1;
        for (int w = valid_w_end; w < output_w; ++w) {
          int padding = w + 3 - (padding_w + input_w);
          if (padding >= 3) {
            *output_ptr0 = 0.f;
            *output_ptr1 = 0.f;
          } else {
            acc0 = vmul_f32(row0, vget_low_f32(_ker[0]));
            acc0 = vmla_f32(acc0, row1, vget_low_f32(_ker[1]));
            acc0 = vmla_f32(acc0, row2, vget_low_f32(_ker[2]));
            acc1 = vmul_f32(row1, vget_low_f32(_ker[0]));
            acc1 = vmla_f32(acc1, row2, vget_low_f32(_ker[1]));
            acc1 = vmla_f32(acc1, row3, vget_low_f32(_ker[2]));
            float32x2_t sum = vpadd_f32(acc0, acc1);
            vst1_lane_f32(output_ptr0, sum, 0);
            vst1_lane_f32(output_ptr1, sum, 1);
            row0 = vext_f32(row0, zero, 1);
            row1 = vext_f32(row1, zero, 1);
            row2 = vext_f32(row2, zero, 1);
            row3 = vext_f32(row3, zero, 1);
          }
          output_ptr0++;
          output_ptr1++;
        }
      }
    }
    // remain height
    int start_h = valid_h_start + (valid_h & 0xfffffffe);
    if (start_h < valid_h_end) {
      const float *input_ptr0 = input_ptr + (start_h - padding_h) * input_w;
      const float *input_ptr1 = input_ptr0 + input_w;
      const float *input_ptr2 = input_ptr1 + input_w;
      float *output_ptr0 = output_ptr + start_h * output_w;
      // pad left
      if (padding_w) {
        float32x4_t row0 = vld1q_f32(input_ptr0);
        float32x4_t row1 = vld1q_f32(input_ptr1);
        float32x4_t row2 = vld1q_f32(input_ptr2);
        float32x4_t zero = vdupq_n_f32(0.f);
        row0 = vextq_f32(zero, row0, 3);
        row1 = vextq_f32(zero, row1, 3);
        row2 = vextq_f32(zero, row2, 3);
        float32x4_t acc;
        for (int w = valid_w_start - 1; w >= 0; --w) {
          int padding = padding_w - w;
          if (padding >= 3) {
            output_ptr0[w] = 0.f;
          } else {
            acc = vmulq_f32(row0, _ker[0]);
            acc = vmlaq_f32(acc, row1, _ker[1]);
            acc = vmlaq_f32(acc, row2, _ker[2]);
            acc = vextq_f32(acc, acc, 1);
            float32x2_t sum = vpadd_f32(vget_low_f32(acc), vget_low_f32(acc));
            vst1_lane_f32(output_ptr0 + w, sum, 0);

            row0 = vextq_f32(zero, row0, 3);
            row1 = vextq_f32(zero, row1, 3);
            row2 = vextq_f32(zero, row2, 3);
          }
        }
        output_ptr0 += valid_w_start;
      }
      // valid
      float32x4_t _result0, _result1;
      for (int loop = 0; loop < output_w_tiles; ++loop) {
        float32x4_t _row00 = vld1q_f32(input_ptr0);
        float32x4_t _row01 = vld1q_f32(input_ptr0 + 4);
        float32x4_t _row10 = vld1q_f32(input_ptr1);
        float32x4_t _row11 = vld1q_f32(input_ptr1 + 4);

        float32x4_t _ext01 = vextq_f32(_row00, _row01, 1);
        float32x4_t _ext02 = vextq_f32(_row00, _row01, 2);
        float32x4_t _ext03 = vextq_f32(_row01, _row01, 1);
        float32x4_t _ext04 = vextq_f32(_row01, _row01, 2);

        _result0 = vmulq_lane_f32(_row00, vget_low_f32(_ker[0]), 0);
        _result0 = vmlaq_lane_f32(_result0, _ext01, vget_low_f32(_ker[0]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext02, vget_high_f32(_ker[0]), 0);
        _result1 = vmulq_lane_f32(_row01, vget_low_f32(_ker[0]), 0);
        _result1 = vmlaq_lane_f32(_result1, _ext03, vget_low_f32(_ker[0]), 1);
        _result1 = vmlaq_lane_f32(_result1, _ext04, vget_high_f32(_ker[0]), 0);

        _ext01 = vextq_f32(_row10, _row11, 1);
        _ext02 = vextq_f32(_row10, _row11, 2);
        _ext03 = vextq_f32(_row11, _row11, 1);
        _ext04 = vextq_f32(_row11, _row11, 2);

        _result0 = vmlaq_lane_f32(_result0, _row10, vget_low_f32(_ker[1]), 0);
        _result0 = vmlaq_lane_f32(_result0, _ext01, vget_low_f32(_ker[1]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext02, vget_high_f32(_ker[1]), 0);
        _result1 = vmlaq_lane_f32(_result1, _row11, vget_low_f32(_ker[1]), 0);
        _result1 = vmlaq_lane_f32(_result1, _ext03, vget_low_f32(_ker[1]), 1);
        _result1 = vmlaq_lane_f32(_result1, _ext04, vget_high_f32(_ker[1]), 0);

        _row00 = vld1q_f32(input_ptr2);
        _row01 = vld1q_f32(input_ptr2 + 4);

        _ext01 = vextq_f32(_row00, _row01, 1);
        _ext02 = vextq_f32(_row00, _row01, 2);
        _ext03 = vextq_f32(_row01, _row01, 1);
        _ext04 = vextq_f32(_row01, _row01, 2);

        _result0 = vmlaq_lane_f32(_result0, _row00, vget_low_f32(_ker[2]), 0);
        _result0 = vmlaq_lane_f32(_result0, _ext01, vget_low_f32(_ker[2]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext02, vget_high_f32(_ker[2]), 0);
        _result1 = vmlaq_lane_f32(_result1, _row01, vget_low_f32(_ker[2]), 0);
        _result1 = vmlaq_lane_f32(_result1, _ext03, vget_low_f32(_ker[2]), 1);
        _result1 = vmlaq_lane_f32(_result1, _ext04, vget_high_f32(_ker[2]), 0);

        vst1q_f32(output_ptr0, _result0);
        vst1_f32(output_ptr0 + 4, vget_low_f32(_result1));

        input_ptr0 += 6;
        input_ptr1 += 6;
        input_ptr2 += 6;
        output_ptr0 += 6;
      }

      if (output_w_remain > 0) {
        float32x4_t _row00 = vld1q_f32(input_ptr0);
        float32x4_t _row01 = vld1q_f32(input_ptr0 + 4);
        float32x4_t _row10 = vld1q_f32(input_ptr1);
        float32x4_t _row11 = vld1q_f32(input_ptr1 + 4);

        float32x4_t _ext01 = vextq_f32(_row00, _row01, 1);
        float32x4_t _ext02 = vextq_f32(_row00, _row01, 2);
        float32x4_t _ext03 = vextq_f32(_row01, _row01, 1);
        float32x4_t _ext04 = vextq_f32(_row01, _row01, 2);

        _result0 = vmulq_lane_f32(_row00, vget_low_f32(_ker[0]), 0);
        _result0 = vmlaq_lane_f32(_result0, _ext01, vget_low_f32(_ker[0]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext02, vget_high_f32(_ker[0]), 0);
        _result1 = vmulq_lane_f32(_row01, vget_low_f32(_ker[0]), 0);
        _result1 = vmlaq_lane_f32(_result1, _ext03, vget_low_f32(_ker[0]), 1);
        _result1 = vmlaq_lane_f32(_result1, _ext04, vget_high_f32(_ker[0]), 0);

        _ext01 = vextq_f32(_row10, _row11, 1);
        _ext02 = vextq_f32(_row10, _row11, 2);
        _ext03 = vextq_f32(_row11, _row11, 1);
        _ext04 = vextq_f32(_row11, _row11, 2);

        _result0 = vmlaq_lane_f32(_result0, _row10, vget_low_f32(_ker[1]), 0);
        _result0 = vmlaq_lane_f32(_result0, _ext01, vget_low_f32(_ker[1]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext02, vget_high_f32(_ker[1]), 0);
        _result1 = vmlaq_lane_f32(_result1, _row11, vget_low_f32(_ker[1]), 0);
        _result1 = vmlaq_lane_f32(_result1, _ext03, vget_low_f32(_ker[1]), 1);
        _result1 = vmlaq_lane_f32(_result1, _ext04, vget_high_f32(_ker[1]), 0);

        _row00 = vld1q_f32(input_ptr2);
        _row01 = vld1q_f32(input_ptr2 + 4);

        _ext01 = vextq_f32(_row00, _row01, 1);
        _ext02 = vextq_f32(_row00, _row01, 2);
        _ext03 = vextq_f32(_row01, _row01, 1);
        _ext04 = vextq_f32(_row01, _row01, 2);

        _result0 = vmlaq_lane_f32(_result0, _row00, vget_low_f32(_ker[2]), 0);
        _result0 = vmlaq_lane_f32(_result0, _ext01, vget_low_f32(_ker[2]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext02, vget_high_f32(_ker[2]), 0);
        _result1 = vmlaq_lane_f32(_result1, _row01, vget_low_f32(_ker[2]), 0);
        _result1 = vmlaq_lane_f32(_result1, _ext03, vget_low_f32(_ker[2]), 1);
        _result1 = vmlaq_lane_f32(_result1, _ext04, vget_high_f32(_ker[2]), 0);

        switch (output_w_remain) {
          case 5:
            vst1q_lane_f32(output_ptr0 + 4, _result1, 0);
          case 4:
            vst1q_f32(output_ptr0, _result0);
            break;
          case 3:
            vst1q_lane_f32(output_ptr0 + 2, _result0, 2);
          case 2:
            vst1_f32(output_ptr0, vget_low_f32(_result0));
            break;
          case 1:
            vst1q_lane_f32(output_ptr0, _result0, 0);
            break;
        }

        input_ptr0 += output_w_remain;
        input_ptr1 += output_w_remain;
        input_ptr2 += output_w_remain;
        output_ptr0 += output_w_remain;
      }
      // pad right
      if (padding_w) {
        float32x2_t row0 = vld1_f32(input_ptr0);
        float32x2_t row1 = vld1_f32(input_ptr1);
        float32x2_t row2 = vld1_f32(input_ptr2);
        float32x2_t zero = vdup_n_f32(0.f);
        float32x2_t acc;
        for (int w = valid_w_end; w < output_w; ++w) {
          int padding = w + 3 - (padding_w + input_w);
          if (padding >= 3) {
            *output_ptr0 = 0.f;
          } else {
            acc = vmul_f32(row0, vget_low_f32(_ker[0]));
            acc = vmla_f32(acc, row1, vget_low_f32(_ker[1]));
            acc = vmla_f32(acc, row2, vget_low_f32(_ker[2]));
            float32x2_t sum = vpadd_f32(acc, acc);
            vst1_lane_f32(output_ptr0, sum, 0);
            row0 = vext_f32(row0, zero, 1);
            row1 = vext_f32(row1, zero, 1);
            row2 = vext_f32(row2, zero, 1);
          }
          output_ptr0++;
        }
      }
    }
    // pad bottom
    for (int h = valid_h_end; h < output_h; ++h) {
      DepthwiseConv3x3NormalRow<1, 1>(input_ptr, filter_ptr, h, input_h,
                                      input_w, padding_h, padding_w, output_w,
                                      output_ptr, _ker);
    }
  }
}

template <>
void DepthwiseConv3x3S2<float, float>(const framework::Tensor &input,
                                      const framework::Tensor &filter,
                                      const std::vector<int> &paddings,
                                      framework::Tensor *output) {
  const float *input_data = input.data<float>();
  const float *filter_data = filter.data<float>();
  float *out_data = output->mutable_data<float>();

  const int input_h = input.dims()[2];
  const int input_w = input.dims()[3];
  const int output_h = output->dims()[2];
  const int output_w = output->dims()[3];
  const int padding_h = paddings[0];
  const int padding_w = paddings[1];
  const int image_size = input_h * input_w;
  const int out_image_size = output_h * output_w;
  const int valid_h_start = (padding_h + 1) / 2;
  const int valid_h_end = (input_h + padding_h - 1) / 2;
  const int valid_h = valid_h_end - valid_h_start;
  const int valid_w_start = (padding_w + 1) / 2;
  const int valid_w_end = (input_w + padding_w - 1) / 2;
  const int valid_w = valid_w_end - valid_w_start;
  const int input_w_start = 2 * valid_w_start - padding_w;

  #pragma omp parallel for
  for (int g = 0; g < input.dims()[1]; ++g) {
    const float *input_ptr = input_data + g * image_size;
    const float *filter_ptr = filter_data + g * 9;
    float *output_ptr = out_data + g * out_image_size;

    const float *filter_ptr0 = filter_ptr;
    const float *filter_ptr1 = filter_ptr0 + 3;
    const float *filter_ptr2 = filter_ptr1 + 3;
    float32x4_t _ker[3];
    _ker[0] = vld1q_f32(filter_ptr0);
    _ker[1] = vld1q_f32(filter_ptr1);
    _ker[2] = vld1q_f32(filter_ptr2);

    // pad top
    for (int h = 0; h < valid_h_start; ++h) {
      DepthwiseConv3x3NormalRow<2, 2>(input_ptr, filter_ptr, h, input_h,
                                      input_w, padding_h, padding_w, output_w,
                                      output_ptr, _ker);
    }
    // valid 2x4
    int output_w_tiles = valid_w / 4;
    int output_w_remain = valid_w - output_w_tiles * 4;
    for (int h = valid_h_start; h < valid_h_end - 1; h += 2) {
      const float *input_ptr0 = input_ptr + (2 * h - padding_h) * input_w;
      const float *input_ptr1 = input_ptr0 + input_w;
      const float *input_ptr2 = input_ptr1 + input_w;
      const float *input_ptr3 = input_ptr2 + input_w;
      const float *input_ptr4 = input_ptr3 + input_w;
      float *output_ptr0 = output_ptr + h * output_w;
      float *output_ptr1 = output_ptr0 + output_w;
      // pad left
      if (padding_w) {
        for (int w = valid_w_start - 1; w >= 0; --w) {
          int padding = padding_w - (w << 1);
          if (padding >= 3) {
            output_ptr0[w] = 0;
            output_ptr1[w] = 0;
          } else {
            float32x4_t row0 = vld1q_f32(input_ptr0 - padding);
            float32x4_t row1 = vld1q_f32(input_ptr1 - padding);
            float32x4_t row2 = vld1q_f32(input_ptr2 - padding);
            float32x4_t row3 = vld1q_f32(input_ptr3 - padding);
            float32x4_t row4 = vld1q_f32(input_ptr4 - padding);
            float32x4_t acc0 = vmulq_f32(row0, _ker[0]);
            float32x4_t acc1 = vmulq_f32(row2, _ker[0]);
            acc0 = vmlaq_f32(acc0, row1, _ker[1]);
            acc1 = vmlaq_f32(acc1, row3, _ker[1]);
            acc0 = vmlaq_f32(acc0, row2, _ker[2]);
            acc1 = vmlaq_f32(acc1, row4, _ker[2]);
            float sum0 = vgetq_lane_f32(acc0, 2);
            float sum1 = vgetq_lane_f32(acc1, 2);
            if (padding == 1) {
              sum0 += vgetq_lane_f32(acc0, 1);
              sum1 += vgetq_lane_f32(acc1, 1);
            }
            output_ptr0[w] = sum0;
            output_ptr1[w] = sum1;
          }
        }
        input_ptr0 += input_w_start;
        input_ptr1 += input_w_start;
        input_ptr2 += input_w_start;
        input_ptr3 += input_w_start;
        input_ptr4 += input_w_start;
        output_ptr0 += valid_w_start;
        output_ptr1 += valid_w_start;
      }
      // valid
      float32x4_t _result0, _result1, _ext;
      for (int loop = 0; loop < output_w_tiles; ++loop) {
        float32x4x2_t _row0 = vld2q_f32(input_ptr0);
        float32x4x2_t _row1 = vld2q_f32(input_ptr1);

        _ext = vextq_f32(_row0.val[0], _ext, 1);
        _ext = vsetq_lane_f32(input_ptr0[8], _ext, 3);
        _result0 = vmulq_lane_f32(_row0.val[0], vget_low_f32(_ker[0]), 0);
        _result0 =
            vmlaq_lane_f32(_result0, _row0.val[1], vget_low_f32(_ker[0]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext, vget_high_f32(_ker[0]), 0);

        _ext = vextq_f32(_row1.val[0], _ext, 1);
        _ext = vsetq_lane_f32(input_ptr1[8], _ext, 3);
        _result0 =
            vmlaq_lane_f32(_result0, _row1.val[0], vget_low_f32(_ker[1]), 0);
        _result0 =
            vmlaq_lane_f32(_result0, _row1.val[1], vget_low_f32(_ker[1]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext, vget_high_f32(_ker[1]), 0);

        _row0 = vld2q_f32(input_ptr2);
        _row1 = vld2q_f32(input_ptr3);

        _ext = vextq_f32(_row0.val[0], _ext, 1);
        _ext = vsetq_lane_f32(input_ptr2[8], _ext, 3);
        _result0 =
            vmlaq_lane_f32(_result0, _row0.val[0], vget_low_f32(_ker[2]), 0);
        _result0 =
            vmlaq_lane_f32(_result0, _row0.val[1], vget_low_f32(_ker[2]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext, vget_high_f32(_ker[2]), 0);
        _result1 = vmulq_lane_f32(_row0.val[0], vget_low_f32(_ker[0]), 0);
        _result1 =
            vmlaq_lane_f32(_result1, _row0.val[1], vget_low_f32(_ker[0]), 1);
        _result1 = vmlaq_lane_f32(_result1, _ext, vget_high_f32(_ker[0]), 0);

        _ext = vextq_f32(_row1.val[0], _ext, 1);
        _ext = vsetq_lane_f32(input_ptr3[8], _ext, 3);
        _result1 =
            vmlaq_lane_f32(_result1, _row1.val[0], vget_low_f32(_ker[1]), 0);
        _result1 =
            vmlaq_lane_f32(_result1, _row1.val[1], vget_low_f32(_ker[1]), 1);
        _result1 = vmlaq_lane_f32(_result1, _ext, vget_high_f32(_ker[1]), 0);

        _row0 = vld2q_f32(input_ptr4);

        _ext = vextq_f32(_row0.val[0], _ext, 1);
        _ext = vsetq_lane_f32(input_ptr4[8], _ext, 3);
        _result1 =
            vmlaq_lane_f32(_result1, _row0.val[0], vget_low_f32(_ker[2]), 0);
        _result1 =
            vmlaq_lane_f32(_result1, _row0.val[1], vget_low_f32(_ker[2]), 1);
        _result1 = vmlaq_lane_f32(_result1, _ext, vget_high_f32(_ker[2]), 0);

        vst1q_f32(output_ptr0, _result0);
        vst1q_f32(output_ptr1, _result1);

        input_ptr0 += 8;
        input_ptr1 += 8;
        input_ptr2 += 8;
        input_ptr3 += 8;
        input_ptr4 += 8;
        output_ptr0 += 4;
        output_ptr1 += 4;
      }
      // remain w
      if (output_w_remain > 0) {
        float32x4x2_t _row0 = vld2q_f32(input_ptr0);
        float32x4x2_t _row1 = vld2q_f32(input_ptr1);

        _ext = vextq_f32(_row0.val[0], _ext, 1);
        _ext = vsetq_lane_f32(input_ptr0[8], _ext, 3);
        _result0 = vmulq_lane_f32(_row0.val[0], vget_low_f32(_ker[0]), 0);
        _result0 =
            vmlaq_lane_f32(_result0, _row0.val[1], vget_low_f32(_ker[0]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext, vget_high_f32(_ker[0]), 0);

        _ext = vextq_f32(_row1.val[0], _ext, 1);
        _ext = vsetq_lane_f32(input_ptr1[8], _ext, 3);
        _result0 =
            vmlaq_lane_f32(_result0, _row1.val[0], vget_low_f32(_ker[1]), 0);
        _result0 =
            vmlaq_lane_f32(_result0, _row1.val[1], vget_low_f32(_ker[1]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext, vget_high_f32(_ker[1]), 0);

        _row0 = vld2q_f32(input_ptr2);
        _row1 = vld2q_f32(input_ptr3);

        _ext = vextq_f32(_row0.val[0], _ext, 1);
        _ext = vsetq_lane_f32(input_ptr2[8], _ext, 3);
        _result0 =
            vmlaq_lane_f32(_result0, _row0.val[0], vget_low_f32(_ker[2]), 0);
        _result0 =
            vmlaq_lane_f32(_result0, _row0.val[1], vget_low_f32(_ker[2]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext, vget_high_f32(_ker[2]), 0);
        _result1 = vmulq_lane_f32(_row0.val[0], vget_low_f32(_ker[0]), 0);
        _result1 =
            vmlaq_lane_f32(_result1, _row0.val[1], vget_low_f32(_ker[0]), 1);
        _result1 = vmlaq_lane_f32(_result1, _ext, vget_high_f32(_ker[0]), 0);

        _ext = vextq_f32(_row1.val[0], _ext, 1);
        _ext = vsetq_lane_f32(input_ptr3[8], _ext, 3);
        _result1 =
            vmlaq_lane_f32(_result1, _row1.val[0], vget_low_f32(_ker[1]), 0);
        _result1 =
            vmlaq_lane_f32(_result1, _row1.val[1], vget_low_f32(_ker[1]), 1);
        _result1 = vmlaq_lane_f32(_result1, _ext, vget_high_f32(_ker[1]), 0);

        _row0 = vld2q_f32(input_ptr4);

        _ext = vextq_f32(_row0.val[0], _ext, 1);
        _ext = vsetq_lane_f32(input_ptr4[8], _ext, 3);
        _result1 =
            vmlaq_lane_f32(_result1, _row0.val[0], vget_low_f32(_ker[2]), 0);
        _result1 =
            vmlaq_lane_f32(_result1, _row0.val[1], vget_low_f32(_ker[2]), 1);
        _result1 = vmlaq_lane_f32(_result1, _ext, vget_high_f32(_ker[2]), 0);

        switch (output_w_remain) {
          case 3:
            vst1q_lane_f32(output_ptr0 + 2, _result0, 2);
            vst1q_lane_f32(output_ptr1 + 2, _result1, 2);
          case 2:
            vst1_f32(output_ptr0, vget_low_f32(_result0));
            vst1_f32(output_ptr1, vget_low_f32(_result1));
            break;
          case 1:
            vst1q_lane_f32(output_ptr0, _result0, 0);
            vst1q_lane_f32(output_ptr1, _result1, 0);
            break;
        }
        input_ptr0 += output_w_remain * 2;
        input_ptr1 += output_w_remain * 2;
        input_ptr2 += output_w_remain * 2;
        input_ptr3 += output_w_remain * 2;
        input_ptr4 += output_w_remain * 2;
        output_ptr0 += output_w_remain;
        output_ptr1 += output_w_remain;
      }
      // pad right
      if (padding_w > 0) {
        float32x4_t row0 = vld1q_f32(input_ptr0);
        float32x4_t row1 = vld1q_f32(input_ptr1);
        float32x4_t row2 = vld1q_f32(input_ptr2);
        float32x4_t row3 = vld1q_f32(input_ptr3);
        float32x4_t row4 = vld1q_f32(input_ptr4);
        float32x4_t acc0, acc1;
        for (int w = valid_w_end; w < output_w; ++w) {
          int padding = 2 * w + 3 - (padding_w + input_w);
          if (padding >= 3) {
            *output_ptr0 = 0;
            *output_ptr1 = 0;
          } else {
            acc0 = vmulq_f32(row0, _ker[0]);
            acc1 = vmulq_f32(row2, _ker[0]);
            acc0 = vmlaq_f32(acc0, row1, _ker[1]);
            acc1 = vmlaq_f32(acc1, row3, _ker[1]);
            acc0 = vmlaq_f32(acc0, row2, _ker[2]);
            acc1 = vmlaq_f32(acc1, row4, _ker[2]);
            float sum0 = vgetq_lane_f32(acc0, 0);
            float sum1 = vgetq_lane_f32(acc1, 0);
            if (padding == 1) {
              sum0 += vgetq_lane_f32(acc0, 1);
              sum1 += vgetq_lane_f32(acc1, 1);
            }
            *output_ptr0 = sum0;
            *output_ptr1 = sum1;
          }
          output_ptr0++;
          output_ptr1++;
        }
      }
    }
    // remain height
    int start_h = valid_h_start + (valid_h & 0xfffffffe);
    if (start_h < valid_h_end) {
      const float *input_ptr0 = input_ptr + (2 * start_h - padding_h) * input_w;
      const float *input_ptr1 = input_ptr0 + input_w;
      const float *input_ptr2 = input_ptr1 + input_w;
      float *output_ptr0 = output_ptr + start_h * output_w;
      // pad left
      if (padding_w) {
        for (int w = valid_w_start - 1; w >= 0; --w) {
          int padding = padding_w - (w << 1);
          if (padding >= 3) {
            output_ptr0[w] = 0;
          } else {
            float32x4_t row0 = vld1q_f32(input_ptr0 - padding);
            float32x4_t row1 = vld1q_f32(input_ptr1 - padding);
            float32x4_t row2 = vld1q_f32(input_ptr2 - padding);
            float32x4_t acc0 = vmulq_f32(row0, _ker[0]);
            acc0 = vmlaq_f32(acc0, row1, _ker[1]);
            acc0 = vmlaq_f32(acc0, row2, _ker[2]);
            float sum0 = vgetq_lane_f32(acc0, 2);
            if (padding == 1) {
              sum0 += vgetq_lane_f32(acc0, 1);
            }
            output_ptr0[w] = sum0;
          }
        }
        input_ptr0 += input_w_start;
        input_ptr1 += input_w_start;
        input_ptr2 += input_w_start;
        output_ptr0 += valid_w_start;
      }
      // valid
      float32x4_t _result0, _ext;
      for (int loop = 0; loop < output_w_tiles; ++loop) {
        float32x4x2_t _row0 = vld2q_f32(input_ptr0);
        float32x4x2_t _row1 = vld2q_f32(input_ptr1);
        float32x4x2_t _row2 = vld2q_f32(input_ptr2);

        _ext = vextq_f32(_row0.val[0], _ext, 1);
        _ext = vsetq_lane_f32(input_ptr0[8], _ext, 3);
        _result0 = vmulq_lane_f32(_row0.val[0], vget_low_f32(_ker[0]), 0);
        _result0 =
            vmlaq_lane_f32(_result0, _row0.val[1], vget_low_f32(_ker[0]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext, vget_high_f32(_ker[0]), 0);

        _ext = vextq_f32(_row1.val[0], _ext, 1);
        _ext = vsetq_lane_f32(input_ptr1[8], _ext, 3);
        _result0 =
            vmlaq_lane_f32(_result0, _row1.val[0], vget_low_f32(_ker[1]), 0);
        _result0 =
            vmlaq_lane_f32(_result0, _row1.val[1], vget_low_f32(_ker[1]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext, vget_high_f32(_ker[1]), 0);

        _ext = vextq_f32(_row2.val[0], _ext, 1);
        _ext = vsetq_lane_f32(input_ptr2[8], _ext, 3);
        _result0 =
            vmlaq_lane_f32(_result0, _row2.val[0], vget_low_f32(_ker[2]), 0);
        _result0 =
            vmlaq_lane_f32(_result0, _row2.val[1], vget_low_f32(_ker[2]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext, vget_high_f32(_ker[2]), 0);

        vst1q_f32(output_ptr0, _result0);

        input_ptr0 += 8;
        input_ptr1 += 8;
        input_ptr2 += 8;
        output_ptr0 += 4;
      }
      // remain w
      if (output_w_remain > 0) {
        float32x4x2_t _row0 = vld2q_f32(input_ptr0);
        float32x4x2_t _row1 = vld2q_f32(input_ptr1);
        float32x4x2_t _row2 = vld2q_f32(input_ptr2);

        _ext = vextq_f32(_row0.val[0], _ext, 1);
        _ext = vsetq_lane_f32(input_ptr0[8], _ext, 3);
        _result0 = vmulq_lane_f32(_row0.val[0], vget_low_f32(_ker[0]), 0);
        _result0 =
            vmlaq_lane_f32(_result0, _row0.val[1], vget_low_f32(_ker[0]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext, vget_high_f32(_ker[0]), 0);

        _ext = vextq_f32(_row1.val[0], _ext, 1);
        _ext = vsetq_lane_f32(input_ptr1[8], _ext, 3);
        _result0 =
            vmlaq_lane_f32(_result0, _row1.val[0], vget_low_f32(_ker[1]), 0);
        _result0 =
            vmlaq_lane_f32(_result0, _row1.val[1], vget_low_f32(_ker[1]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext, vget_high_f32(_ker[1]), 0);

        _ext = vextq_f32(_row2.val[0], _ext, 1);
        _ext = vsetq_lane_f32(input_ptr2[8], _ext, 3);
        _result0 =
            vmlaq_lane_f32(_result0, _row2.val[0], vget_low_f32(_ker[2]), 0);
        _result0 =
            vmlaq_lane_f32(_result0, _row2.val[1], vget_low_f32(_ker[2]), 1);
        _result0 = vmlaq_lane_f32(_result0, _ext, vget_high_f32(_ker[2]), 0);

        switch (output_w_remain) {
          case 3:
            vst1q_lane_f32(output_ptr0 + 2, _result0, 2);
          case 2:
            vst1_f32(output_ptr0, vget_low_f32(_result0));
            break;
          case 1:
            vst1q_lane_f32(output_ptr0, _result0, 0);
            break;
        }
        input_ptr0 += output_w_remain * 2;
        input_ptr1 += output_w_remain * 2;
        input_ptr2 += output_w_remain * 2;
        output_ptr0 += output_w_remain;
      }
      // pad right
      if (padding_w) {
        float32x4_t row0 = vld1q_f32(input_ptr0);
        float32x4_t row1 = vld1q_f32(input_ptr1);
        float32x4_t row2 = vld1q_f32(input_ptr2);
        float32x4_t acc0;
        for (int w = valid_w_end; w < output_w; ++w) {
          int padding = 2 * w + 3 - (padding_w + input_w);
          if (padding >= 3) {
            *output_ptr0 = 0;
          } else {
            acc0 = vmulq_f32(row0, _ker[0]);
            acc0 = vmlaq_f32(acc0, row1, _ker[1]);
            acc0 = vmlaq_f32(acc0, row2, _ker[2]);
            float sum0 = vgetq_lane_f32(acc0, 0);
            if (padding == 1) {
              sum0 += vgetq_lane_f32(acc0, 1);
            }
            *output_ptr0 = sum0;
          }
          output_ptr0++;
        }
      }
    }
    // pad bottom
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
