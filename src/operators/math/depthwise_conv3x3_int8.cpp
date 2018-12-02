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

#include "operators/math/depthwise_conv3x3.h"
#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {
namespace math {

template <int Stride>
inline void Depth3x3ValidColLoadInput(const int8_t *input, const int input_w,
                                      const int valid_cols, int16x8_t *y0,
                                      int16x8_t *y1, int16x8_t *y2) {
  PADDLE_MOBILE_THROW_EXCEPTION("Stride %d is not supported.", Stride);
}

template <>
inline void Depth3x3ValidColLoadInput<1>(const int8_t *input, const int input_w,
                                         const int valid_cols, int16x8_t *y0,
                                         int16x8_t *y1, int16x8_t *y2) {
  int8_t fake_input[3][8];
  if (valid_cols == 1) {
    for (int i = 0; i < 8; ++i, input += input_w) {
      fake_input[0][i] = input[0];
    }
  } else if (valid_cols == 2) {
    for (int i = 0; i < 8; ++i, input += input_w) {
      fake_input[0][i] = input[0];
      fake_input[1][i] = input[1];
    }
  } else {
    for (int i = 0; i < 8; ++i, input += input_w) {
      fake_input[0][i] = input[0];
      fake_input[1][i] = input[1];
      fake_input[2][i] = input[2];
    }
  }
  int8x8_t input0 = vld1_s8(fake_input[0]);
  int8x8_t input1 = vld1_s8(fake_input[1]);
  int8x8_t input2 = vld1_s8(fake_input[2]);
  y0[0] = vmovl_s8(input0);
  y1[0] = vmovl_s8(input1);
  y2[0] = vmovl_s8(input2);
  y0[1] = vextq_s16(y0[0], y0[0], 1);
  y0[2] = vextq_s16(y0[0], y0[0], 2);
  y1[1] = vextq_s16(y1[0], y1[0], 1);
  y1[2] = vextq_s16(y1[0], y1[0], 2);
  y2[1] = vextq_s16(y2[0], y2[0], 1);
  y2[2] = vextq_s16(y2[0], y2[0], 2);
}

template <>
inline void Depth3x3ValidColLoadInput<2>(const int8_t *input, const int input_w,
                                         const int valid_cols, int16x8_t *y0,
                                         int16x8_t *y1, int16x8_t *y2) {
  int8_t fake_input[3][13];
  if (valid_cols == 1) {
    for (int i = 0; i < 13; ++i, input += input_w) {
      fake_input[0][i] = input[0];
    }
  } else if (valid_cols == 2) {
    for (int i = 0; i < 13; ++i, input += input_w) {
      fake_input[0][i] = input[0];
      fake_input[1][i] = input[1];
    }
  } else {
    for (int i = 0; i < 13; ++i, input += input_w) {
      fake_input[0][i] = input[0];
      fake_input[1][i] = input[1];
      fake_input[2][i] = input[2];
    }
  }
  int8x8x2_t input0 = vld2_s8(fake_input[0]);
  int8x8x2_t input1 = vld2_s8(fake_input[1]);
  int8x8x2_t input2 = vld2_s8(fake_input[2]);
  y0[0] = vmovl_s8(input0.val[0]);
  y0[1] = vmovl_s8(input0.val[1]);
  y0[2] = vextq_s16(y0[0], y0[0], 1);
  y1[0] = vmovl_s8(input1.val[0]);
  y1[1] = vmovl_s8(input1.val[1]);
  y1[2] = vextq_s16(y1[0], y1[0], 1);
  y2[0] = vmovl_s8(input2.val[0]);
  y2[1] = vmovl_s8(input2.val[1]);
  y2[2] = vextq_s16(y2[0], y2[0], 1);
}

template <int Stride_h, int Stride_w>
inline void DepthwiseConv3x3ValidCol(const int8_t *input, const int8_t *filter,
                                     const int h_output, const int h_output_end,
                                     const int w_output, const int input_h,
                                     const int input_w, const int padding_h,
                                     const int padding_w, const int output_w,
                                     int32_t *output) {
  const int w_in_start = -padding_w + w_output * Stride_w;
  const int w_in_end = w_in_start + 3;
  const int w_start = w_in_start > 0 ? w_in_start : 0;
  const int w_end = w_in_end < input_w ? w_in_end : input_w;
  int remain_start = h_output;

#ifdef __ARM_NEON__
  int output_tiles = (h_output_end - h_output) / 6;
  remain_start = h_output + output_tiles * 6;
  int input_h_start = h_output * Stride_h - padding_h;
  size_t input_offset = input_h_start * input_w + w_start;
  size_t output_offset = h_output * output_w + w_output;
  int16x8_t _input[3][3];
  int16x4_t _kernel[3];
  int32x4_t _sum0, _sum1;
  const int8_t *filter_ptr = filter;
  asm volatile(
      "mov        r0, #3                        \n"
      "vld1.s8    d10, [%[filter]], r0  \n"
      "vld1.s8    d11, [%[filter]], r0  \n"
      "vld1.s8    d12, [%[filter]]      \n"
      "vtrn.8     d10, d11              \n"
      "vtrn.8     d12, d13              \n"
      "vtrn.16    d10, d12              \n"
      "vtrn.16    d11, d13              \n"
      "vmovl.s8   q7, d10               \n"
      "vmovl.s8   q8, d11               \n"
      "vmovl.s8   q9, d12               \n"
      "vmov.32    %[_kernel0], d14      \n"
      "vmov.32    %[_kernel1], d16      \n"
      "vmov.32    %[_kernel2], d18      \n"
      : [_kernel0] "+w"(_kernel[0]), [_kernel1] "+w"(_kernel[1]),
        [_kernel2] "+w"(_kernel[2])
      : [filter] "r"(filter_ptr)
      : "memory", "q5", "q6", "q7", "q8", "q9", "r0");
  int valid_cols = w_end - w_start;
  for (int h = 0; h < output_tiles * 6; h += 6) {
    int32_t *output0 = output + output_offset;
    int32_t *output1 = output0 + output_w;
    int32_t *output2 = output1 + output_w;
    int32_t *output3 = output2 + output_w;
    int32_t *output4 = output3 + output_w;
    int32_t *output5 = output4 + output_w;
    Depth3x3ValidColLoadInput<Stride_w>(input + input_offset, input_w,
                                        valid_cols, _input[0], _input[1],
                                        _input[2]);
    _sum0 = veorq_s32(_sum0, _sum0);
    _sum1 = veorq_s32(_sum1, _sum1);
    for (int w_in = 0; w_in < valid_cols; ++w_in) {
      int index = w_in + w_start - w_in_start;
      _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_input[w_in][0]),
                             _kernel[index], 0);
      _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_input[w_in][1]),
                             _kernel[index], 1);
      _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_input[w_in][2]),
                             _kernel[index], 2);
      _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_input[w_in][0]),
                             _kernel[index], 0);
      _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_input[w_in][1]),
                             _kernel[index], 1);
      _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_input[w_in][2]),
                             _kernel[index], 2);
    }
    vst1q_lane_s32(output0, _sum0, 0);
    vst1q_lane_s32(output1, _sum0, 1);
    vst1q_lane_s32(output2, _sum0, 2);
    vst1q_lane_s32(output3, _sum0, 3);
    vst1q_lane_s32(output4, _sum1, 0);
    vst1q_lane_s32(output5, _sum1, 1);
    input_offset += 6 * Stride_h * input_w;
    output_offset += 6 * output_w;
  }
#endif
  for (int h = remain_start; h < h_output_end; ++h) {
    int32_t value = 0;
    const int h_in_start = -padding_h + h * Stride_h;
    for (int i = 0; i < 3; ++i) {
      for (int w_in = w_start; w_in < w_end; ++w_in) {
        value += filter[i * 3 + (w_in - w_in_start)] *
                 input[(h_in_start + i) * input_w + w_in];
      }
    }
    output[h * output_w + w_output] = value;
  }
}

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

template <int Stride>
inline void Depth3x3NormalRowLoadInput(const int8_t *input,
                                       int16x8_t &y0,    // NOLINT
                                       int16x8_t &y1,    // NOLINT
                                       int16x8_t &y2) {  // NOLINT
  PADDLE_MOBILE_THROW_EXCEPTION("Stride %d is not supported.", Stride);
}

template <>
inline void Depth3x3NormalRowLoadInput<1>(const int8_t *input,
                                          int16x8_t &y0,    // NOLINT
                                          int16x8_t &y1,    // NOLINT
                                          int16x8_t &y2) {  // NOLINT
  int8x8_t x0 = vld1_s8(input);
  y0 = vmovl_s8(x0);
  y1 = vextq_s16(y0, y0, 1);
  y2 = vextq_s16(y1, y1, 1);
}

template <>
inline void Depth3x3NormalRowLoadInput<2>(const int8_t *input,
                                          int16x8_t &y0,    // NOLINT
                                          int16x8_t &y1,    // NOLINT
                                          int16x8_t &y2) {  // NOLINT
  int8x8x2_t x0 = vld2_s8(input);
  y0 = vmovl_s8(x0.val[0]);
  y1 = vmovl_s8(x0.val[1]);
  y2 = vextq_s16(y0, y0, 1);
}

template <int Stride_h, int Stride_w>
inline void DepthwiseConv3x3NormalRow(const int8_t *input, const int8_t *filter,
                                      const int h_output, const int input_h,
                                      const int input_w, const int padding_h,
                                      const int padding_w, const int output_w,
                                      int32_t *output) {
  const int h_in_start = -padding_h + h_output * Stride_h;
  const int h_in_end = h_in_start + 3;
  const int h_start = h_in_start > 0 ? h_in_start : 0;
  const int h_end = h_in_end < input_h ? h_in_end : input_h;

  int valid_w_start = (padding_w + Stride_w - 1) / Stride_w;
  int valid_w_end = output_w - valid_w_start;

  int32_t *output_ptr = output + h_output * output_w;
  // border left
  DEPTHWISE_CONV_NORMAL_BORDER(0, valid_w_start)
  // middle
  int remain_start = valid_w_start;
#ifdef __ARM_NEON__
  int output_tiles = (valid_w_end - valid_w_start) / 6;
  remain_start = valid_w_start + output_tiles * 6;
  int32x4_t _sum0, _sum1;
  int16x8_t y0, y1, y2;
  int16x4_t _kernel[3];
  for (int h_in = h_start; h_in < h_end; ++h_in) {
    int index = h_in - h_in_start;
    int8x8_t w0 = vld1_s8(filter + index * 3);
    int16x8_t w1 = vmovl_s8(w0);
    _kernel[index] = vget_low_s16(w1);
  }
  for (int w = 0; w < output_tiles * 6; w += 6) {
    _sum0 = veorq_s32(_sum0, _sum0);
    _sum1 = veorq_s32(_sum1, _sum1);
    int output_offset = valid_w_start + w;
    int input_w_offset = output_offset * Stride_w - padding_w;
    for (int h_in = h_start; h_in < h_end; ++h_in) {
      int index = h_in - h_in_start;
      Depth3x3NormalRowLoadInput<Stride_w>(
          input + h_in * input_w + input_w_offset, y0, y1, y2);
      _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(y0), _kernel[index], 0);
      _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(y1), _kernel[index], 1);
      _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(y2), _kernel[index], 2);
      _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(y0), _kernel[index], 0);
      _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(y1), _kernel[index], 1);
      _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(y2), _kernel[index], 2);
    }
    vst1q_s32(output_ptr + output_offset, _sum0);
    vst1q_lane_s32(output_ptr + output_offset + 4, _sum1, 0);
    vst1q_lane_s32(output_ptr + output_offset + 5, _sum1, 1);
  }
#endif
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

// template<>
// void DepthwiseConv3x3<int8_t, int32_t>(
//     const framework::Tensor *input, const framework::Tensor *filter,
//     const std::vector<int> &strides, framework::Tensor *output) {
//   PADDLE_MOBILE_THROW_EXCEPTION(
//       "Depthwise conv with generic strides has not been implemented.");
// }

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
    // top
    for (int h = 0; h < valid_h_start; ++h) {
      DepthwiseConv3x3NormalRow<1, 1>(input_ptr, filter_ptr, h, input_h,
                                      input_w, padding_h, padding_w, output_w,
                                      output_ptr);
    }
    // left
    for (int w = 0; w < valid_w_start; ++w) {
      DepthwiseConv3x3ValidCol<1, 1>(
          input_ptr, filter_ptr, valid_h_start, valid_h_end, w, input_h,
          input_w, padding_h, padding_w, output_w, output_ptr);
    }
    // right
    for (int w = valid_w_end; w < output_w; ++w) {
      DepthwiseConv3x3ValidCol<1, 1>(
          input_ptr, filter_ptr, valid_h_start, valid_h_end, w, input_h,
          input_w, padding_h, padding_w, output_w, output_ptr);
    }
    // bottom
    for (int h = valid_h_end; h < output_h; ++h) {
      DepthwiseConv3x3NormalRow<1, 1>(input_ptr, filter_ptr, h, input_h,
                                      input_w, padding_h, padding_w, output_w,
                                      output_ptr);
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
      int32_t *output_ptr0 = output_ptr + h * output_w + valid_w_start;
      int32_t *output_ptr1 = output_ptr0 + output_w;
      int32_t *output_ptr2 = output_ptr1 + output_w;
      int32_t *output_ptr3 = output_ptr2 + output_w;
      int loop = output_w_tiles;
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
          "end_%=:                          \n"
          : [output_ptr0] "+r"(output_ptr0), [output_ptr1] "+r"(output_ptr1),
            [output_ptr2] "+r"(output_ptr2), [output_ptr3] "+r"(output_ptr3),
            [input_ptr0] "+r"(input_ptr0), [input_ptr1] "+r"(input_ptr1),
            [input_ptr2] "+r"(input_ptr2), [input_ptr3] "+r"(input_ptr3),
            [input_ptr4] "+r"(input_ptr4), [input_ptr5] "+r"(input_ptr5),
            [loop] "+r"(loop)
          : [remain] "r"(output_w_remain)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "r0");
    }
    // remain height
    int start_h = valid_h_start + (valid_h & 0xFFFC);
    for (int h = start_h; h < valid_h_end - 1; h += 2) {
      const int8_t *input_ptr0 = input_ptr + (h - padding_h) * input_w;
      const int8_t *input_ptr1 = input_ptr0 + input_w;
      const int8_t *input_ptr2 = input_ptr1 + input_w;
      const int8_t *input_ptr3 = input_ptr2 + input_w;
      int32_t *output_ptr0 = output_ptr + h * output_w + valid_w_start;
      int32_t *output_ptr1 = output_ptr0 + output_w;
      int loop = output_w_tiles;
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
          "cmp        %[loop], #0           \n"
          "ble        start_remain_%=       \n"
          "mov        r0, #6                \n"
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
          "end_%=:                           \n"
          : [output_ptr0] "+r"(output_ptr0), [output_ptr1] "+r"(output_ptr1),
            [input_ptr0] "+r"(input_ptr0), [input_ptr1] "+r"(input_ptr1),
            [input_ptr2] "+r"(input_ptr2), [input_ptr3] "+r"(input_ptr3),
            [loop] "+r"(loop)
          : [remain] "r"(output_w_remain)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
            "q8", "q9", "q10", "q11", "q12", "q13", "r0");
    }

    start_h = valid_h_start + (valid_h & 0xFFFE);
    if (start_h < valid_h_end) {
      const int8_t *input_ptr0 = input_ptr + (start_h - padding_h) * input_w;
      const int8_t *input_ptr1 = input_ptr0 + input_w;
      const int8_t *input_ptr2 = input_ptr1 + input_w;
      int32_t *output_ptr0 = output_ptr + start_h * output_w + valid_w_start;
      int loop = output_w_tiles;
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
          "cmp        %[loop], #0           \n"
          "ble        start_remain_%=       \n"
          "mov        r0, #6                \n"
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
          "end_%=:                           \n"
          : [output_ptr0] "+r"(output_ptr0), [input_ptr0] "+r"(input_ptr0),
            [input_ptr1] "+r"(input_ptr1), [input_ptr2] "+r"(input_ptr2),
            [loop] "+r"(loop)
          : [remain] "r"(output_w_remain)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
            "q8", "q9", "q10", "q11", "r0");
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
  int valid_h_end = output_h - valid_h_start;
  int valid_h = valid_h_end - valid_h_start;
  int valid_w_start = (padding_w + 1) / 2;
  int valid_w_end = output_w - valid_w_start;
  int valid_w = valid_w_end - valid_w_start;

  //  DLOG << "valid_h_start: " << valid_h_start;
  //  DLOG << "valid_h_end: " << valid_h_end;
  //  DLOG << "valid_w_start: " << valid_w_start;
  //  DLOG << "valid_w_end: " << valid_w_end;

  #pragma omp parallel for
  for (int g = 0; g < input.dims()[1]; ++g) {
    const int8_t *input_ptr = input_data + g * image_size;
    const int8_t *filter_ptr = filter_data + g * 9;
    int32_t *output_ptr = out_data + g * out_image_size;
    // top
    for (int h = 0; h < valid_h_start; ++h) {
      DepthwiseConv3x3NormalRow<2, 2>(input_ptr, filter_ptr, h, input_h,
                                      input_w, padding_h, padding_w, output_w,
                                      output_ptr);
    }
    // left
    for (int w = 0; w < valid_w_start; ++w) {
      DepthwiseConv3x3ValidCol<2, 2>(
          input_ptr, filter_ptr, valid_h_start, valid_h_end, w, input_h,
          input_w, padding_h, padding_w, output_w, output_ptr);
    }
    // right
    for (int w = valid_w_end; w < output_w; ++w) {
      DepthwiseConv3x3ValidCol<2, 2>(
          input_ptr, filter_ptr, valid_h_start, valid_h_end, w, input_h,
          input_w, padding_h, padding_w, output_w, output_ptr);
    }
    // bottom
    for (int h = valid_h_end; h < output_h; ++h) {
      DepthwiseConv3x3NormalRow<2, 2>(input_ptr, filter_ptr, h, input_h,
                                      input_w, padding_h, padding_w, output_w,
                                      output_ptr);
    }
    // valid
    int input_w_start = 2 * valid_w_start - padding_w;
    int output_w_tiles = valid_w / 6;
    int output_w_remain = valid_w - output_w_tiles * 6;
    for (int h = valid_h_start; h < valid_h_end - 2; h += 3) {
      size_t offset = (2 * h - padding_h) * input_w + input_w_start;
      const int8_t *input_ptr0 = input_ptr + offset;
      const int8_t *input_ptr1 = input_ptr0 + input_w;
      const int8_t *input_ptr2 = input_ptr1 + input_w;
      const int8_t *input_ptr3 = input_ptr2 + input_w;
      const int8_t *input_ptr4 = input_ptr3 + input_w;
      const int8_t *input_ptr5 = input_ptr4 + input_w;
      const int8_t *input_ptr6 = input_ptr5 + input_w;
      int32_t *output_ptr0 = output_ptr + h * output_w + valid_w_start;
      int32_t *output_ptr1 = output_ptr0 + output_w;
      int32_t *output_ptr2 = output_ptr1 + output_w;
      int loop = output_w_tiles;
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
          "cmp        %[loop], #0           \n"
          "ble        start_remain_%=       \n"
          "mov        r0, #12               \n"
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
          "end_%=:                          \n"
          : [output_ptr0] "+r"(output_ptr0), [output_ptr1] "+r"(output_ptr1),
            [output_ptr2] "+r"(output_ptr2), [input_ptr6] "+r"(input_ptr6),
            [input_ptr0] "+r"(input_ptr0), [input_ptr1] "+r"(input_ptr1),
            [input_ptr2] "+r"(input_ptr2), [input_ptr3] "+r"(input_ptr3),
            [input_ptr4] "+r"(input_ptr4), [input_ptr5] "+r"(input_ptr5),
            [loop] "+r"(loop)
          : [remain] "r"(output_w_remain)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "r0");
    }

    int start_h = valid_h_start + valid_h / 3 * 3;
    for (int h = start_h; h < valid_h_end; ++h) {
      size_t offset = (2 * h - padding_h) * input_w + input_w_start;
      const int8_t *input_ptr0 = input_ptr + offset;
      const int8_t *input_ptr1 = input_ptr0 + input_w;
      const int8_t *input_ptr2 = input_ptr1 + input_w;
      int32_t *output_ptr0 = output_ptr + h * output_w + valid_w_start;
      int loop = output_w_tiles;
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
          "cmp        %[loop], #0           \n"
          "ble        start_remain_%=       \n"
          "mov        r0, #12               \n"
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
          "end_%=:                          \n"
          : [output_ptr0] "+r"(output_ptr0), [input_ptr0] "+r"(input_ptr0),
            [input_ptr1] "+r"(input_ptr1), [input_ptr2] "+r"(input_ptr2),
            [loop] "+r"(loop)
          : [remain] "r"(output_w_remain)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
            "q8", "q9", "q10", "q11", "q12", "r0");
    }
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif
