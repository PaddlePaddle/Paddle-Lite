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

#ifdef POOL_OP

#include "operators/math/pooling.h"
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif  // __ARM_NEON

namespace paddle_mobile {
namespace operators {
namespace math {

#define POOLING3X3_NORMAL_BORDER(start, end)                   \
  for (int w = start; w < end; ++w) {                          \
    const int w_in_start = -padding_w + w * Stride;            \
    const int w_in_end = w_in_start + 3;                       \
    const int w_start = w_in_start > 0 ? w_in_start : 0;       \
    const int w_end = w_in_end < input_w ? w_in_end : input_w; \
    PoolingVal<P> val;                                         \
    for (int h_in = h_start; h_in < h_end; ++h_in) {           \
      for (int w_in = w_start; w_in < w_end; ++w_in) {         \
        val += input[h_in * input_w + w_in];                   \
      }                                                        \
    }                                                          \
    output_ptr[w] = val.Value();                               \
  }

template <PoolingType P, int Stride>
inline void Pooling3x3ValidCol(const float *input, const int h_output,
                               const int h_output_end, const int w_output,
                               const int input_h, const int input_w,
                               const int padding_h, const int padding_w,
                               const int output_w, float *output) {
  const int w_in_start = -padding_w + w_output * Stride;
  const int w_in_end = w_in_start + 3;
  const int w_start = w_in_start > 0 ? w_in_start : 0;
  const int w_end = w_in_end < input_w ? w_in_end : input_w;
  for (int h = h_output; h < h_output_end; ++h) {
    PoolingVal<P> val;
    const int h_in_start = -padding_h + h * Stride;
    for (int i = 0; i < 3; ++i) {
      for (int w_in = w_start; w_in < w_end; ++w_in) {
        val += input[(h_in_start + i) * input_w + w_in];
      }
    }
    output[h * output_w + w_output] = val.Value();
  }
}

template <PoolingType P, int Stride>
inline void Pooling3x3NormalRow(const float *input, const int h_output,
                                const int input_h, const int input_w,
                                const int padding_h, const int padding_w,
                                const int output_w, float *output) {
  const int h_in_start = -padding_h + h_output * Stride;
  const int h_in_end = h_in_start + 3;
  const int h_start = h_in_start > 0 ? h_in_start : 0;
  const int h_end = h_in_end < input_h ? h_in_end : input_h;

  int valid_w_start = (padding_w + Stride - 1) / Stride;
  int valid_w_end = output_w - valid_w_start;

  float *output_ptr = output + h_output * output_w;
  // border left
  POOLING3X3_NORMAL_BORDER(0, valid_w_start)
  // middle
  for (int w = valid_w_start; w < valid_w_end; ++w) {
    PoolingVal<P> val;
    int input_start = -padding_w + w * Stride;
    for (int h_in = h_start; h_in < h_end; ++h_in) {
      for (int j = 0; j < 3; ++j) {
        val += input[h_in * input_w + j + input_start];
      }
    }
    output_ptr[w] = val.Value();
  }
  // border right
  POOLING3X3_NORMAL_BORDER(valid_w_end, output_w)
}

template <PoolingType P>
struct Pooling3x3<P, 1> {
  inline void operator()(const framework::Tensor &input,
                         const std::vector<int> &paddings,
                         framework::Tensor *output) {
    const float *input_data = input.data<float>();
    float *output_data = output->mutable_data<float>();
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
    for (int c = 0; c < output->dims()[1]; ++c) {
      const float *input_ptr = input_data + c * image_size;
      float *output_ptr = output_data + c * out_image_size;
      // top
      for (int h = 0; h < valid_h_start; ++h) {
        Pooling3x3NormalRow<P, 1>(input_ptr, h, input_h, input_w, padding_h,
                                  padding_w, output_w, output_ptr);
      }
      // left
      for (int w = 0; w < valid_w_start; ++w) {
        Pooling3x3ValidCol<P, 1>(input_ptr, valid_h_start, valid_h_end, w,
                                 input_h, input_w, padding_h, padding_w,
                                 output_w, output_ptr);
      }
      // right
      for (int w = valid_w_end; w < output_w; ++w) {
        Pooling3x3ValidCol<P, 1>(input_ptr, valid_h_start, valid_h_end, w,
                                 input_h, input_w, padding_h, padding_w,
                                 output_w, output_ptr);
      }
      // bottom
      for (int h = valid_h_end; h < output_h; ++h) {
        Pooling3x3NormalRow<P, 1>(input_ptr, h, input_h, input_w, padding_h,
                                  padding_w, output_w, output_ptr);
      }
      // valid
      int output_w_tiles = valid_w / 6;
      int output_w_remain = valid_w - output_w_tiles * 6;
      for (int h = valid_h_start; h < valid_h_end - 3; h += 4) {
        const float *input_ptr0 = input_ptr + (h - padding_h) * input_w;
        const float *input_ptr1 = input_ptr0 + input_w;
        const float *input_ptr2 = input_ptr1 + input_w;
        const float *input_ptr3 = input_ptr2 + input_w;
        const float *input_ptr4 = input_ptr3 + input_w;
        const float *input_ptr5 = input_ptr4 + input_w;
        float *output_ptr0 = output_ptr + h * output_w + valid_w_start;
        float *output_ptr1 = output_ptr0 + output_w;
        float *output_ptr2 = output_ptr1 + output_w;
        float *output_ptr3 = output_ptr2 + output_w;
        int remain = output_w_remain;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
        float32x4x2_t x0, x1, x2;
        float32x4x2_t y0, y1, y2;
        for (int loop = 0; loop < output_w_tiles; ++loop) {
          x0.val[0] = vld1q_f32(input_ptr0);
          x0.val[1] = vld1q_f32(input_ptr0 + 4);
          x1.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
          x1.val[1] = vextq_f32(x0.val[1], x0.val[1], 1);
          x2.val[0] = vextq_f32(x0.val[0], x0.val[1], 2);
          x2.val[1] = vextq_f32(x0.val[1], x0.val[1], 2);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x1.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x1.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y0.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);

          x0.val[0] = vld1q_f32(input_ptr1);
          x0.val[1] = vld1q_f32(input_ptr1 + 4);
          x1.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
          x1.val[1] = vextq_f32(x0.val[1], x0.val[1], 1);
          x2.val[0] = vextq_f32(x0.val[0], x0.val[1], 2);
          x2.val[1] = vextq_f32(x0.val[1], x0.val[1], 2);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x1.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x1.val[1]);
          y1.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y1.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(y1.val[0], y0.val[0]);
          y0.val[1] = vPoolPreq_f32<P>(y1.val[1], y0.val[1]);

          x0.val[0] = vld1q_f32(input_ptr2);
          x0.val[1] = vld1q_f32(input_ptr2 + 4);
          x1.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
          x1.val[1] = vextq_f32(x0.val[1], x0.val[1], 1);
          x2.val[0] = vextq_f32(x0.val[0], x0.val[1], 2);
          x2.val[1] = vextq_f32(x0.val[1], x0.val[1], 2);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x1.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x1.val[1]);
          y2.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y2.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);
          y1.val[0] = vPoolPreq_f32<P>(y2.val[0], y1.val[0]);
          y1.val[1] = vPoolPreq_f32<P>(y2.val[1], y1.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(y2.val[0], y0.val[0]);
          y0.val[1] = vPoolPreq_f32<P>(y2.val[1], y0.val[1]);
          y0.val[0] = vPoolPostq_f32<P>(y0.val[0]);
          y0.val[1] = vPoolPostq_f32<P>(y0.val[1]);
          vst1q_f32(output_ptr0, y0.val[0]);
          vst1_f32(output_ptr0 + 4, vget_low_f32(y0.val[1]));

          x0.val[0] = vld1q_f32(input_ptr3);
          x0.val[1] = vld1q_f32(input_ptr3 + 4);
          x1.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
          x1.val[1] = vextq_f32(x0.val[1], x0.val[1], 1);
          x2.val[0] = vextq_f32(x0.val[0], x0.val[1], 2);
          x2.val[1] = vextq_f32(x0.val[1], x0.val[1], 2);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x1.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x1.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y0.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);
          y1.val[0] = vPoolPreq_f32<P>(y0.val[0], y1.val[0]);
          y1.val[1] = vPoolPreq_f32<P>(y0.val[1], y1.val[1]);
          y2.val[0] = vPoolPreq_f32<P>(y0.val[0], y2.val[0]);
          y2.val[1] = vPoolPreq_f32<P>(y0.val[1], y2.val[1]);
          y1.val[0] = vPoolPostq_f32<P>(y1.val[0]);
          y1.val[1] = vPoolPostq_f32<P>(y1.val[1]);
          vst1q_f32(output_ptr1, y1.val[0]);
          vst1_f32(output_ptr1 + 4, vget_low_f32(y1.val[1]));

          x0.val[0] = vld1q_f32(input_ptr4);
          x0.val[1] = vld1q_f32(input_ptr4 + 4);
          x1.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
          x1.val[1] = vextq_f32(x0.val[1], x0.val[1], 1);
          x2.val[0] = vextq_f32(x0.val[0], x0.val[1], 2);
          x2.val[1] = vextq_f32(x0.val[1], x0.val[1], 2);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x1.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x1.val[1]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], y0.val[0]);
          y0.val[1] = vPoolPreq_f32<P>(x0.val[1], y0.val[1]);
          y2.val[0] = vPoolPreq_f32<P>(x0.val[0], y2.val[0]);
          y2.val[1] = vPoolPreq_f32<P>(x0.val[1], y2.val[1]);
          y2.val[0] = vPoolPostq_f32<P>(y2.val[0]);
          y2.val[1] = vPoolPostq_f32<P>(y2.val[1]);
          vst1q_f32(output_ptr2, y2.val[0]);
          vst1_f32(output_ptr2 + 4, vget_low_f32(y2.val[1]));

          x0.val[0] = vld1q_f32(input_ptr5);
          x0.val[1] = vld1q_f32(input_ptr5 + 4);
          x1.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
          x1.val[1] = vextq_f32(x0.val[1], x0.val[1], 1);
          x2.val[0] = vextq_f32(x0.val[0], x0.val[1], 2);
          x2.val[1] = vextq_f32(x0.val[1], x0.val[1], 2);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x1.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x1.val[1]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], y0.val[0]);
          y0.val[1] = vPoolPreq_f32<P>(x0.val[1], y0.val[1]);
          y0.val[0] = vPoolPostq_f32<P>(y0.val[0]);
          y0.val[1] = vPoolPostq_f32<P>(y0.val[1]);
          vst1q_f32(output_ptr3, y0.val[0]);
          vst1_f32(output_ptr3 + 4, vget_low_f32(y0.val[1]));

          input_ptr0 += 6;
          input_ptr1 += 6;
          input_ptr2 += 6;
          input_ptr3 += 6;
          input_ptr4 += 6;
          input_ptr5 += 6;
          output_ptr0 += 6;
          output_ptr1 += 6;
          output_ptr2 += 6;
          output_ptr3 += 6;
        }
        // remain w
        if (remain >= 4) {
          x0.val[0] = vld1q_f32(input_ptr0);
          x0.val[1] = vld1q_f32(input_ptr0 + 4);
          x1.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
          x2.val[0] = vextq_f32(x0.val[0], x0.val[1], 2);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x1.val[0]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);

          x0.val[0] = vld1q_f32(input_ptr1);
          x0.val[1] = vld1q_f32(input_ptr1 + 4);
          x1.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
          x2.val[0] = vextq_f32(x0.val[0], x0.val[1], 2);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x1.val[0]);
          y1.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y0.val[0] = vPoolPreq_f32<P>(y1.val[0], y0.val[0]);

          x0.val[0] = vld1q_f32(input_ptr2);
          x0.val[1] = vld1q_f32(input_ptr2 + 4);
          x1.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
          x2.val[0] = vextq_f32(x0.val[0], x0.val[1], 2);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x1.val[0]);
          y2.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y1.val[0] = vPoolPreq_f32<P>(y2.val[0], y1.val[0]);
          y0.val[0] = vPoolPreq_f32<P>(y2.val[0], y0.val[0]);
          y0.val[0] = vPoolPostq_f32<P>(y0.val[0]);
          vst1q_f32(output_ptr0, y0.val[0]);

          x0.val[0] = vld1q_f32(input_ptr3);
          x0.val[1] = vld1q_f32(input_ptr3 + 4);
          x1.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
          x2.val[0] = vextq_f32(x0.val[0], x0.val[1], 2);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x1.val[0]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y1.val[0] = vPoolPreq_f32<P>(y0.val[0], y1.val[0]);
          y2.val[0] = vPoolPreq_f32<P>(y0.val[0], y2.val[0]);
          y1.val[0] = vPoolPostq_f32<P>(y1.val[0]);
          vst1q_f32(output_ptr1, y1.val[0]);

          x0.val[0] = vld1q_f32(input_ptr4);
          x0.val[1] = vld1q_f32(input_ptr4 + 4);
          x1.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
          x2.val[0] = vextq_f32(x0.val[0], x0.val[1], 2);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x1.val[0]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], y0.val[0]);
          y2.val[0] = vPoolPreq_f32<P>(x0.val[0], y2.val[0]);
          y2.val[0] = vPoolPostq_f32<P>(y2.val[0]);
          vst1q_f32(output_ptr2, y2.val[0]);

          x0.val[0] = vld1q_f32(input_ptr5);
          x0.val[1] = vld1q_f32(input_ptr5 + 4);
          x1.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
          x2.val[0] = vextq_f32(x0.val[0], x0.val[1], 2);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x1.val[0]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], y0.val[0]);
          y0.val[0] = vPoolPostq_f32<P>(y0.val[0]);
          vst1q_f32(output_ptr3, y0.val[0]);

          input_ptr0 += 4;
          input_ptr1 += 4;
          input_ptr2 += 4;
          input_ptr3 += 4;
          input_ptr4 += 4;
          input_ptr5 += 4;
          output_ptr0 += 4;
          output_ptr1 += 4;
          output_ptr2 += 4;
          output_ptr3 += 4;
          remain -= 4;
        }
#endif  // __ARM_NEON__
        for (int r = 0; r < remain; ++r) {
          float m0 = PoolPre<P>(input_ptr0[r], input_ptr0[r + 1]);
          m0 = PoolPre<P>(m0, input_ptr0[r + 2]);
          float m1 = PoolPre<P>(input_ptr1[r], input_ptr1[r + 1]);
          m1 = PoolPre<P>(m1, input_ptr1[r + 2]);
          float m2 = PoolPre<P>(input_ptr2[r], input_ptr2[r + 1]);
          m2 = PoolPre<P>(m2, input_ptr2[r + 2]);
          float m3 = PoolPre<P>(input_ptr3[r], input_ptr3[r + 1]);
          m3 = PoolPre<P>(m3, input_ptr3[r + 2]);
          float m4 = PoolPre<P>(input_ptr4[r], input_ptr4[r + 1]);
          m4 = PoolPre<P>(m4, input_ptr4[r + 2]);
          float m5 = PoolPre<P>(input_ptr5[r], input_ptr5[r + 1]);
          m5 = PoolPre<P>(m5, input_ptr5[r + 2]);

          m0 = PoolPre<P>(PoolPre<P>(m0, m1), m2);
          m1 = PoolPre<P>(PoolPre<P>(m1, m2), m3);
          m2 = PoolPre<P>(PoolPre<P>(m2, m3), m4);
          m3 = PoolPre<P>(PoolPre<P>(m3, m4), m5);
          output_ptr0[r] = PoolPost<P>(m0);
          output_ptr1[r] = PoolPost<P>(m1);
          output_ptr2[r] = PoolPost<P>(m2);
          output_ptr3[r] = PoolPost<P>(m3);
        }
      }
      // remain h
      int start_h = valid_h_start + (valid_h & 0xFFFC);
      for (int h = start_h; h < valid_h_end; ++h) {
        const float *input_ptr0 = input_ptr + (h - padding_h) * input_w;
        const float *input_ptr1 = input_ptr0 + input_w;
        const float *input_ptr2 = input_ptr1 + input_w;
        float *output_ptr0 = output_ptr + h * output_w + valid_w_start;
        int remain = output_w_remain;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
        float32x4x2_t x0, x1, x2, y0;
        for (int loop = 0; loop < output_w_tiles; ++loop) {
          x0.val[0] = vld1q_f32(input_ptr0);
          x0.val[1] = vld1q_f32(input_ptr0 + 4);
          x1.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
          x1.val[1] = vextq_f32(x0.val[1], x0.val[1], 1);
          x2.val[0] = vextq_f32(x0.val[0], x0.val[1], 2);
          x2.val[1] = vextq_f32(x0.val[1], x0.val[1], 2);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x1.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x1.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y0.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);

          x0.val[0] = vld1q_f32(input_ptr1);
          x0.val[1] = vld1q_f32(input_ptr1 + 4);
          x1.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
          x1.val[1] = vextq_f32(x0.val[1], x0.val[1], 1);
          x2.val[0] = vextq_f32(x0.val[0], x0.val[1], 2);
          x2.val[1] = vextq_f32(x0.val[1], x0.val[1], 2);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x1.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x1.val[1]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], y0.val[0]);
          y0.val[1] = vPoolPreq_f32<P>(x0.val[1], y0.val[1]);

          x0.val[0] = vld1q_f32(input_ptr2);
          x0.val[1] = vld1q_f32(input_ptr2 + 4);
          x1.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
          x1.val[1] = vextq_f32(x0.val[1], x0.val[1], 1);
          x2.val[0] = vextq_f32(x0.val[0], x0.val[1], 2);
          x2.val[1] = vextq_f32(x0.val[1], x0.val[1], 2);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x1.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x1.val[1]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], y0.val[0]);
          y0.val[1] = vPoolPreq_f32<P>(x0.val[1], y0.val[1]);
          y0.val[0] = vPoolPostq_f32<P>(y0.val[0]);
          y0.val[1] = vPoolPostq_f32<P>(y0.val[1]);
          vst1q_f32(output_ptr0, y0.val[0]);
          vst1_f32(output_ptr0 + 4, vget_low_f32(y0.val[1]));

          input_ptr0 += 6;
          input_ptr1 += 6;
          input_ptr2 += 6;
          output_ptr0 += 6;
        }
        // remain w
        if (remain >= 4) {
          x0.val[0] = vld1q_f32(input_ptr0);
          x0.val[1] = vld1q_f32(input_ptr0 + 4);
          x1.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
          x2.val[0] = vextq_f32(x0.val[0], x0.val[1], 2);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x1.val[0]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);

          x0.val[0] = vld1q_f32(input_ptr1);
          x0.val[1] = vld1q_f32(input_ptr1 + 4);
          x1.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
          x2.val[0] = vextq_f32(x0.val[0], x0.val[1], 2);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x1.val[0]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], y0.val[0]);

          x0.val[0] = vld1q_f32(input_ptr2);
          x0.val[1] = vld1q_f32(input_ptr2 + 4);
          x1.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
          x2.val[0] = vextq_f32(x0.val[0], x0.val[1], 2);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x1.val[0]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], y0.val[0]);
          y0.val[0] = vPoolPostq_f32<P>(y0.val[0]);
          vst1q_f32(output_ptr0, y0.val[0]);

          input_ptr0 += 4;
          input_ptr1 += 4;
          input_ptr2 += 4;
          output_ptr0 += 4;
          remain -= 4;
        }
#endif  // __ARM_NEON__
        for (int r = 0; r < remain; ++r) {
          float m0 = PoolPre<P>(input_ptr0[r], input_ptr0[r + 1]);
          m0 = PoolPre<P>(m0, input_ptr0[r + 2]);
          float m1 = PoolPre<P>(input_ptr1[r], input_ptr1[r + 1]);
          m1 = PoolPre<P>(m1, input_ptr1[r + 2]);
          float m2 = PoolPre<P>(input_ptr2[r], input_ptr2[r + 1]);
          m2 = PoolPre<P>(m2, input_ptr2[r + 2]);

          m0 = PoolPre<P>(PoolPre<P>(m0, m1), m2);
          output_ptr0[r] = PoolPost<P>(m0);
        }
      }
    }
  }
};

template <PoolingType P>
struct Pooling3x3<P, 2> {
  inline void operator()(const framework::Tensor &input,
                         const std::vector<int> &paddings,
                         framework::Tensor *output) {
    const float *input_data = input.data<float>();
    float *output_data = output->mutable_data<float>();
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

    #pragma omp parallel for
    for (int c = 0; c < output->dims()[1]; ++c) {
      const float *input_ptr = input_data + c * image_size;
      float *output_ptr = output_data + c * out_image_size;
      // top
      for (int h = 0; h < valid_h_start; ++h) {
        Pooling3x3NormalRow<P, 2>(input_ptr, h, input_h, input_w, padding_h,
                                  padding_w, output_w, output_ptr);
      }
      // left
      for (int w = 0; w < valid_w_start; ++w) {
        Pooling3x3ValidCol<P, 2>(input_ptr, valid_h_start, valid_h_end, w,
                                 input_h, input_w, padding_h, padding_w,
                                 output_w, output_ptr);
      }
      // right
      for (int w = valid_w_end; w < output_w; ++w) {
        Pooling3x3ValidCol<P, 2>(input_ptr, valid_h_start, valid_h_end, w,
                                 input_h, input_w, padding_h, padding_w,
                                 output_w, output_ptr);
      }
      // bottom
      for (int h = valid_h_end; h < output_h; ++h) {
        Pooling3x3NormalRow<P, 2>(input_ptr, h, input_h, input_w, padding_h,
                                  padding_w, output_w, output_ptr);
      }
      // valid
      int input_w_start = 2 * valid_w_start - padding_w;
      int output_w_tiles = valid_w / 6;
      int output_w_remain = valid_w - output_w_tiles * 6;
      for (int h = valid_h_start; h < valid_h_end - 2; h += 3) {
        size_t offset = (2 * h - padding_h) * input_w + input_w_start;
        const float *input_ptr0 = input_ptr + offset;
        const float *input_ptr1 = input_ptr0 + input_w;
        const float *input_ptr2 = input_ptr1 + input_w;
        const float *input_ptr3 = input_ptr2 + input_w;
        const float *input_ptr4 = input_ptr3 + input_w;
        const float *input_ptr5 = input_ptr4 + input_w;
        const float *input_ptr6 = input_ptr5 + input_w;
        float *output_ptr0 = output_ptr + h * output_w + valid_w_start;
        float *output_ptr1 = output_ptr0 + output_w;
        float *output_ptr2 = output_ptr1 + output_w;
        int remain = output_w_remain;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
        float32x4x2_t x0, x1, x2;
        float32x4x2_t y0, y1, y2;
        for (int loop = 0; loop < output_w_tiles; ++loop) {
          x0 = vld2q_f32(input_ptr0);
          x1 = vld2q_f32(input_ptr0 + 8);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x2.val[1] = vextq_f32(x1.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          x0.val[1] = vPoolPreq_f32<P>(x1.val[0], x1.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y0.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);

          x0 = vld2q_f32(input_ptr1);
          x1 = vld2q_f32(input_ptr1 + 8);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x2.val[1] = vextq_f32(x1.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          x0.val[1] = vPoolPreq_f32<P>(x1.val[0], x1.val[1]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], y0.val[0]);
          y0.val[1] = vPoolPreq_f32<P>(x0.val[1], y0.val[1]);

          x0 = vld2q_f32(input_ptr2);
          x1 = vld2q_f32(input_ptr2 + 8);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x2.val[1] = vextq_f32(x1.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          x0.val[1] = vPoolPreq_f32<P>(x1.val[0], x1.val[1]);
          y1.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y1.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(y1.val[0], y0.val[0]);
          y0.val[1] = vPoolPreq_f32<P>(y1.val[1], y0.val[1]);
          y0.val[0] = vPoolPostq_f32<P>(y0.val[0]);
          y0.val[1] = vPoolPostq_f32<P>(y0.val[1]);
          vst1q_f32(output_ptr0, y0.val[0]);
          vst1_f32(output_ptr0 + 4, vget_low_f32(y0.val[1]));

          x0 = vld2q_f32(input_ptr3);
          x1 = vld2q_f32(input_ptr3 + 8);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x2.val[1] = vextq_f32(x1.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          x0.val[1] = vPoolPreq_f32<P>(x1.val[0], x1.val[1]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);
          y1.val[0] = vPoolPreq_f32<P>(x0.val[0], y1.val[0]);
          y1.val[1] = vPoolPreq_f32<P>(x0.val[1], y1.val[1]);

          x0 = vld2q_f32(input_ptr4);
          x1 = vld2q_f32(input_ptr4 + 8);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x2.val[1] = vextq_f32(x1.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          x0.val[1] = vPoolPreq_f32<P>(x1.val[0], x1.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y0.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);
          y1.val[0] = vPoolPreq_f32<P>(y0.val[0], y1.val[0]);
          y1.val[1] = vPoolPreq_f32<P>(y0.val[1], y1.val[1]);
          y1.val[0] = vPoolPostq_f32<P>(y1.val[0]);
          y1.val[1] = vPoolPostq_f32<P>(y1.val[1]);
          vst1q_f32(output_ptr1, y1.val[0]);
          vst1_f32(output_ptr1 + 4, vget_low_f32(y1.val[1]));

          x0 = vld2q_f32(input_ptr5);
          x1 = vld2q_f32(input_ptr5 + 8);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x2.val[1] = vextq_f32(x1.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          x0.val[1] = vPoolPreq_f32<P>(x1.val[0], x1.val[1]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], y0.val[0]);
          y0.val[1] = vPoolPreq_f32<P>(x0.val[1], y0.val[1]);

          x0 = vld2q_f32(input_ptr6);
          x1 = vld2q_f32(input_ptr6 + 8);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x2.val[1] = vextq_f32(x1.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          x0.val[1] = vPoolPreq_f32<P>(x1.val[0], x1.val[1]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], y0.val[0]);
          y0.val[1] = vPoolPreq_f32<P>(x0.val[1], y0.val[1]);
          y0.val[0] = vPoolPostq_f32<P>(y0.val[0]);
          y0.val[1] = vPoolPostq_f32<P>(y0.val[1]);
          vst1q_f32(output_ptr2, y0.val[0]);
          vst1_f32(output_ptr2 + 4, vget_low_f32(y0.val[1]));

          input_ptr0 += 12;
          input_ptr1 += 12;
          input_ptr2 += 12;
          input_ptr3 += 12;
          input_ptr4 += 12;
          input_ptr5 += 12;
          input_ptr6 += 12;
          output_ptr0 += 6;
          output_ptr1 += 6;
          output_ptr2 += 6;
        }
        // remain w
        if (remain >= 4) {
          x0 = vld2q_f32(input_ptr0);
          x1.val[0] = vdupq_n_f32(input_ptr0[8]);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);

          x0 = vld2q_f32(input_ptr1);
          x1.val[0] = vdupq_n_f32(input_ptr1[8]);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], y0.val[0]);

          x0 = vld2q_f32(input_ptr2);
          x1.val[0] = vdupq_n_f32(input_ptr2[8]);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          y1.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y0.val[0] = vPoolPreq_f32<P>(y1.val[0], y0.val[0]);
          y0.val[0] = vPoolPostq_f32<P>(y0.val[0]);
          vst1q_f32(output_ptr0, y0.val[0]);

          x0 = vld2q_f32(input_ptr3);
          x1.val[0] = vdupq_n_f32(input_ptr3[8]);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y1.val[0] = vPoolPreq_f32<P>(x0.val[0], y1.val[0]);

          x0 = vld2q_f32(input_ptr4);
          x1.val[0] = vdupq_n_f32(input_ptr4[8]);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y1.val[0] = vPoolPreq_f32<P>(y0.val[0], y1.val[0]);
          y1.val[0] = vPoolPostq_f32<P>(y1.val[0]);
          vst1q_f32(output_ptr1, y1.val[0]);

          x0 = vld2q_f32(input_ptr5);
          x1.val[0] = vdupq_n_f32(input_ptr5[8]);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], y0.val[0]);

          x0 = vld2q_f32(input_ptr6);
          x1.val[0] = vdupq_n_f32(input_ptr6[8]);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], y0.val[0]);
          y0.val[0] = vPoolPostq_f32<P>(y0.val[0]);
          vst1q_f32(output_ptr2, y0.val[0]);

          input_ptr0 += 8;
          input_ptr1 += 8;
          input_ptr2 += 8;
          input_ptr3 += 8;
          input_ptr4 += 8;
          input_ptr5 += 8;
          input_ptr6 += 8;
          output_ptr0 += 4;
          output_ptr1 += 4;
          output_ptr2 += 4;
          remain -= 4;
        }
#endif  // __ARM_NEON__
        for (int r = 0; r < remain; ++r) {
          float m0 = PoolPre<P>(input_ptr0[2 * r], input_ptr0[2 * r + 1]);
          m0 = PoolPre<P>(m0, input_ptr0[2 * r + 2]);
          float m1 = PoolPre<P>(input_ptr1[2 * r], input_ptr1[2 * r + 1]);
          m1 = PoolPre<P>(m1, input_ptr1[2 * r + 2]);
          float m2 = PoolPre<P>(input_ptr2[2 * r], input_ptr2[2 * r + 1]);
          m2 = PoolPre<P>(m2, input_ptr2[2 * r + 2]);
          float m3 = PoolPre<P>(input_ptr3[2 * r], input_ptr3[2 * r + 1]);
          m3 = PoolPre<P>(m3, input_ptr3[2 * r + 2]);
          float m4 = PoolPre<P>(input_ptr4[2 * r], input_ptr4[2 * r + 1]);
          m4 = PoolPre<P>(m4, input_ptr4[2 * r + 2]);
          float m5 = PoolPre<P>(input_ptr5[2 * r], input_ptr5[2 * r + 1]);
          m5 = PoolPre<P>(m5, input_ptr5[2 * r + 2]);
          float m6 = PoolPre<P>(input_ptr6[2 * r], input_ptr6[2 * r + 1]);
          m6 = PoolPre<P>(m6, input_ptr6[2 * r + 2]);

          m0 = PoolPre<P>(PoolPre<P>(m0, m1), m2);
          m1 = PoolPre<P>(PoolPre<P>(m2, m3), m4);
          m2 = PoolPre<P>(PoolPre<P>(m4, m5), m6);
          output_ptr0[r] = PoolPost<P>(m0);
          output_ptr1[r] = PoolPost<P>(m1);
          output_ptr2[r] = PoolPost<P>(m2);
        }
      }
      // remain h
      int start_h = valid_h_start + valid_h / 3 * 3;
      for (int h = start_h; h < valid_h_end; ++h) {
        size_t offset = (2 * h - padding_h) * input_w + input_w_start;
        const float *input_ptr0 = input_ptr + offset;
        const float *input_ptr1 = input_ptr0 + input_w;
        const float *input_ptr2 = input_ptr1 + input_w;
        float *output_ptr0 = output_ptr + h * output_w + valid_w_start;
        int remain = output_w_remain;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
        float32x4x2_t x0, x1, x2, y0;
        for (int loop = 0; loop < output_w_tiles; ++loop) {
          x0 = vld2q_f32(input_ptr0);
          x1 = vld2q_f32(input_ptr0 + 8);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x2.val[1] = vextq_f32(x1.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          x0.val[1] = vPoolPreq_f32<P>(x1.val[0], x1.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y0.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);

          x0 = vld2q_f32(input_ptr1);
          x1 = vld2q_f32(input_ptr1 + 8);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x2.val[1] = vextq_f32(x1.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          x0.val[1] = vPoolPreq_f32<P>(x1.val[0], x1.val[1]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], y0.val[0]);
          y0.val[1] = vPoolPreq_f32<P>(x0.val[1], y0.val[1]);

          x0 = vld2q_f32(input_ptr2);
          x1 = vld2q_f32(input_ptr2 + 8);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x2.val[1] = vextq_f32(x1.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          x0.val[1] = vPoolPreq_f32<P>(x1.val[0], x1.val[1]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          x0.val[1] = vPoolPreq_f32<P>(x0.val[1], x2.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], y0.val[0]);
          y0.val[1] = vPoolPreq_f32<P>(x0.val[1], y0.val[1]);
          y0.val[0] = vPoolPostq_f32<P>(y0.val[0]);
          y0.val[1] = vPoolPostq_f32<P>(y0.val[1]);
          vst1q_f32(output_ptr0, y0.val[0]);
          vst1_f32(output_ptr0 + 4, vget_low_f32(y0.val[1]));

          input_ptr0 += 12;
          input_ptr1 += 12;
          input_ptr2 += 12;
          output_ptr0 += 6;
        }
        // remain w
        if (remain >= 4) {
          x0 = vld2q_f32(input_ptr0);
          x1.val[0] = vdupq_n_f32(input_ptr0[8]);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);

          x0 = vld2q_f32(input_ptr1);
          x1.val[0] = vdupq_n_f32(input_ptr1[8]);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], y0.val[0]);

          x0 = vld2q_f32(input_ptr2);
          x1.val[0] = vdupq_n_f32(input_ptr2[8]);
          x2.val[0] = vextq_f32(x0.val[0], x1.val[0], 1);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
          x0.val[0] = vPoolPreq_f32<P>(x0.val[0], x2.val[0]);
          y0.val[0] = vPoolPreq_f32<P>(x0.val[0], y0.val[0]);
          y0.val[0] = vPoolPostq_f32<P>(y0.val[0]);
          vst1q_f32(output_ptr0, y0.val[0]);

          input_ptr0 += 8;
          input_ptr1 += 8;
          input_ptr2 += 8;
          output_ptr0 += 4;
          remain -= 4;
        }
#endif  // __ARM_NEON__
        for (int r = 0; r < remain; ++r) {
          float m0 = PoolPre<P>(input_ptr0[2 * r], input_ptr0[2 * r + 1]);
          m0 = PoolPre<P>(m0, input_ptr0[2 * r + 2]);
          float m1 = PoolPre<P>(input_ptr1[2 * r], input_ptr1[2 * r + 1]);
          m1 = PoolPre<P>(m1, input_ptr1[2 * r + 2]);
          float m2 = PoolPre<P>(input_ptr2[2 * r], input_ptr2[2 * r + 1]);
          m2 = PoolPre<P>(m2, input_ptr2[2 * r + 2]);

          m0 = PoolPre<P>(PoolPre<P>(m0, m1), m2);
          output_ptr0[r] = PoolPost<P>(m0);
        }
      }
    }
  }
};

template struct Pooling3x3<Max, 1>;
template struct Pooling3x3<Avg, 1>;
template struct Pooling3x3<Max, 2>;
template struct Pooling3x3<Avg, 2>;

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif  // POOL_OP
