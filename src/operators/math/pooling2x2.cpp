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

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

#include <arm_neon.h>
#include "operators/math/pooling.h"

// TODO(hjchen2): Optimize Pooling2x2NormalRow and use inline assembly

namespace paddle_mobile {
namespace operators {
namespace math {

#define POOLING2X2_NORMAL_BORDER(start, end)                   \
  for (int w = start; w < end; ++w) {                          \
    const int w_in_start = -padding_w + w * Stride;            \
    const int w_in_end = w_in_start + 2;                       \
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

template <PoolingType P, int Stride = 1>
struct Pooling2x2NormalRowLoadInput {
  void operator()(const float *input, float32x4_t *x0, float32x4_t *x1) {
    x0[0] = vld1q_f32(input);
    x0[1] = vld1q_f32(input + 4);
    x1[0] = vextq_f32(x0[0], x0[1], 1);
    x1[1] = vextq_f32(x0[1], x0[1], 1);
  }
};

template <PoolingType P>
struct Pooling2x2NormalRowLoadInput<P, 2> {
  void operator()(const float *input, float32x4_t *x0, float32x4_t *x1) {
    float32x4x2_t t0 = vld2q_f32(input);
    float32x4x2_t t1 = vld2q_f32(input + 8);
    x0[0] = t0.val[0];
    x0[1] = t1.val[0];
    x1[0] = t0.val[1];
    x1[1] = t1.val[1];
  }
};

template <PoolingType P, int Stride>
inline void Pooling2x2NormalRow(const float *input, const int h_output,
                                const int input_h, const int input_w,
                                const int padding_h, const int padding_w,
                                const int output_w, float *output) {
  const int h_in_start = -padding_h + h_output * Stride;
  const int h_in_end = h_in_start + 2;
  const int h_start = h_in_start > 0 ? h_in_start : 0;
  const int h_end = h_in_end < input_h ? h_in_end : input_h;

  float *output_ptr = output + h_output * output_w;
  if (h_end - h_start <= 0) {
    memset(output_ptr, 0, output_w * sizeof(float));
    return;
  }

  const int valid_w_start = (padding_w + Stride - 1) / Stride;
  const int valid_w_end = (input_w + padding_w - 2) / Stride + 1;
  const int valid_w = valid_w_end - valid_w_start;

  // border left
  POOLING2X2_NORMAL_BORDER(0, valid_w_start)
  // valid w
  Pooling2x2NormalRowLoadInput<P, Stride> load_input;
  int output_tiles = valid_w / 6;
  int output_tiles_w = output_tiles * 6;
  float32x4_t x0[2], x1[2], y0[2];
  float32x4_t post = vdupq_n_f32(1.f / (2 * (h_end - h_start)));
  for (int w = 0; w < output_tiles_w; w += 6) {
    int output_offset = valid_w_start + w;
    int input_w_offset = output_offset * Stride - padding_w;
    y0[0] = vPoolInitq_f32<P>();
    y0[1] = vPoolInitq_f32<P>();
    for (int h_in = h_start; h_in < h_end; ++h_in) {
      load_input(input + h_in * input_w + input_w_offset, x0, x1);
      y0[0] = vPoolPreq_f32<P>(y0[0], x0[0]);
      y0[0] = vPoolPreq_f32<P>(y0[0], x1[0]);
      y0[1] = vPoolPreq_f32<P>(y0[1], x0[1]);
      y0[1] = vPoolPreq_f32<P>(y0[1], x1[1]);
    }
    y0[0] = vPoolPostq_f32<P>(y0[0], post);
    y0[1] = vPoolPostq_f32<P>(y0[1], post);
    vst1q_f32(output_ptr + output_offset, y0[0]);
    vst1_f32(output_ptr + output_offset + 4, vget_low_f32(y0[1]));
  }
  // remain valid w
  int remain = valid_w - output_tiles_w;
  if (remain > 0) {
    int remain_start = valid_w_start + output_tiles_w;
    int input_w_offset = remain_start * Stride - padding_w;
    float *output_ptr0 = output_ptr + remain_start;
    y0[0] = vPoolInitq_f32<P>();
    y0[1] = vPoolInitq_f32<P>();
    for (int h_in = h_start; h_in < h_end; ++h_in) {
      load_input(input + h_in * input_w + input_w_offset, x0, x1);
      y0[0] = vPoolPreq_f32<P>(y0[0], x0[0]);
      y0[0] = vPoolPreq_f32<P>(y0[0], x1[0]);
      y0[1] = vPoolPreq_f32<P>(y0[1], x0[1]);
      y0[1] = vPoolPreq_f32<P>(y0[1], x1[1]);
    }
    y0[0] = vPoolPostq_f32<P>(y0[0], post);
    y0[1] = vPoolPostq_f32<P>(y0[1], post);
    switch (remain) {
      case 1:
        vst1q_lane_f32(output_ptr0, y0[0], 0);
        break;
      case 2:
        vst1_f32(output_ptr0, vget_low_f32(y0[0]));
        break;
      case 3:
        vst1_f32(output_ptr0, vget_low_f32(y0[0]));
        vst1q_lane_f32(output_ptr0 + 2, y0[0], 2);
        break;
      case 4:
        vst1q_f32(output_ptr0, y0[0]);
        break;
      case 5:
        vst1q_f32(output_ptr0, y0[0]);
        vst1q_lane_f32(output_ptr0 + 4, y0[1], 0);
        break;
    }
  }
  // border right
  POOLING2X2_NORMAL_BORDER(valid_w_end, output_w)
}

template <PoolingType P>
struct Pooling2x2<P, 1> {
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

    #pragma omp parallel for collapse(2)
    for (int batch = 0; batch < output->dims()[0]; ++batch) {
      for (int c = 0; c < output->dims()[1]; ++c) {
        int channel = batch * output->dims()[1] + c;
        const float *input_ptr = input_data + channel * image_size;
        float *output_ptr = output_data + channel * out_image_size;
        // top
        for (int h = 0; h < valid_h_start; ++h) {
          Pooling2x2NormalRow<P, 1>(input_ptr, h, input_h, input_w, padding_h,
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
          float *output_ptr0 = output_ptr + h * output_w;
          float *output_ptr1 = output_ptr0 + output_w;
          float *output_ptr2 = output_ptr1 + output_w;
          float *output_ptr3 = output_ptr2 + output_w;
          // pad left
          if (padding_w) {
            for (int w = valid_w_start - 1; w >= 0; --w) {
              int padding = padding_w - w;
              if (padding >= 2) {
                output_ptr0[w] = 0.f;
                output_ptr1[w] = 0.f;
                output_ptr2[w] = 0.f;
                output_ptr3[w] = 0.f;
              } else {
                float acc0 = PoolPre<P>(*input_ptr0, *input_ptr1);
                float acc1 = PoolPre<P>(*input_ptr1, *input_ptr2);
                float acc2 = PoolPre<P>(*input_ptr2, *input_ptr3);
                float acc3 = PoolPre<P>(*input_ptr3, *input_ptr4);
                output_ptr0[w] = PoolPost<P>(acc0, 0.5f);
                output_ptr1[w] = PoolPost<P>(acc1, 0.5f);
                output_ptr2[w] = PoolPost<P>(acc2, 0.5f);
                output_ptr3[w] = PoolPost<P>(acc3, 0.5f);
              }
            }
            output_ptr0 += valid_w_start;
            output_ptr1 += valid_w_start;
            output_ptr2 += valid_w_start;
            output_ptr3 += valid_w_start;
          }
          // valid
          float32x4x2_t x0, x1, q0;
          float32x4x2_t y0, y1;
          float32x4_t post = vdupq_n_f32(0.25f);
          for (int loop = 0; loop < output_w_tiles; ++loop) {
            x0.val[0] = vld1q_f32(input_ptr0);
            x0.val[1] = vld1q_f32(input_ptr0 + 4);
            x1.val[0] = vld1q_f32(input_ptr1);
            x1.val[1] = vld1q_f32(input_ptr1 + 4);
            q0.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
            q0.val[1] = vextq_f32(x0.val[1], x0.val[1], 1);
            y0.val[0] = vPoolPreq_f32<P>(x0.val[0], q0.val[0]);
            y0.val[1] = vPoolPreq_f32<P>(x0.val[1], q0.val[1]);

            q0.val[0] = vextq_f32(x1.val[0], x1.val[1], 1);
            q0.val[1] = vextq_f32(x1.val[1], x1.val[1], 1);
            y1.val[0] = vPoolPreq_f32<P>(x1.val[0], q0.val[0]);
            y1.val[1] = vPoolPreq_f32<P>(x1.val[1], q0.val[1]);
            y0.val[0] = vPoolPreq_f32<P>(y0.val[0], y1.val[0]);
            y0.val[1] = vPoolPreq_f32<P>(y0.val[1], y1.val[1]);
            y0.val[0] = vPoolPostq_f32<P>(y0.val[0], post);
            y0.val[1] = vPoolPostq_f32<P>(y0.val[1], post);
            vst1q_f32(output_ptr0, y0.val[0]);
            vst1_f32(output_ptr0 + 4, vget_low_f32(y0.val[1]));

            x0.val[0] = vld1q_f32(input_ptr2);
            x0.val[1] = vld1q_f32(input_ptr2 + 4);
            x1.val[0] = vld1q_f32(input_ptr3);
            x1.val[1] = vld1q_f32(input_ptr3 + 4);
            q0.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
            q0.val[1] = vextq_f32(x0.val[1], x0.val[1], 1);
            y0.val[0] = vPoolPreq_f32<P>(x0.val[0], q0.val[0]);
            y0.val[1] = vPoolPreq_f32<P>(x0.val[1], q0.val[1]);
            y1.val[0] = vPoolPreq_f32<P>(y1.val[0], y0.val[0]);
            y1.val[1] = vPoolPreq_f32<P>(y1.val[1], y0.val[1]);
            y1.val[0] = vPoolPostq_f32<P>(y1.val[0], post);
            y1.val[1] = vPoolPostq_f32<P>(y1.val[1], post);
            vst1q_f32(output_ptr1, y1.val[0]);
            vst1_f32(output_ptr1 + 4, vget_low_f32(y1.val[1]));

            q0.val[0] = vextq_f32(x1.val[0], x1.val[1], 1);
            q0.val[1] = vextq_f32(x1.val[1], x1.val[1], 1);
            y1.val[0] = vPoolPreq_f32<P>(x1.val[0], q0.val[0]);
            y1.val[1] = vPoolPreq_f32<P>(x1.val[1], q0.val[1]);
            y0.val[0] = vPoolPreq_f32<P>(y0.val[0], y1.val[0]);
            y0.val[1] = vPoolPreq_f32<P>(y0.val[1], y1.val[1]);
            y0.val[0] = vPoolPostq_f32<P>(y0.val[0], post);
            y0.val[1] = vPoolPostq_f32<P>(y0.val[1], post);
            vst1q_f32(output_ptr2, y0.val[0]);
            vst1_f32(output_ptr2 + 4, vget_low_f32(y0.val[1]));

            x0.val[0] = vld1q_f32(input_ptr4);
            x0.val[1] = vld1q_f32(input_ptr4 + 4);
            q0.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
            q0.val[1] = vextq_f32(x0.val[1], x0.val[1], 1);
            y1.val[0] = vPoolPreq_f32<P>(y1.val[0], x0.val[0]);
            y1.val[0] = vPoolPreq_f32<P>(y1.val[0], q0.val[0]);
            y1.val[1] = vPoolPreq_f32<P>(y1.val[1], x0.val[1]);
            y1.val[1] = vPoolPreq_f32<P>(y1.val[1], q0.val[1]);
            y1.val[0] = vPoolPostq_f32<P>(y1.val[0], post);
            y1.val[1] = vPoolPostq_f32<P>(y1.val[1], post);
            vst1q_f32(output_ptr3, y1.val[0]);
            vst1_f32(output_ptr3 + 4, vget_low_f32(y1.val[1]));

            input_ptr0 += 6;
            input_ptr1 += 6;
            input_ptr2 += 6;
            input_ptr3 += 6;
            input_ptr4 += 6;
            output_ptr0 += 6;
            output_ptr1 += 6;
            output_ptr2 += 6;
            output_ptr3 += 6;
          }
          // remain width
          if (output_w_remain > 0) {
            float32x4x2_t y2, y3;
            x0.val[0] = vld1q_f32(input_ptr0);
            x0.val[1] = vld1q_f32(input_ptr0 + 4);
            x1.val[0] = vld1q_f32(input_ptr1);
            x1.val[1] = vld1q_f32(input_ptr1 + 4);
            q0.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
            q0.val[1] = vextq_f32(x0.val[1], x0.val[1], 1);
            y0.val[0] = vPoolPreq_f32<P>(x0.val[0], q0.val[0]);
            y0.val[1] = vPoolPreq_f32<P>(x0.val[1], q0.val[1]);

            q0.val[0] = vextq_f32(x1.val[0], x1.val[1], 1);
            q0.val[1] = vextq_f32(x1.val[1], x1.val[1], 1);
            y1.val[0] = vPoolPreq_f32<P>(x1.val[0], q0.val[0]);
            y1.val[1] = vPoolPreq_f32<P>(x1.val[1], q0.val[1]);
            y0.val[0] = vPoolPreq_f32<P>(y0.val[0], y1.val[0]);
            y0.val[1] = vPoolPreq_f32<P>(y0.val[1], y1.val[1]);
            y0.val[0] = vPoolPostq_f32<P>(y0.val[0], post);
            y0.val[1] = vPoolPostq_f32<P>(y0.val[1], post);

            x0.val[0] = vld1q_f32(input_ptr2);
            x0.val[1] = vld1q_f32(input_ptr2 + 4);
            x1.val[0] = vld1q_f32(input_ptr3);
            x1.val[1] = vld1q_f32(input_ptr3 + 4);
            q0.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
            q0.val[1] = vextq_f32(x0.val[1], x0.val[1], 1);
            y2.val[0] = vPoolPreq_f32<P>(x0.val[0], q0.val[0]);
            y2.val[1] = vPoolPreq_f32<P>(x0.val[1], q0.val[1]);
            y1.val[0] = vPoolPreq_f32<P>(y1.val[0], y2.val[0]);
            y1.val[1] = vPoolPreq_f32<P>(y1.val[1], y2.val[1]);
            y1.val[0] = vPoolPostq_f32<P>(y1.val[0], post);
            y1.val[1] = vPoolPostq_f32<P>(y1.val[1], post);

            q0.val[0] = vextq_f32(x1.val[0], x1.val[1], 1);
            q0.val[1] = vextq_f32(x1.val[1], x1.val[1], 1);
            y3.val[0] = vPoolPreq_f32<P>(x1.val[0], q0.val[0]);
            y3.val[1] = vPoolPreq_f32<P>(x1.val[1], q0.val[1]);
            y2.val[0] = vPoolPreq_f32<P>(y2.val[0], y3.val[0]);
            y2.val[1] = vPoolPreq_f32<P>(y2.val[1], y3.val[1]);
            y2.val[0] = vPoolPostq_f32<P>(y2.val[0], post);
            y2.val[1] = vPoolPostq_f32<P>(y2.val[1], post);

            x0.val[0] = vld1q_f32(input_ptr4);
            x0.val[1] = vld1q_f32(input_ptr4 + 4);
            q0.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
            q0.val[1] = vextq_f32(x0.val[1], x0.val[1], 1);
            y3.val[0] = vPoolPreq_f32<P>(y3.val[0], x0.val[0]);
            y3.val[0] = vPoolPreq_f32<P>(y3.val[0], q0.val[0]);
            y3.val[1] = vPoolPreq_f32<P>(y3.val[1], x0.val[1]);
            y3.val[1] = vPoolPreq_f32<P>(y3.val[1], q0.val[1]);
            y3.val[0] = vPoolPostq_f32<P>(y3.val[0], post);
            y3.val[1] = vPoolPostq_f32<P>(y3.val[1], post);

            switch (output_w_remain) {
              case 1:
                vst1q_lane_f32(output_ptr0, y0.val[0], 0);
                vst1q_lane_f32(output_ptr1, y1.val[0], 0);
                vst1q_lane_f32(output_ptr2, y2.val[0], 0);
                vst1q_lane_f32(output_ptr3, y3.val[0], 0);
                break;
              case 2:
                vst1_f32(output_ptr0, vget_low_f32(y0.val[0]));
                vst1_f32(output_ptr1, vget_low_f32(y1.val[0]));
                vst1_f32(output_ptr2, vget_low_f32(y2.val[0]));
                vst1_f32(output_ptr3, vget_low_f32(y3.val[0]));
                break;
              case 3:
                vst1_f32(output_ptr0, vget_low_f32(y0.val[0]));
                vst1_f32(output_ptr1, vget_low_f32(y1.val[0]));
                vst1_f32(output_ptr2, vget_low_f32(y2.val[0]));
                vst1_f32(output_ptr3, vget_low_f32(y3.val[0]));
                vst1q_lane_f32(output_ptr0 + 2, y0.val[0], 2);
                vst1q_lane_f32(output_ptr1 + 2, y1.val[0], 2);
                vst1q_lane_f32(output_ptr2 + 2, y2.val[0], 2);
                vst1q_lane_f32(output_ptr3 + 2, y3.val[0], 2);
                break;
              case 4:
                vst1q_f32(output_ptr0, y0.val[0]);
                vst1q_f32(output_ptr1, y1.val[0]);
                vst1q_f32(output_ptr2, y2.val[0]);
                vst1q_f32(output_ptr3, y3.val[0]);
                break;
              case 5:
                vst1q_f32(output_ptr0, y0.val[0]);
                vst1q_f32(output_ptr1, y1.val[0]);
                vst1q_f32(output_ptr2, y2.val[0]);
                vst1q_f32(output_ptr3, y3.val[0]);
                vst1q_lane_f32(output_ptr0 + 4, y0.val[1], 0);
                vst1q_lane_f32(output_ptr1 + 4, y1.val[1], 0);
                vst1q_lane_f32(output_ptr2 + 4, y2.val[1], 0);
                vst1q_lane_f32(output_ptr3 + 4, y3.val[1], 0);
                break;
            }
            input_ptr0 += output_w_remain;
            input_ptr1 += output_w_remain;
            input_ptr2 += output_w_remain;
            input_ptr3 += output_w_remain;
            input_ptr4 += output_w_remain;
            output_ptr0 += output_w_remain;
            output_ptr1 += output_w_remain;
            output_ptr2 += output_w_remain;
            output_ptr3 += output_w_remain;
          }
          // pad right
          if (padding_w) {
            for (int w = valid_w_end; w < output_w; ++w) {
              int padding = w + 2 - (padding_w + input_w);
              if (padding >= 2) {
                *output_ptr0 = 0.f;
                *output_ptr1 = 0.f;
                *output_ptr2 = 0.f;
                *output_ptr3 = 0.f;
              } else {
                float acc0 = PoolPre<P>(*input_ptr0, *input_ptr1);
                float acc1 = PoolPre<P>(*input_ptr1, *input_ptr2);
                float acc2 = PoolPre<P>(*input_ptr2, *input_ptr3);
                float acc3 = PoolPre<P>(*input_ptr3, *input_ptr4);
                *output_ptr0 = PoolPost<P>(acc0, 0.5f);
                *output_ptr1 = PoolPost<P>(acc1, 0.5f);
                *output_ptr2 = PoolPost<P>(acc2, 0.5f);
                *output_ptr3 = PoolPost<P>(acc3, 0.5f);
              }
              output_ptr0++;
              output_ptr1++;
              output_ptr2++;
              output_ptr3++;
            }
          }
        }
        // remain height
        int start_h = valid_h_start + (valid_h & 0xFFFFFFFC);
        for (int h = start_h; h < valid_h_end; ++h) {
          const float *input_ptr0 = input_ptr + (h - padding_h) * input_w;
          const float *input_ptr1 = input_ptr0 + input_w;
          float *output_ptr0 = output_ptr + h * output_w;
          // pad left
          if (padding_w) {
            for (int w = valid_w_start - 1; w >= 0; --w) {
              int padding = padding_w - w;
              if (padding >= 2) {
                output_ptr0[w] = 0.f;
              } else {
                float acc0 = PoolPre<P>(*input_ptr0, *input_ptr1);
                output_ptr0[w] = PoolPost<P>(acc0, 0.5f);
              }
            }
            output_ptr0 += valid_w_start;
          }
          // valid
          float32x4x2_t x0, x1, q0, y0;
          float32x4_t post = vdupq_n_f32(0.25f);
          for (int loop = 0; loop < output_w_tiles; ++loop) {
            x0.val[0] = vld1q_f32(input_ptr0);
            x0.val[1] = vld1q_f32(input_ptr0 + 4);
            x1.val[0] = vld1q_f32(input_ptr1);
            x1.val[1] = vld1q_f32(input_ptr1 + 4);
            q0.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
            q0.val[1] = vextq_f32(x0.val[1], x0.val[1], 1);
            y0.val[0] = vPoolPreq_f32<P>(x0.val[0], q0.val[0]);
            y0.val[1] = vPoolPreq_f32<P>(x0.val[1], q0.val[1]);

            q0.val[0] = vextq_f32(x1.val[0], x1.val[1], 1);
            q0.val[1] = vextq_f32(x1.val[1], x1.val[1], 1);
            y0.val[0] = vPoolPreq_f32<P>(y0.val[0], x1.val[0]);
            y0.val[1] = vPoolPreq_f32<P>(y0.val[1], x1.val[1]);
            y0.val[0] = vPoolPreq_f32<P>(y0.val[0], q0.val[0]);
            y0.val[1] = vPoolPreq_f32<P>(y0.val[1], q0.val[1]);
            y0.val[0] = vPoolPostq_f32<P>(y0.val[0], post);
            y0.val[1] = vPoolPostq_f32<P>(y0.val[1], post);
            vst1q_f32(output_ptr0, y0.val[0]);
            vst1_f32(output_ptr0 + 4, vget_low_f32(y0.val[1]));

            input_ptr0 += 6;
            input_ptr1 += 6;
            output_ptr0 += 6;
          }
          // remain width
          if (output_w_remain > 0) {
            x0.val[0] = vld1q_f32(input_ptr0);
            x0.val[1] = vld1q_f32(input_ptr0 + 4);
            x1.val[0] = vld1q_f32(input_ptr1);
            x1.val[1] = vld1q_f32(input_ptr1 + 4);
            q0.val[0] = vextq_f32(x0.val[0], x0.val[1], 1);
            q0.val[1] = vextq_f32(x0.val[1], x0.val[1], 1);
            y0.val[0] = vPoolPreq_f32<P>(x0.val[0], q0.val[0]);
            y0.val[1] = vPoolPreq_f32<P>(x0.val[1], q0.val[1]);

            q0.val[0] = vextq_f32(x1.val[0], x1.val[1], 1);
            q0.val[1] = vextq_f32(x1.val[1], x1.val[1], 1);
            y0.val[0] = vPoolPreq_f32<P>(y0.val[0], x1.val[0]);
            y0.val[1] = vPoolPreq_f32<P>(y0.val[1], x1.val[1]);
            y0.val[0] = vPoolPreq_f32<P>(y0.val[0], q0.val[0]);
            y0.val[1] = vPoolPreq_f32<P>(y0.val[1], q0.val[1]);
            y0.val[0] = vPoolPostq_f32<P>(y0.val[0], post);
            y0.val[1] = vPoolPostq_f32<P>(y0.val[1], post);

            switch (output_w_remain) {
              case 1:
                vst1q_lane_f32(output_ptr0, y0.val[0], 0);
                break;
              case 2:
                vst1_f32(output_ptr0, vget_low_f32(y0.val[0]));
                break;
              case 3:
                vst1_f32(output_ptr0, vget_low_f32(y0.val[0]));
                vst1q_lane_f32(output_ptr0 + 2, y0.val[0], 2);
                break;
              case 4:
                vst1q_f32(output_ptr0, y0.val[0]);
                break;
              case 5:
                vst1q_f32(output_ptr0, y0.val[0]);
                vst1q_lane_f32(output_ptr0 + 4, y0.val[1], 0);
                break;
            }
            input_ptr0 += output_w_remain;
            input_ptr1 += output_w_remain;
            output_ptr0 += output_w_remain;
          }
          // pad right
          if (padding_w) {
            for (int w = valid_w_end; w < output_w; ++w) {
              int padding = w + 2 - (padding_w + input_w);
              if (padding >= 2) {
                *output_ptr0 = 0.f;
              } else {
                float acc0 = PoolPre<P>(*input_ptr0, *input_ptr1);
                *output_ptr0 = PoolPost<P>(acc0, 0.5f);
              }
              output_ptr0++;
            }
          }
        }
        // bottom
        for (int h = valid_h_end; h < output_h; ++h) {
          Pooling2x2NormalRow<P, 1>(input_ptr, h, input_h, input_w, padding_h,
                                    padding_w, output_w, output_ptr);
        }
      }
    }
  }
};

template <PoolingType P>
struct Pooling2x2<P, 2> {
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
    int valid_h_end = (input_h + padding_h) / 2;
    int valid_h = valid_h_end - valid_h_start;
    int valid_w_start = (padding_w + 1) / 2;
    int valid_w_end = (input_w + padding_w) / 2;
    int valid_w = valid_w_end - valid_w_start;

    bool ceil_mode = (((input_h + 2 * padding_h) / 2) < output_h) ||
                     (((input_w + 2 * padding_w) / 2) < output_w);
    int padding_b =
        padding_h + (ceil_mode ? 2 * output_h - (input_h + 2 * padding_h) : 0);
    int padding_r =
        padding_w + (ceil_mode ? 2 * output_w - (input_w + 2 * padding_w) : 0);

    #pragma omp parallel for collapse(2)
    for (int batch = 0; batch < output->dims()[0]; ++batch) {
      for (int c = 0; c < output->dims()[1]; ++c) {
        int channel = batch * output->dims()[1] + c;
        const float *input_ptr = input_data + channel * image_size;
        float *output_ptr = output_data + channel * out_image_size;
        // top
        for (int h = 0; h < valid_h_start; ++h) {
          Pooling2x2NormalRow<P, 2>(input_ptr, h, input_h, input_w, padding_h,
                                    padding_w, output_w, output_ptr);
        }
        // valid
        int output_w_tiles = valid_w / 4;
        int output_w_remain = valid_w - output_w_tiles * 4;
        for (int h = valid_h_start; h < valid_h_end - 1; h += 2) {
          const float *input_ptr0 = input_ptr + (2 * h - padding_h) * input_w;
          const float *input_ptr1 = input_ptr0 + input_w;
          const float *input_ptr2 = input_ptr1 + input_w;
          const float *input_ptr3 = input_ptr2 + input_w;
          float *output_ptr0 = output_ptr + h * output_w;
          float *output_ptr1 = output_ptr0 + output_w;
          // pad left
          if (padding_w) {
            for (int w = valid_w_start - 1; w >= 0; --w) {
              int padding = padding_w - w * 2;
              if (padding >= 2) {
                output_ptr0[w] = 0.f;
                output_ptr1[w] = 0.f;
              } else {
                float acc0 = PoolPre<P>(*input_ptr0, *input_ptr1);
                float acc1 = PoolPre<P>(*input_ptr2, *input_ptr3);
                output_ptr0[w] = PoolPost<P>(acc0, 0.5f);
                output_ptr1[w] = PoolPost<P>(acc1, 0.5f);
              }
            }
            input_ptr0 += (padding_w & 0x1);
            input_ptr1 += (padding_w & 0x1);
            input_ptr2 += (padding_w & 0x1);
            input_ptr3 += (padding_w & 0x1);
            output_ptr0 += valid_w_start;
            output_ptr1 += valid_w_start;
          }
          // valid
          float32x4x2_t x0, x1, x2, x3;
          float32x4_t y0, y1;
          float32x4_t post = vdupq_n_f32(0.25f);
          for (int loop = 0; loop < output_w_tiles; ++loop) {
            x0 = vld2q_f32(input_ptr0);
            x1 = vld2q_f32(input_ptr1);
            x2 = vld2q_f32(input_ptr2);
            x3 = vld2q_f32(input_ptr3);
            y0 = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
            y1 = vPoolPreq_f32<P>(x2.val[0], x2.val[1]);
            y0 = vPoolPreq_f32<P>(y0, x1.val[0]);
            y1 = vPoolPreq_f32<P>(y1, x3.val[0]);
            y0 = vPoolPreq_f32<P>(y0, x1.val[1]);
            y1 = vPoolPreq_f32<P>(y1, x3.val[1]);
            y0 = vPoolPostq_f32<P>(y0, post);
            y1 = vPoolPostq_f32<P>(y1, post);
            vst1q_f32(output_ptr0, y0);
            vst1q_f32(output_ptr1, y1);

            input_ptr0 += 8;
            input_ptr1 += 8;
            input_ptr2 += 8;
            input_ptr3 += 8;
            output_ptr0 += 4;
            output_ptr1 += 4;
          }
          // remain width
          if (output_w_remain > 0) {
            x0 = vld2q_f32(input_ptr0);
            x1 = vld2q_f32(input_ptr1);
            x2 = vld2q_f32(input_ptr2);
            x3 = vld2q_f32(input_ptr3);
            y0 = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
            y1 = vPoolPreq_f32<P>(x2.val[0], x2.val[1]);
            y0 = vPoolPreq_f32<P>(y0, x1.val[0]);
            y1 = vPoolPreq_f32<P>(y1, x3.val[0]);
            y0 = vPoolPreq_f32<P>(y0, x1.val[1]);
            y1 = vPoolPreq_f32<P>(y1, x3.val[1]);
            y0 = vPoolPostq_f32<P>(y0, post);
            y1 = vPoolPostq_f32<P>(y1, post);

            switch (output_w_remain) {
              case 1:
                vst1q_lane_f32(output_ptr0, y0, 0);
                vst1q_lane_f32(output_ptr1, y1, 0);
                break;
              case 2:
                vst1_f32(output_ptr0, vget_low_f32(y0));
                vst1_f32(output_ptr1, vget_low_f32(y1));
                break;
              case 3:
                vst1_f32(output_ptr0, vget_low_f32(y0));
                vst1q_lane_f32(output_ptr0 + 2, y0, 2);
                vst1_f32(output_ptr1, vget_low_f32(y1));
                vst1q_lane_f32(output_ptr1 + 2, y1, 2);
                break;
            }
            input_ptr0 += 2 * output_w_remain;
            input_ptr1 += 2 * output_w_remain;
            input_ptr2 += 2 * output_w_remain;
            input_ptr3 += 2 * output_w_remain;
            output_ptr0 += output_w_remain;
            output_ptr1 += output_w_remain;
          }
          // pad right
          if (padding_r) {
            for (int w = valid_w_end; w < output_w; ++w) {
              int padding = 2 * w + 2 - (padding_w + input_w);
              if (padding >= 2) {
                *output_ptr0 = 0.f;
                *output_ptr1 = 0.f;
              } else {
                float acc0 = PoolPre<P>(*input_ptr0, *input_ptr1);
                float acc1 = PoolPre<P>(*input_ptr2, *input_ptr3);
                *output_ptr0 = PoolPost<P>(acc0, 0.5f);
                *output_ptr1 = PoolPost<P>(acc1, 0.5f);
              }
              output_ptr0++;
              output_ptr1++;
            }
          }
        }
        // remain height
        int start_h = valid_h_start + (valid_h & 0xfffffffe);
        for (int h = start_h; h < valid_h_end; ++h) {
          const float *input_ptr0 = input_ptr + (2 * h - padding_h) * input_w;
          const float *input_ptr1 = input_ptr0 + input_w;
          float *output_ptr0 = output_ptr + h * output_w;
          // pad left
          if (padding_w) {
            for (int w = valid_w_start - 1; w >= 0; --w) {
              int padding = padding_w - 2 * w;
              if (padding >= 2) {
                output_ptr0[w] = 0.f;
              } else {
                float acc0 = PoolPre<P>(*input_ptr0, *input_ptr1);
                output_ptr0[w] = PoolPost<P>(acc0, 0.5f);
              }
            }
            input_ptr0 += (padding_w & 0x1);
            input_ptr1 += (padding_w & 0x1);
            output_ptr0 += valid_w_start;
          }
          // valid
          float32x4x2_t x0, x1;
          float32x4_t y0;
          float32x4_t post = vdupq_n_f32(0.25f);
          for (int loop = 0; loop < output_w_tiles; ++loop) {
            x0 = vld2q_f32(input_ptr0);
            x1 = vld2q_f32(input_ptr1);
            y0 = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
            y0 = vPoolPreq_f32<P>(y0, x1.val[0]);
            y0 = vPoolPreq_f32<P>(y0, x1.val[1]);
            y0 = vPoolPostq_f32<P>(y0, post);
            vst1q_f32(output_ptr0, y0);

            input_ptr0 += 8;
            input_ptr1 += 8;
            output_ptr0 += 4;
          }
          // remain width
          if (output_w_remain > 0) {
            x0 = vld2q_f32(input_ptr0);
            x1 = vld2q_f32(input_ptr1);
            y0 = vPoolPreq_f32<P>(x0.val[0], x0.val[1]);
            y0 = vPoolPreq_f32<P>(y0, x1.val[0]);
            y0 = vPoolPreq_f32<P>(y0, x1.val[1]);
            y0 = vPoolPostq_f32<P>(y0, post);

            switch (output_w_remain) {
              case 1:
                vst1q_lane_f32(output_ptr0, y0, 0);
                break;
              case 2:
                vst1_f32(output_ptr0, vget_low_f32(y0));
                break;
              case 3:
                vst1_f32(output_ptr0, vget_low_f32(y0));
                vst1q_lane_f32(output_ptr0 + 2, y0, 2);
                break;
            }
            input_ptr0 += 2 * output_w_remain;
            input_ptr1 += 2 * output_w_remain;
            output_ptr0 += output_w_remain;
          }
          // pad right
          if (padding_r) {
            for (int w = valid_w_end; w < output_w; ++w) {
              int padding = 2 * w + 2 - (padding_w + input_w);
              if (padding >= 2) {
                *output_ptr0 = 0.f;
              } else {
                float acc0 = PoolPre<P>(*input_ptr0, *input_ptr1);
                *output_ptr0 = PoolPost<P>(acc0, 0.5f);
              }
              output_ptr0++;
            }
          }
        }
        // bottom
        for (int h = valid_h_end; h < output_h; ++h) {
          Pooling2x2NormalRow<P, 2>(input_ptr, h, input_h, input_w, padding_h,
                                    padding_w, output_w, output_ptr);
        }
      }
    }
  }
};

template struct Pooling2x2<MAX, 1>;
template struct Pooling2x2<AVG, 1>;
template struct Pooling2x2<MAX, 2>;
template struct Pooling2x2<AVG, 2>;

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif  // __ARM_NEON__
#endif  // POOL_OP
