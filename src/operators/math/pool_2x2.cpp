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
#include "operators/math/pool_2x2.h"
#include <algorithm>
#include <vector>

namespace paddle_mobile {
namespace operators {
namespace math {
#define FLT_MAX __FLT_MAX__

void Pool2x2Maxs2p0(vector<int> strides, vector<int> paddings,
                    const Tensor *input, Tensor *output) {
  const int batch_size = input->dims()[0];
  const int input_height = input->dims()[2];
  const int input_width = input->dims()[3];

  const int output_channels = output->dims()[1];
  int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  const int ksize_height = 2;
  const int ksize_width = 2;
  const int stride_height = strides[0];
  const int stride_width = strides[1];
  const int padding_height = paddings[0];
  const int padding_width = paddings[1];

  const int input_channel_stride = input_height * input_width;
  const int output_channel_stride = output_height * output_width;

  const int input_batch_stride = output_channels * input_channel_stride;
  const int output_batch_stride = output_channels * output_channel_stride;

  const float *input_data = input->data<float>();
  float *output_data = output->mutable_data<float>();

  int w1 = input_width / 16;
  int _w1 = input_width % 16;
  int w2 = _w1 / 4;
  int _w2 = _w1 % 4;

  for (int i = 0; i < batch_size; ++i) {
    for (int c = 0; c < output_channels; ++c) {
      for (int ph = 0; ph < input_height; ph += 2) {
        const float *in_ptr1 = input_data + i * input_batch_stride +
                               c * input_channel_stride + ph * input_width;
        const float *in_ptr2 = in_ptr1 + input_width;
        if (ph != input_height && ph + 1 >= input_height) {
          in_ptr2 = static_cast<float *>(
              paddle_mobile::memory::Alloc(sizeof(float) * input_width));
          memset(static_cast<void *>(const_cast<float *>(in_ptr2)), -FLT_MAX,
                 sizeof(float) * input_width);
        }
        float *out_ptr = output_data + i * output_batch_stride +
                         c * output_channel_stride + ph / 2 * output_width;
#if __ARM_NEON
#if __aarch64__
#else
        asm volatile(
            "subs       %[w1], %[w1], #1        \n\t"
            "blt        end_w1_%=               \n\t"
            "loop_w1_%=:                        \n\t"

            "pld        [%[in_ptr1], #64]       \n\t"
            "pld        [%[in_ptr2], #64]       \n\t"

            "vld1.f32   {q0, q1},   [%[in_ptr1]]!   \n\t"
            "vld1.f32   {q2, q3},   [%[in_ptr2]]!   \n\t"
            "vld1.f32   {q6, q7},   [%[in_ptr1]]!   \n\t"
            "vld1.f32   {q8, q9},   [%[in_ptr2]]!   \n\t"

            "vmax.f32   q0,     q0,   q2        \n\t"
            "vmax.f32   q1,     q1,   q3        \n\t"

            "vmax.f32   q6,     q6,   q8        \n\t"
            "vmax.f32   q7,     q7,   q9        \n\t"

            "vpmax.f32  d8,     d0,   d1        \n\t"
            "vpmax.f32  d9,     d2,   d3        \n\t"

            "vpmax.f32  d10,    d12,  d13       \n\t"
            "vpmax.f32  d11,    d14,  d15       \n\t"

            "vst1.32  {q4, q5},  [%[out_ptr]]!  \n\t"

            "subs       %[w1], %[w1], #1        \n\t"
            "bge        loop_w1_%=              \n\t"
            "end_w1_%=:                         \n\t"

            "subs       %[w2], %[w2], #1        \n\t"
            "blt        end_w2_%=               \n\t"
            "loop_w2_%=:                        \n\t"

            "vld1.f32   {q0},   [%[in_ptr1]]!   \n\t"
            "vld1.f32   {q1},   [%[in_ptr2]]!   \n\t"
            "vmax.f32   q0,     q0,   q1        \n\t"
            "vpmax.f32  d4,     d0,   d1        \n\t"
            "vst1.32    {d4},   [%[out_ptr]]!   \n\t"

            "subs       %[w2], %[w2], #1        \n\t"
            "bge        loop_w2_%=              \n\t"
            "end_w2_%=:                         \n\t"
            :
            : [w1] "r"(w1), [w2] "r"(w2), [in_ptr1] "r"(in_ptr1),
              [in_ptr2] "r"(in_ptr2), [out_ptr] "r"(out_ptr)
            : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
              "q9");
#endif
#endif

        if (_w2 != 0) {
          in_ptr1 = input_data + i * input_batch_stride +
                    c * input_channel_stride + ph * input_width + 16 * w1 +
                    4 * w2;
          in_ptr2 = in_ptr1 + input_width;
          out_ptr = output_data + i * output_batch_stride +
                    c * output_channel_stride + ph / 2 * output_width + 8 * w1 +
                    2 * w2;
          if (_w2 == 1) {
            *out_ptr = (*in_ptr1 > *in_ptr2) ? *in_ptr1 : *in_ptr2;
          } else if (_w2 == 2) {
            float temp = (*in_ptr1 > *in_ptr2) ? *in_ptr1 : *in_ptr2;
            in_ptr1++;
            in_ptr2++;
            float temp1 = (*in_ptr1 > *in_ptr2) ? *in_ptr1 : *in_ptr2;
            *out_ptr = (temp > temp1) ? temp : temp1;
          } else if (_w2 == 3) {
            float temp = (*in_ptr1 > *in_ptr2) ? *in_ptr1 : *in_ptr2;
            in_ptr1++;
            in_ptr2++;
            float temp1 = (*in_ptr1 > *in_ptr2) ? *in_ptr1 : *in_ptr2;
            in_ptr1++;
            in_ptr2++;
            *out_ptr = (temp > temp1) ? temp : temp1;
            out_ptr++;
            *out_ptr = (*in_ptr1 > *in_ptr2) ? *in_ptr1 : *in_ptr2;
          }
        }
      }
    }
  }
}

void Pool2x2Avgs2p0(vector<int> strides, vector<int> paddings,
                    const Tensor *input, Tensor *output) {
  const int batch_size = input->dims()[0];
  const int input_height = input->dims()[2];
  const int input_width = input->dims()[3];

  const int output_channels = output->dims()[1];
  int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  const int ksize_height = 2;
  const int ksize_width = 2;
  const int stride_height = strides[0];
  const int stride_width = strides[1];
  const int padding_height = paddings[0];
  const int padding_width = paddings[1];

  const int input_channel_stride = input_height * input_width;
  const int output_channel_stride = output_height * output_width;

  const int input_batch_stride = output_channels * input_channel_stride;
  const int output_batch_stride = output_channels * output_channel_stride;

  const float *input_data = input->data<float>();
  float *output_data = output->mutable_data<float>();

  int w1 = input_width / 16;
  int _w1 = input_width % 16;
  int w2 = _w1 / 4;
  int _w2 = _w1 % 4;

  float quarter = 0.25;
  for (int i = 0; i < batch_size; ++i) {
    for (int c = 0; c < output_channels; ++c) {
      for (int ph = 0; ph < input_height; ph += 2) {
        const float *in_ptr1 = input_data + i * input_batch_stride +
                               c * input_channel_stride + ph * input_width;
        const float *in_ptr2 = in_ptr1 + input_width;
        if (ph + 1 >= input_height) {
          in_ptr2 = static_cast<float *>(
              paddle_mobile::memory::Alloc(sizeof(float) * input_width));
          memset(static_cast<void *>(const_cast<float *>(in_ptr2)), 0,
                 sizeof(float) * input_width);
        }
        float *out_ptr = output_data + i * output_batch_stride +
                         c * output_channel_stride + ph / 2 * output_width;
#if __ARM_NEON
#if __aarch64__
#else
        asm volatile(
            "subs       %[w1], %[w1], #1        \n\t"
            "blt        end_w1_%=               \n\t"
            "loop_w1_%=:                        \n\t"

            "pld        [%[in_ptr1], #64]       \n\t"
            "pld        [%[in_ptr2], #64]       \n\t"

            "vmov.f32   d0[0],      %[quarter]      \n\t"
            "vld1.f32   {q1, q2},   [%[in_ptr1]]!   \n\t"
            "vld1.f32   {q3, q4},   [%[in_ptr2]]!   \n\t"
            "vld1.f32   {q7, q8},   [%[in_ptr1]]!   \n\t"
            "vld1.f32   {q9, q10},  [%[in_ptr2]]!   \n\t"

            "vadd.f32   q1,     q1,   q3        \n\t"
            "vadd.f32   q2,     q2,   q4        \n\t"

            "vadd.f32   q7,     q7,   q9        \n\t"
            "vadd.f32   q8,     q8,   q10       \n\t"

            "vpadd.f32  d10,    d2,   d3        \n\t"
            "vpadd.f32  d11,    d4,   d5        \n\t"

            "vpadd.f32  d12,    d14,  d15       \n\t"
            "vpadd.f32  d13,    d16,  d17       \n\t"

            "vmul.f32   q5,     q5,   d0[0]     \n\t"
            "vmul.f32   q6,     q6,   d0[0]     \n\t"

            "vst1.32  {q5, q6},  [%[out_ptr]]!  \n\t"

            "subs       %[w1], %[w1], #1        \n\t"
            "bge        loop_w1_%=              \n\t"
            "end_w1_%=:                         \n\t"

            "subs       %[w2], %[w2], #1        \n\t"
            "blt        end_w2_%=               \n\t"
            "loop_w2_%=:                        \n\t"

            "vld1.f32   {q1},   [%[in_ptr1]]!   \n\t"
            "vld1.f32   {q2},   [%[in_ptr2]]!   \n\t"
            "vadd.f32   q1,     q1,   q2        \n\t"
            "vpadd.f32  d4,     d2,   d3        \n\t"
            "vmul.f32   d4,     d4,   d0[0]     \n\t"
            "vst1.32    {d4},   [%[out_ptr]]!   \n\t"

            "subs       %[w2], %[w2], #1        \n\t"
            "bge        loop_w2_%=              \n\t"
            "end_w2_%=:                         \n\t"
            :
            : [w1] "r"(w1), [w2] "r"(w2), [in_ptr1] "r"(in_ptr1),
              [in_ptr2] "r"(in_ptr2), [out_ptr] "r"(out_ptr),
              [quarter] "r"(quarter)
            : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
              "q9", "q10");
#endif
#endif

        if (_w2 != 0) {
          in_ptr1 = input_data + i * input_batch_stride +
                    c * input_channel_stride + ph * input_width + 16 * w1 +
                    4 * w2;
          in_ptr2 = in_ptr1 + input_width;
          out_ptr = output_data + i * output_batch_stride +
                    c * output_channel_stride + ph / 2 * output_width + 8 * w1 +
                    2 * w2;
          if (_w2 == 1) {
            *out_ptr = 0.5 * (*in_ptr1 + *in_ptr2);
          } else if (_w2 == 2) {
            float temp = 0;
            temp += *in_ptr1;
            temp += *in_ptr2;
            in_ptr1++;
            in_ptr2++;
            temp += *in_ptr1;
            temp += *in_ptr2;
            *out_ptr = 0.25 * temp;
          } else if (_w2 == 3) {
            float temp = 0;
            temp += *in_ptr1++;
            temp += *in_ptr2++;
            temp += *in_ptr1++;
            temp += *in_ptr2++;
            *out_ptr = 0.25 * temp;
            out_ptr++;
            *out_ptr = 0.5 * (*in_ptr1 + *in_ptr2);
          }
        }
      }
    }
  }
}

//}
}  // namespace math

}  // namespace operators
}  // namespace paddle_mobile

#endif
