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
#define __ARM_NEON true
#include "pool_3x3.h"
#include "framework/tensor.h"
#if __ARM_NEON
#include <arm_neon.h>
#endif  // __ARM_NEON

namespace paddle_mobile {
namespace operators {
namespace math {
using framework::Tensor;
using std::max;
using std::min;
using std::vector;

void Pool3x3Max(vector<int> strides, vector<int> paddings, const Tensor *input,
                Tensor *output) {
#if __ARM_NEON
  const int batch_size = input->dims()[0];

  const int input_height = input->dims()[2];

  const int input_width = input->dims()[3];

  const int output_channels = output->dims()[1];

  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  const int _kernel_size = 3;
  const int stride_height = strides[0];
  const int stride_width = strides[1];
  const int padding_height = paddings[0];
  const int padding_width = paddings[1];
  const float negative_max = -INT_MAX;
  const int input_channel_stride = input_height * input_width;
  const int output_channel_stride = output_height * output_width;

  const float *input_data = input->data<float>();
  float *output_data = output->mutable_data<float>();

  const int input_batch_stride = output_channels * input_channel_stride;
  const int output_batch_stride = output_channels * output_channel_stride;
  const float *pos1, *pos2, *pos3, *output_ptr;
  int hstart, wstart, hend, wend;
  for (int i = 0; i < batch_size; ++i) {
    for (int c = 0; c < output_channels; ++c) {
      for (int ph = 0; ph < output_height; ph++) {
        for (int pw = 0; pw < output_width; pw++) {
          hstart = ph * stride_height - padding_height;
          wstart = pw * stride_width - padding_width;
          hend = min(hstart + _kernel_size, input_height + padding_height);
          wend = min(wstart + _kernel_size, input_width + padding_width);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          hend = min(hend, input_height);
          wend = min(wend, input_width);
          pos1 = input_data + hstart * input_width + wstart;
          pos2 = input_data + (hstart + 1) * input_width + wstart;
          pos3 = input_data + (hstart + 2) * input_width + wstart;
          output_ptr = output_data + ph * output_width + pw;

          if (hend - hstart != 3 || wend - wstart != 3) {
            float max_value = -INT_MAX;
            for (int h = hstart; h < hend; h++) {
              for (int w = wstart; w < wend; w++) {
                float value = input_data[h * input_width + w];
                if (value > max_value) {
                  max_value = value;
                }
              }
            }
            output_data[ph * output_width + pw] = max_value;
          } else {
#if defined(ARMV7)
            asm volatile(
                "vld1.32  {q1}, [%[pos1]]        \n\t"
                "vld1.32  {q2}, [%[pos2]]        \n\t"
                "vld1.32  {q3}, [%[pos3]]        \n\t"
                "vmax.f32 q1, q1, q2            \n\t"
                "vmax.f32 q2, q1, q3            \n\t"
                "vmov.f32 d5[1],  %[negative_max]         \n\t"
                "vpmax.f32  d6, d4, d5            \n\t"
                "vpmax.f32  d7, d6, d6             \n\t"
                "vst1.32 {d7[0]},[%[output_ptr]]    \n\t"
                :
                : [input_data] "r"(input_data), [pos1] "r"(pos1),
                  [pos2] "r"(pos2), [pos3] "r"(pos3),
                  [output_ptr] "r"(output_ptr), [negative_max] "r"(negative_max)
                : "memory", "q1", "q2", "q3", "q4");
#else
            const float32x4_t data1 = vld1q_f32(pos1);
            const float32x4_t data2 = vld1q_f32(pos2);
            const float32x4_t data3 = vld1q_f32(pos3);
            const float32x4_t max_data =
                vmaxq_f32(vmaxq_f32(data1, data3), data2);
            float32x2_t res =
                vpmax_f32(vget_high_f32(vsetq_lane_f32(-INT_MAX, max_data, 3)),
                          vget_low_f32(max_data));
            res = vpmax_f32(res, res);
            output_data[ph * output_width + pw] = vget_lane_f32(res, 0);
#endif
          }
        }
      }
      input_data += input_channel_stride;
      output_data += output_channel_stride;
    }
    input_data += input_batch_stride;
    output_data += output_batch_stride;
  }
#endif
}

void Pool3x3Avg(vector<int> strides, vector<int> paddings, const Tensor *input,
                Tensor *output) {
#if __ARM_NEON
  const int batch_size = input->dims()[0];

  const int input_height = input->dims()[2];

  const int input_width = input->dims()[3];

  const int output_channels = output->dims()[1];

  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  const int _kernel_size = 3;
  const int stride_height = strides[0];
  const int stride_width = strides[1];
  const int padding_height = paddings[0];
  const int padding_width = paddings[1];

  const int input_channel_stride = input_height * input_width;
  const int output_channel_stride = output_height * output_width;

  const float *input_data = input->data<float>();
  float *output_data = output->mutable_data<float>();
  const float zero = 0;
  const float nine = 1.0 / 9.0;
  const float nine_ptr[] = {nine, nine};

  const int input_batch_stride = output_channels * input_channel_stride;
  const int output_batch_stride = output_channels * output_channel_stride;
  for (int i = 0; i < batch_size; ++i) {
    for (int c = 0; c < output_channels; ++c) {
      for (int ph = 0; ph < output_height; ph++) {
        for (int pw = 0; pw < output_width; pw++) {
          int hstart = ph * stride_height - padding_height;
          int wstart = pw * stride_width - padding_width;
          int hend = min(hstart + _kernel_size, input_height + padding_height);
          int wend = min(wstart + _kernel_size, input_width + padding_width);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          hend = min(hend, input_height);
          wend = min(wend, input_width);
          const float *pos1 = input_data + hstart * input_width + wstart;
          const float *pos2 = input_data + (hstart + 1) * input_width + wstart;
          const float *pos3 = input_data + (hstart + 2) * input_width + wstart;
          const float *output_ptr = output_data + ph * output_width + pw;

          if (hend - hstart != 3 || wend - wstart != 3) {
            float sum = 0;
            for (int h = hstart; h < hend; h++) {
              for (int w = wstart; w < wend; w++) {
                sum += input_data[h * input_width + w];
              }
            }
            output_data[ph * output_width + pw] = sum / 9.0;
          } else {
#if defined(ARMV7)

            asm volatile(
                "vld1.32  {q1}, [%[pos1]]        \n\t"
                "vld1.32  {q2}, [%[pos2]]        \n\t"
                "vld1.32  {q3}, [%[pos3]]        \n\t"
                "vadd.f32 q1, q1, q2            \n\t"
                "vadd.f32 q2, q1, q3            \n\t"
                "vmov.f32 d5[1],  %[zero]         \n\t"
                "vpadd.f32  d6, d4, d5            \n\t"
                "vpadd.f32  d6, d6, d6             \n\t"
                "vld1.f32 d7, [%[nine_ptr]]!        \n\t"
                "vmul.f32 d6,d7                     \n\t"
                "vst1.32 {d6[0]},[%[output_ptr]]    \n\t"
                :
                : [input_data] "r"(input_data), [pos1] "r"(pos1),
                  [pos2] "r"(pos2), [pos3] "r"(pos3),
                  [output_ptr] "r"(output_ptr), [zero] "r"(zero),
                  [nine_ptr] "r"(nine_ptr)
                : "memory", "r6", "q1", "q2", "q3", "q4");
#else
            const float32x4_t data1 = vld1q_f32(pos1);
            const float32x4_t data2 = vld1q_f32(pos2);
            const float32x4_t data3 = vld1q_f32(pos3);
            const float32x4_t sum_data =
                vaddq_f32(vaddq_f32(data1, data3), data2);
            float32x2_t res =
                vpadd_f32(vget_high_f32(vsetq_lane_f32(0, sum_data, 3)),
                          vget_low_f32(sum_data));
            res = vpadd_f32(res, res);
            output_data[ph * output_width + pw] = vget_lane_f32(res, 0) / 9.0;
#endif
          }
        }
      }
      input_data += input_channel_stride;
      output_data += output_channel_stride;
    }
    input_data += input_batch_stride;
    output_data += output_batch_stride;
  }
#endif
}
}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif
