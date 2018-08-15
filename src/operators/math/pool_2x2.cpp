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

void Pool2x2Max(vector<int> strides, vector<int> paddings, const Tensor *input,
                Tensor *output) {
#if __ARM_NEON

#if __aarch64__
#else
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

  const float *input_data = input->data<float>();
  float *output_data = output->mutable_data<float>();

  int out_w_num = output_width >> 2;
  const int in_h_num = output_height >> 1;
  const int input_batch_stride = output_channels * input_channel_stride;
  const int output_batch_stride = output_channels * output_channel_stride;
  int remain = output_width - out_w_num << 2;
  for (int i = 0; i < batch_size; ++i) {
    for (int c = 0; c < output_channels; ++c) {
      const float *input_data_chanel_row_next = input_data + input_width;
      for (; output_height > 0; output_height--) {
        if (out_w_num > 0) {
          asm volatile(
              "max_loop:                            \n\t"
              "vld1.f32  {q0,q1},  [%[in_ptr1]]!         \n\t"
              "vld1.f32  {q2,q3},  [%[in_ptr2]]!         \n\t"
              "vmax.f32  q0,  q0,  q2                 \n\t"
              "vmax.f32  q1,  q1,  q3                 \n\t"
              "vpmax.f32  d4,  d0, d1                  \n\t"
              "vpmax.f32  d5,  d2, d3                  \n\t"
              "subs %[out_w_num],  #1                  \n\t"
              "vst1.32  {q2},  [%[out_ptr]]!                 \n\t"
              "bne  max_loop                            \n\t"
              : [in_ptr1] "+r"(input_data),
                [in_ptr2] "+r"(input_data_chanel_row_next),
                [out_ptr] "+r"(output_data), [out_w_num] "+r"(out_w_num)
              :
              : "memory", "q0", "q1", "q2", "q3");
        }

        for (; remain > 0; remain--) {
          float max_row1 = std::max(input_data[0], input_data[1]);
          float max_row2 = std::max(input_data_chanel_row_next[0],
                                    input_data_chanel_row_next[1]);
          *output_data = std::max(max_row1, max_row2);
          input_data += 2;
          input_data_chanel_row_next += 2;
          output_data++;
        }
      }
      input_data += input_channel_stride;
      output_data += output_channel_stride;
    }
    input_data += input_batch_stride;
    output_data += output_batch_stride;
  }
#endif
#else
#endif
}

void Pool2x2Avg(vector<int> strides, vector<int> paddings, const Tensor *input,
                Tensor *output) {
#if __ARM_NEON

#if __aarch64__
#else
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

  const float *input_data = input->data<float>();
  float *output_data = output->mutable_data<float>();

  int out_w_num = output_width >> 2;
  const int input_batch_stride = output_channels * input_channel_stride;
  const int output_batch_stride = output_channels * output_channel_stride;
  float vqua[] = {0.25f, 0.25f, 0.25f, 0.25f};
  int remain = output_width - out_w_num << 2;
  for (int i = 0; i < batch_size; ++i) {
    for (int c = 0; c < output_channels; ++c) {
      const float *input_data_chanel_row_next = input_data + input_width;
      for (; output_height > 0; output_height--) {
        if (out_w_num > 0) {
          asm volatile(
              "avg_loop:                            \n\t"
              "vld1.32  {q0,q1},  [%[in_ptr1]]!         \n\t"
              "vld1.32  {q2,q3},  [%[in_ptr2]]!         \n\t"
              "vadd.f32  q0,  q0,  q2                 \n\t"
              "vadd.f32  q1,  q1,  q3                 \n\t"
              "vpadd.f32  d4,  d0, d1                  \n\t"
              "vpadd.f32  d5,  d2, d3                  \n\t"
              "vld1.32  {q4}, [%[vqua]]!                  \n\t"
              "vmul.f32  q2,  q2,  q4                          \n\t"
              "subs %[out_w_num],  #1                  \n\t"
              "vst1.32  {q2},  [%[out_ptr]]!                 \n\t"
              "bne  avg_loop                            \n\t"
              : [in_ptr1] "+r"(input_data),
                [in_ptr2] "+r"(input_data_chanel_row_next),
                [out_ptr] "+r"(output_data), [out_w_num] "+r"(out_w_num)
              : [vqua] "r"(vqua)
              : "memory", "q0", "q1", "q2", "q3", "q4");
        }

        for (; remain > 0; remain--) {
          float max_row1 = std::max(input_data[0], input_data[1]);
          float max_row2 = std::max(input_data_chanel_row_next[0],
                                    input_data_chanel_row_next[1]);
          *output_data = std::max(max_row1, max_row2);
          input_data += 2;
          input_data_chanel_row_next += 2;
          output_data++;
        }
      }
      input_data += input_channel_stride;
      output_data += output_channel_stride;
    }
    input_data += input_batch_stride;
    output_data += output_batch_stride;
  }

#endif
#else
#endif
}

//}
}  // namespace math

}  // namespace operators
}  // namespace paddle_mobile

#endif
