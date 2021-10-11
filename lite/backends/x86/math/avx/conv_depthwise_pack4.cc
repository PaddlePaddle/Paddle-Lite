/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/x86/math/avx/conv_depthwise_pack4.h"
#include <vector>
#include "lite/backends/x86/math/avx/conv_utils.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

void conv_depthwise_m128(lite::Tensor* input,
                         lite::Tensor* output,
                         lite::Tensor* filter,
                         lite::Tensor* bias,
                         const int stride_h,
                         const int stride_w,
                         const int dilation_h,
                         const int dilation_w,
                         const bool has_act,
                         const lite_api::ActivationType act_type,
                         const operators::ActivationParam act_param) {
  // input [bs, ic/8, ih, iw, 8]
  CHECK_EQ(input->dims().size(), 5UL);
  const int batch_size = input->dims()[0];
  const int channel_num = input->dims()[1];
  const int input_height = input->dims()[2];
  const int input_width = input->dims()[3];
  const float* input_data = input->data<float>();

  // filter [1, oc/8, kh, kw, 8]
  CHECK_EQ(filter->dims().size(), 5UL);
  const int kernel_h = filter->dims()[2];
  const int kernel_w = filter->dims()[3];
  const float* filter_data = filter->data<float>();

  // output [bs, oc/8, oh, ow, 8]
  CHECK_EQ(output->dims().size(), 5UL);
  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  float* output_data = output->mutable_data<float>();

  const int input_group_step = input_width * 4;
  const int input_channel_step = input_height * input_width * 4;
  const int input_batch_step = channel_num * input_height * input_width * 4;

  const int filter_kernel_size = kernel_h * kernel_w;
  const int filter_channel_step = kernel_h * kernel_w * 4;

  // kernel offsets
  std::vector<int> _space_ofs(filter_kernel_size);
  int* space_ofs = &_space_ofs[0];
  {
    int p1 = 0;
    int p2 = 0;
    int gap = input_width * dilation_h - kernel_w * dilation_w;
    for (int i = 0; i < kernel_h; i++) {
      for (int j = 0; j < kernel_w; j++) {
        space_ofs[p1++] = p2;
        p2 += dilation_w;
      }
      p2 += gap;
    }
  }

  for (int bs = 0; bs < batch_size; ++bs) {
    for (int ic = 0; ic < channel_num; ++ic) {
      const float* input_ptr =
          input_data + bs * input_batch_step + ic * input_channel_step;
      const float* filter_ptr = filter_data + ic * filter_channel_step;
      for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
          __m128 _sum = _mm_set1_ps(0.f);

          if (bias) {
            _sum = _mm_loadu_ps((bias->data<float>()) + ic * 4);
          }

          const float* start_ptr =
              input_ptr + i * stride_h * input_group_step + j * stride_w * 4;

          for (int k = 0; k < filter_kernel_size; k++) {
            __m128 _input = _mm_loadu_ps(start_ptr + space_ofs[k] * 4);
            __m128 _filter = _mm_loadu_ps(filter_ptr + k * 4);
            __m128 _mul = _mm_mul_ps(_input, _filter);
            _sum = _mm_add_ps(_mul, _sum);
          }

          if (has_act) {
            _sum = activation4_m128(_sum, act_type, act_param);
          }

          _mm_storeu_ps(output_data, _sum);
          output_data += 4;
        }
      }
    }
  }
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
