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

#pragma once

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#include "framework/ddim.h"
#include "framework/tensor.h"

namespace paddle_mobile {
namespace operators {
namespace math {

using framework::DDim;
using framework::Tensor;

inline int ConvOutputSize(int input_size, int filter_size, int dilation,
                          int padding, int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + 2 * padding - dkernel) / stride + 1;
  return output_size;
}

inline void expand_bias(Tensor &bias, int axis, const DDim &dDim) {
  auto bias_ptr = bias.data<float>();
  const DDim bias_ddim = bias.dims();
  PADDLE_MOBILE_ENFORCE(bias.dims().size() == 1,
                        "the bias tensor's dims size != 1")
  DDim outer_ddim = paddle_mobile::framework::slice_ddim(dDim, 0, axis + 1);
  DDim inner_ddim =
      paddle_mobile::framework::slice_ddim(dDim, axis + 1, dDim.size());
  int outer_size = paddle_mobile::framework::product(outer_ddim);
  int inner_size = paddle_mobile::framework::product(inner_ddim);
  bias.Resize(dDim);
  auto new_ptr = bias.mutable_data<float>();
  int axis_size = dDim[axis];

#ifdef __ARM_NEON
  for (int i = 0; i < outer_size; ++i) {
    int inner_num = inner_size >> 4;
    int remain = inner_size - (inner_num << 4);
    float v_bias = bias_ptr[i * axis_size / outer_size];
    for (; inner_num > 0; inner_num--) {
      float32x4_t v_newptr1 = vdupq_n_f32(v_bias);
      float32x4_t v_newptr2 = vdupq_n_f32(v_bias);
      float32x4_t v_newptr3 = vdupq_n_f32(v_bias);
      float32x4_t v_newptr4 = vdupq_n_f32(v_bias);
      vst1q_f32(new_ptr, v_newptr1);
      new_ptr += 4;
      vst1q_f32(new_ptr, v_newptr2);
      new_ptr += 4;
      vst1q_f32(new_ptr, v_newptr3);
      new_ptr += 4;
      vst1q_f32(new_ptr, v_newptr4);
      new_ptr += 4;
    }
    for (; remain > 0; remain--) {
      *new_ptr = v_bias;
      new_ptr++;
    }
  }
#else
  for (int i = 0; i < outer_size; ++i) {
    float v_bias = bias_ptr[i * axis_size / outer_size];
    for (int j = 0; j < inner_size; ++j) {
      new_ptr[i * inner_size + j] = v_bias;
    }
  }
#endif
}

inline bool IsExpand(const std::vector<int64_t> &filter_dim,
                     const std::vector<int> &strides,
                     const std::vector<int> &paddings,
                     const std::vector<int> &dilations) {
  bool filter_1 = true, strides_1 = true, padding_0 = true, dilation_1 = true;
  for (size_t j = 0; j < strides.size(); ++j) {
    filter_1 = filter_1 && (static_cast<int>(filter_dim[j + 2]) == 1);
    strides_1 = strides_1 && (strides[j] == 1);
    padding_0 = padding_0 && (paddings[j] == 0);
    dilation_1 = dilation_1 && (dilations[j] == 1);
  }

  return !(filter_1 && strides_1 && padding_0 && dilation_1);
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
