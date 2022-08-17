// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <math.h>
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>
#include "operation/math/dequantize.h"
#include "operation/math/quantize.h"
#include "operation/math/utility.h"

namespace nnadapter {
namespace operation {
namespace math {

template <typename T>
static int conv2d(const T* input_data,
                  const std::vector<int32_t>& input_shape,
                  const T* filter_data,
                  const std::vector<int32_t>& filter_shape,
                  const T* bias_data,
                  int pad_height_top,
                  int pad_height_bottom,
                  int pad_width_left,
                  int pad_width_right,
                  int stride_height,
                  int stride_width,
                  int dilation_height,
                  int dilation_width,
                  int group,
                  FuseCode fuse_code,
                  T* output_data) {
  if (!input_data || !filter_data || !output_data) {
    return -1;
  }
  auto input_rank = input_shape.size();
  auto filter_rank = filter_shape.size();
  if (input_rank != 4 || filter_rank != 4) {
    return -1;
  }
  auto batch_size = input_shape[0];
  auto input_channel_size = input_shape[1];
  auto input_height = input_shape[2];
  auto input_width = input_shape[3];
  auto output_channel_size = filter_shape[0];
  auto kernel_height = filter_shape[2];
  auto kernel_width = filter_shape[3];
  auto output_height = (input_height + (pad_height_top + pad_height_bottom) -
                        dilation_height * (kernel_height - 1) - 1) /
                           stride_height +
                       1;
  auto output_width = (input_width + (pad_width_left + pad_width_right) -
                       dilation_width * (kernel_width - 1) - 1) /
                          stride_width +
                      1;
  auto output_channel_group = output_channel_size / group;
  auto input_channel_group = input_channel_size / group;
  for (int bs = 0; bs < batch_size; bs++) {
    for (int g = 0; g < group; g++) {
      for (int oc = 0; oc < output_channel_group; oc++) {
        for (int oh = 0; oh < output_height; oh++) {
          for (int ow = 0; ow < output_width; ow++) {
            int output_index =
                bs * group * output_channel_group * output_height *
                    output_width +
                g * output_channel_group * output_height * output_width +
                oc * output_height * output_width + oh * output_width + ow;
            T output_value =
                bias_data ? bias_data[g * output_channel_group + oc] : 0;
            for (int ic = 0; ic < input_channel_group; ic++) {
              for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                  int iw =
                      ow * stride_width - pad_width_left + kw * dilation_width;
                  int ih = oh * stride_height - pad_height_top +
                           kh * dilation_height;
                  if (iw < 0 || iw >= input_width) continue;
                  if (ih < 0 || ih >= input_height) continue;
                  int input_index =
                      bs * input_channel_size * input_height * input_width +
                      g * input_channel_group * input_height * input_width +
                      ic * input_height * input_width + ih * input_width + iw;
                  int filter_index =
                      g * output_channel_group * input_channel_group *
                          kernel_height * kernel_width +
                      oc * input_channel_group * kernel_height * kernel_width +
                      ic * kernel_height * kernel_width + kh * kernel_width +
                      kw;
                  output_value +=
                      input_data[input_index] * filter_data[filter_index];
                }
              }
            }
            if (fuse_code == FUSE_RELU) {
              output_value = output_value > 0 ? output_value : 0;
            } else if (fuse_code == FUSE_RELU1) {
              output_value = std::min(std::max(static_cast<T>(0), output_value),
                                      static_cast<T>(1));
            } else if (fuse_code == FUSE_RELU6) {
              output_value = std::min(std::max(static_cast<T>(0), output_value),
                                      static_cast<T>(6));
            } else if (fuse_code == FUSE_NONE) {
            } else {
              return -1;
            }
            output_data[output_index] = output_value;
          }
        }
      }
    }
  }
  return 0;
}

int conv2d(const int8_t* input_data,
           const std::vector<int32_t>& input_shape,
           float input_scale,
           const int8_t* filter_data,
           const std::vector<int32_t>& filter_shape,
           const std::pair<const std::vector<float>, int>& filter_scales,
           const int32_t* bias_data,
           int pad_height_top,
           int pad_height_bottom,
           int pad_width_left,
           int pad_width_right,
           int stride_height,
           int stride_width,
           int dilation_height,
           int dilation_width,
           int group,
           FuseCode fuse_code,
           int8_t* output_data,
           float output_scale);

int conv2d(const int8_t* input_data,
           const std::vector<int32_t>& input_shape,
           float input_scale,
           const int8_t* filter_data,
           const std::vector<int32_t>& filter_shape,
           float filter_scale,
           const int32_t* bias_data,
           int pad_height_top,
           int pad_height_bottom,
           int pad_width_left,
           int pad_width_right,
           int stride_height,
           int stride_width,
           int dilation_height,
           int dilation_width,
           int group,
           FuseCode fuse_code,
           int8_t* output_data,
           float output_scale);

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
