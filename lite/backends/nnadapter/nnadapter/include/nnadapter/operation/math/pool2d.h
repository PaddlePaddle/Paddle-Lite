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
#include <vector>
#include "operation/math/dequantize.h"
#include "operation/math/quantize.h"
#include "operation/math/utility.h"

namespace nnadapter {
namespace operation {
namespace math {

template <typename T>
static int average_pool2d(const T* input_data,
                          const std::vector<int32_t>& input_shape,
                          int kernel_height,
                          int kernel_width,
                          int pad_height_top,
                          int pad_height_bottom,
                          int pad_width_left,
                          int pad_width_right,
                          int stride_height,
                          int stride_width,
                          bool ceil_mode,
                          bool count_include_pad,
                          FuseCode fuse_code,
                          T* output_data) {
  if (!input_data || !output_data) {
    return -1;
  }
  auto input_rank = input_shape.size();
  if (input_rank != 4) {
    return -1;
  }
  auto batch_size = input_shape[0];
  auto input_channel_size = input_shape[1];
  auto input_height = input_shape[2];
  auto input_width = input_shape[3];
  auto output_channel_size = input_channel_size;
  auto output_height = (input_height + (pad_height_top + pad_height_bottom) -
                        kernel_height + (ceil_mode ? stride_height - 1 : 0)) /
                           stride_height +
                       1;
  auto output_width = (input_width + (pad_width_left + pad_width_right) -
                       kernel_width + (ceil_mode ? stride_width - 1 : 0)) /
                          stride_width +
                      1;
  for (int bs = 0; bs < batch_size; bs++) {
    for (int c = 0; c < input_channel_size; c++) {
      for (int h = 0; h < output_height; h++) {
        auto sh = h * stride_height;
        auto eh = sh + kernel_height;
        sh = (sh - pad_height_top) < 0 ? 0 : sh - pad_height_top;
        eh = (eh - pad_height_top) > input_height ? input_height
                                                  : eh - pad_height_top;
        for (int w = 0; w < output_width; w++) {
          auto sw = w * stride_width;
          auto ew = sw + kernel_width;
          sw = (sw - pad_width_left) < 0 ? 0 : sw - pad_width_left;
          ew = (ew - pad_width_left) > input_width ? input_width
                                                   : ew - pad_width_left;
          auto pooling_size = (ew - sw) * (eh - sh);
          if (pooling_size == 0) continue;
          float output_value = 0.f;
          for (int kh = sh; kh < eh; kh++) {
            for (int kw = sw; kw < ew; kw++) {
              auto input_index =
                  bs * input_channel_size * input_height * input_width +
                  c * input_height * input_width + kh * input_width + kw;
              auto input_value = input_data[input_index];
              if (kh == sh && kw == sw) {
                output_value = input_value;
              } else {
                output_value += input_value;
              }
            }
          }
          if (!count_include_pad) {
            output_value /= pooling_size;
          } else {
            output_value /= kernel_height * kernel_width;
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
          auto output_index =
              bs * output_channel_size * output_height * output_width +
              c * output_height * output_width + h * output_width + w;
          output_data[output_index] = output_value;
        }
      }
    }
  }
  return 0;
}

int average_pool2d(const int8_t* input_data,
                   const std::vector<int32_t>& input_shape,
                   float input_scale,
                   int kernel_height,
                   int kernel_width,
                   int pad_height_top,
                   int pad_height_bottom,
                   int pad_width_left,
                   int pad_width_right,
                   int stride_height,
                   int stride_width,
                   bool ceil_mode,
                   bool count_include_pad,
                   FuseCode fuse_code,
                   int8_t* output_data,
                   float output_scale);

template <typename T>
int max_pool2d(const T* input_data,
               const std::vector<int32_t>& input_shape,
               int kernel_height,
               int kernel_width,
               int pad_height_top,
               int pad_height_bottom,
               int pad_width_left,
               int pad_width_right,
               int stride_height,
               int stride_width,
               bool ceil_mode,
               bool return_indices,
               DataTypeCode indices_type,
               FuseCode fuse_code,
               T* output_data) {
  if (!input_data || !output_data) {
    return -1;
  }
  auto input_rank = input_shape.size();
  if (input_rank != 4) {
    return -1;
  }
  auto batch_size = input_shape[0];
  auto input_channel_size = input_shape[1];
  auto input_height = input_shape[2];
  auto input_width = input_shape[3];
  auto output_channel_size = input_channel_size;
  auto output_height = (input_height + (pad_height_top + pad_height_bottom) -
                        kernel_height + (ceil_mode ? stride_height - 1 : 0)) /
                           stride_height +
                       1;
  auto output_width = (input_width + (pad_width_left + pad_width_right) -
                       kernel_width + (ceil_mode ? stride_width - 1 : 0)) /
                          stride_width +
                      1;
  for (int bs = 0; bs < batch_size; bs++) {
    for (int c = 0; c < input_channel_size; c++) {
      for (int h = 0; h < output_height; h++) {
        auto sh = h * stride_height;
        auto eh = sh + kernel_height;
        sh = (sh - pad_height_top) < 0 ? 0 : sh - pad_height_top;
        eh = (eh - pad_height_top) > input_height ? input_height
                                                  : eh - pad_height_top;
        for (int w = 0; w < output_width; w++) {
          auto sw = w * stride_width;
          auto ew = sw + kernel_width;
          sw = (sw - pad_width_left) < 0 ? 0 : sw - pad_width_left;
          ew = (ew - pad_width_left) > input_width ? input_width
                                                   : ew - pad_width_left;
          auto pooling_size = (ew - sw) * (eh - sh);
          if (pooling_size == 0) continue;
          float output_value = 0.f;
          for (int kh = sh; kh < eh; kh++) {
            for (int kw = sw; kw < ew; kw++) {
              auto input_index =
                  bs * input_channel_size * input_height * input_width +
                  c * input_height * input_width + kh * input_width + kw;
              auto input_value = input_data[input_index];
              if (kh == sh && kw == sw) {
                output_value = input_value;
              } else {
                output_value =
                    output_value >= input_value ? output_value : input_value;
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
          auto output_index =
              bs * output_channel_size * output_height * output_width +
              c * output_height * output_width + h * output_width + w;
          output_data[output_index] = output_value;
        }
      }
    }
  }
  return 0;
}

int max_pool2d(const int8_t* input_data,
               const std::vector<int32_t>& input_shape,
               float input_scale,
               int kernel_height,
               int kernel_width,
               int pad_height_top,
               int pad_height_bottom,
               int pad_width_left,
               int pad_width_right,
               int stride_height,
               int stride_width,
               bool ceil_mode,
               bool return_indices,
               DataTypeCode indices_type,
               FuseCode fuse_code,
               int8_t* output_data,
               float output_scale);

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
