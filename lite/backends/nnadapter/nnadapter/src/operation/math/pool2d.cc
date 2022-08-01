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

#include "operation/math/pool2d.h"
#include "operation/math/dequantize.h"
#include "operation/math/quantize.h"

namespace nnadapter {
namespace operation {
namespace math {

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
                   float output_scale) {
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
  std::vector<int32_t> output_shape = {
      batch_size, output_channel_size, output_height, output_width};
  // Dequantize input data
  auto input_count = shape_production(input_shape);
  std::vector<float> dequantized_input_data(input_count);
  int status = dequantize(
      input_data, input_shape, input_scale, dequantized_input_data.data());
  if (status) return status;
  // Prepare dequantized output data
  auto output_count = shape_production(output_shape);
  std::vector<float> dequantized_output_data(output_count);
  status = average_pool2d<float>(dequantized_input_data.data(),
                                 input_shape,
                                 kernel_height,
                                 kernel_width,
                                 pad_height_top,
                                 pad_height_bottom,
                                 pad_width_left,
                                 pad_width_right,
                                 stride_height,
                                 stride_width,
                                 ceil_mode,
                                 count_include_pad,
                                 fuse_code,
                                 dequantized_output_data.data());
  if (status) return status;
  // Quantize output data
  return quantize(
      dequantized_output_data.data(), output_shape, output_scale, output_data);
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
               float output_scale) {
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
  std::vector<int32_t> output_shape = {
      batch_size, output_channel_size, output_height, output_width};
  // Dequantize input data
  auto input_count = shape_production(input_shape);
  std::vector<float> dequantized_input_data(input_count);
  int status = dequantize(
      input_data, input_shape, input_scale, dequantized_input_data.data());
  if (status) return status;
  // Prepare dequantized output data
  auto output_count = shape_production(output_shape);
  std::vector<float> dequantized_output_data(output_count);
  status = max_pool2d<float>(dequantized_input_data.data(),
                             input_shape,
                             kernel_height,
                             kernel_width,
                             pad_height_top,
                             pad_height_bottom,
                             pad_width_left,
                             pad_width_right,
                             stride_height,
                             stride_width,
                             ceil_mode,
                             return_indices,
                             indices_type,
                             fuse_code,
                             dequantized_output_data.data());
  if (status) return status;
  // Quantize output data
  return quantize(
      dequantized_output_data.data(), output_shape, output_scale, output_data);
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
