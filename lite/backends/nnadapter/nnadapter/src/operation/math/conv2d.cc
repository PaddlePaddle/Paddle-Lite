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

#include "operation/math/conv2d.h"
#include "operation/math/dequantize.h"
#include "operation/math/quantize.h"

namespace nnadapter {
namespace operation {
namespace math {

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
           float output_scale) {
  if (!input_data || !filter_data) {
    return -1;
  }
  auto input_rank = input_shape.size();
  auto filter_rank = filter_shape.size();
  if (input_rank != 4 || filter_rank != 4) {
    return -1;
  }
  auto batch_size = input_shape[0];
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
  std::vector<int32_t> output_shape = {
      batch_size, output_channel_size, output_height, output_width};
  // Dequantize input data
  auto input_count = shape_production(input_shape);
  std::vector<float> dequantized_input_data(input_count);
  int status = dequantize(
      input_data, input_shape, input_scale, dequantized_input_data.data());
  if (status) return status;
  // Dequantize filter data
  auto filter_count = shape_production(filter_shape);
  std::vector<float> dequantized_filter_data(filter_count);
  status = dequantize(
      filter_data, filter_shape, filter_scales, dequantized_filter_data.data());
  if (status) return status;
  // Dequantize bias data
  std::vector<float> dequantized_bias_data;
  if (bias_data) {
    std::vector<float> bias_scales;
    for (auto filter_scale : filter_scales.first) {
      bias_scales.push_back(input_scale * filter_scale);
    }
    dequantized_bias_data.resize(output_channel_size);
    status =
        dequantize(bias_data,
                   {output_channel_size},
                   std::make_pair(bias_scales, bias_scales.size() > 0 ? 0 : -1),
                   dequantized_bias_data.data());
    if (status) return status;
  }
  // Prepare dequantized output data
  auto output_count = shape_production(output_shape);
  std::vector<float> dequantized_output_data(output_count);
  status = conv2d<float>(
      dequantized_input_data.data(),
      input_shape,
      dequantized_filter_data.data(),
      filter_shape,
      dequantized_bias_data.size() > 0 ? dequantized_bias_data.data() : nullptr,
      pad_height_top,
      pad_height_bottom,
      pad_width_left,
      pad_width_right,
      stride_height,
      stride_width,
      dilation_height,
      dilation_width,
      group,
      fuse_code,
      dequantized_output_data.data());
  if (status) return status;
  // Quantize output data
  return quantize(
      dequantized_output_data.data(), output_shape, output_scale, output_data);
}

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
           float output_scale) {
  return conv2d(input_data,
                input_shape,
                input_scale,
                filter_data,
                filter_shape,
                std::make_pair(std::vector<float>({filter_scale}), -1),
                bias_data,
                pad_height_top,
                pad_height_bottom,
                pad_width_left,
                pad_width_right,
                stride_height,
                stride_width,
                dilation_height,
                dilation_width,
                group,
                fuse_code,
                output_data,
                output_scale);
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
