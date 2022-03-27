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

#include "operation/math/fully_connected.h"
#include "operation/math/dequantize.h"
#include "operation/math/quantize.h"

namespace nnadapter {
namespace operation {
namespace math {

int fully_connected(
    const int8_t* input_data,
    const std::vector<int32_t>& input_shape,
    float input_scale,
    const int8_t* weight_data,
    const std::vector<int32_t>& weight_shape,
    const std::pair<const std::vector<float>, int>& weight_scales,
    const int32_t* bias_data,
    FuseCode fuse_code,
    int8_t* output_data,
    float output_scale) {
  if (!input_data || !weight_data) {
    return -1;
  }
  auto input_rank = input_shape.size();
  auto weight_rank = weight_shape.size();
  if (input_rank < 2 || weight_rank != 2) {
    return -1;
  }
  auto input_size = input_shape[input_rank - 1];
  auto batch_size = shape_production(input_shape) / input_size;
  auto num_units = weight_shape[0];
  if (input_size != weight_shape[1]) {
    return -1;
  }
  std::vector<int32_t> output_shape = {static_cast<int32_t>(batch_size),
                                       num_units};
  // Dequantize input data
  auto input_count = shape_production(input_shape);
  std::vector<float> dequantized_input_data(input_count);
  int status = dequantize(
      input_data, input_shape, input_scale, dequantized_input_data.data());
  if (status) return status;
  // Dequantize weight data
  auto weight_count = shape_production(weight_shape);
  std::vector<float> dequantized_weight_data(weight_count);
  status = dequantize(
      weight_data, weight_shape, weight_scales, dequantized_weight_data.data());
  if (status) return status;
  // Dequantize bias data
  std::vector<float> dequantized_bias_data;
  if (bias_data) {
    std::vector<float> bias_scales;
    for (auto weight_scale : weight_scales.first) {
      bias_scales.push_back(input_scale * weight_scale);
    }
    dequantized_bias_data.resize(num_units);
    status =
        dequantize(bias_data,
                   {num_units},
                   std::make_pair(bias_scales, bias_scales.size() > 0 ? 0 : -1),
                   dequantized_bias_data.data());
    if (status) return status;
  }
  // Prepare dequantized output data
  auto output_count = shape_production(output_shape);
  std::vector<float> dequantized_output_data(output_count);
  status = fully_connected<float>(
      dequantized_input_data.data(),
      input_shape,
      dequantized_weight_data.data(),
      weight_shape,
      dequantized_bias_data.size() > 0 ? dequantized_bias_data.data() : nullptr,
      fuse_code,
      dequantized_output_data.data());
  if (status) return status;
  // Quantize output data
  return quantize(
      dequantized_output_data.data(), output_shape, output_scale, output_data);
}

int fully_connected(const int8_t* input_data,
                    const std::vector<int32_t>& input_shape,
                    float input_scale,
                    const int8_t* weight_data,
                    const std::vector<int32_t>& weight_shape,
                    float weight_scale,
                    const int32_t* bias_data,
                    FuseCode fuse_code,
                    int8_t* output_data,
                    float output_scale) {
  return fully_connected(input_data,
                         input_shape,
                         input_scale,
                         weight_data,
                         weight_shape,
                         std::make_pair(std::vector<float>({weight_scale}), -1),
                         bias_data,
                         fuse_code,
                         output_data,
                         output_scale);
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
