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

#include <algorithm>
#include <utility>
#include <vector>
#include "operation/math/utility.h"

namespace nnadapter {
namespace operation {
namespace math {

template <typename T>
static int dequantize(
    const T* input_data,
    const std::vector<int32_t>& input_shape,
    const std::pair<const std::vector<float>, int>& input_scales,
    float* output_data) {
  if (!input_data || input_shape.empty() || input_scales.first.empty() ||
      !output_data) {
    return -1;
  }
  int quant_bits = sizeof(T) * 8;
  auto dtype_max = static_cast<int>((1 << (quant_bits - 1)) - 1);
  auto dtype_min = static_cast<int>(0 - dtype_max);
  auto input_rank = input_shape.size();
  auto input_count = shape_production(input_shape);
  auto scale_count = input_scales.first.size();
  auto channel_dim = input_scales.second;
  if (scale_count > 1 && channel_dim < 0) {
    return -1;
  }
  int64_t outer_count = input_count;
  int64_t inner_count = 1;
  if (scale_count > 1 && channel_dim >= 0) {
    auto channel_count = input_shape[channel_dim];
    if (channel_count != scale_count) {
      return -1;
    }
    outer_count = shape_production(shape_slice(input_shape, 0, channel_dim));
    inner_count =
        shape_production(shape_slice(input_shape, channel_dim + 1, input_rank));
  }
  for (int64_t i = 0; i < outer_count; i++) {
    for (size_t j = 0; j < scale_count; j++) {
      for (int64_t k = 0; k < inner_count; k++) {
        auto index = i * scale_count * inner_count + j * inner_count + k;
        output_data[index] =
            std::min(std::max(static_cast<int>(input_data[index]), dtype_min),
                     dtype_max) *
            input_scales.first[j];
      }
    }
  }
  return 0;
}

template <typename T>
static int dequantize(const T* input_data,
                      const std::vector<int32_t>& input_shape,
                      float input_scale,
                      float* output_data) {
  return dequantize<T>(input_data,
                       input_shape,
                       std::make_pair(std::vector<float>({input_scale}), -1),
                       output_data);
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
