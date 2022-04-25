// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <cmath>
#include <limits>
#include <vector>
#include "operation/math/dequantize.h"
#include "operation/math/quantize.h"
#include "operation/math/utility.h"

namespace nnadapter {
namespace operation {
namespace math {

template <typename T>
static int log_softmax(const T* input_data,
                       const std::vector<int32_t>& input_shape,
                       int axis,
                       T* output_data) {
  if (!input_data || !output_data) {
    return -1;
  }
  auto input_rank = input_shape.size();
  if (axis < 0) {
    axis += input_rank;
  }
  auto axis_count = input_shape[axis];
  auto outer_count = shape_production(shape_slice(input_shape, 0, axis));
  auto inner_count =
      shape_production(shape_slice(input_shape, axis + 1, input_rank));
  auto compute_count = outer_count * inner_count;
  for (int64_t i = 0; i < compute_count; i++) {
    auto inner_index = i % inner_count;
    auto outer_index = (i / inner_count) * axis_count;
    auto start = outer_index * inner_count + inner_index;
    auto offset = start;
    auto max_value = std::numeric_limits<T>::lowest();
    for (int j = 0; j < axis_count; j++) {
      max_value =
          input_data[offset] > max_value ? input_data[offset] : max_value;
      offset += inner_count;
    }
    offset = start;
    T sum_value = 0;
    for (int j = 0; j < axis_count; j++) {
      output_data[offset] = std::exp(input_data[offset] - max_value);
      sum_value += output_data[offset];
      offset += inner_count;
    }
    offset = start;
    for (int j = 0; j < axis_count; j++) {
      output_data[offset] /= sum_value;
      output_data[offset] = std::log(output_data[offset]);
      offset += inner_count;
    }
  }
  return 0;
}

int log_softmax(const int8_t* input_data,
                const std::vector<int32_t>& input_shape,
                float input_scale,
                int axis,
                int8_t* output_data,
                float output_scale);

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
