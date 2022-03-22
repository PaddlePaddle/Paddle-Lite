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
static void softmax(const T* input_data_ptr,
                    const std::vector<int32_t>& input_shape,
                    int32_t axis,
                    T* output_data_ptr) {
  auto input_rank = input_shape.size();
  if (axis < 0) {
    axis += input_rank;
  }
  auto axis_count = input_shape[axis];
  auto outer_count = production_of_shape(slice_of_shape(input_shape, 0, axis));
  auto inner_count =
      production_of_shape(slice_of_shape(input_shape, axis + 1, input_rank));
  auto compute_count = outer_count * inner_count;
  for (int64_t i = 0; i < compute_count; i++) {
    auto inner_index = i % inner_count;
    auto outer_index = (i / inner_count) * axis_count;
    auto start = outer_index * inner_count + inner_index;
    auto offset = start;
    auto max_data = std::numeric_limits<T>::lowest();
    for (int j = 0; j < axis_count; j++) {
      max_data =
          input_data_ptr[offset] > max_data ? input_data_ptr[offset] : max_data;
      offset += inner_count;
    }
    offset = start;
    T sum_data = 0;
    for (int j = 0; j < axis_count; j++) {
      output_data_ptr[offset] = exp(input_data_ptr[offset] - max_data);
      sum_data += output_data_ptr[offset];
      offset += inner_count;
    }
    offset = start;
    for (int j = 0; j < axis_count; j++) {
      output_data_ptr[offset] /= sum_data;
      offset += inner_count;
    }
  }
}

void softmax(const int8_t* input_data_ptr,
             const std::vector<int32_t>& input_shape,
             float* input_scale_ptr,
             size_t input_scale_count,
             int32_t axis,
             int8_t* output_data_ptr,
             float* output_scale_ptr,
             size_t output_scale_count);

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
