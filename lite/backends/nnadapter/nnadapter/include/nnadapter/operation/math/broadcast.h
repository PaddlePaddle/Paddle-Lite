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
static int broadcast(const T* input_data,
                     const std::vector<int32_t>& input_shape,
                     const std::vector<int32_t>& output_shape,
                     T* output_data) {
  if (!input_data || !output_data) {
    return -1;
  }
  int input_rank = input_shape.size();
  int output_rank = output_shape.size();
  int64_t output_count = shape_production(output_shape);
  int distance = output_rank - input_rank;
  auto output_strides = shape_strides(output_shape);
  auto input_strides = shape_strides(input_shape);
  for (int64_t i = 0; i < output_count; i++) {
    int64_t index = 0;
    int64_t remain = i;
    for (int j = 0; j < output_rank; j++) {
      int dimension = remain / output_strides[j];
      remain = remain % output_strides[j];
      if (j >= distance) {
        if (dimension >= input_shape[j - distance]) {
          dimension = 0;
        }
        index += dimension * input_strides[j - distance];
      }
    }
    output_data[i] = input_data[index];
  }
  return -1;
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
