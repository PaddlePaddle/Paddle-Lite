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
static int unary_activations(ActivationTypeCode act_type,
                             const T* input_data,
                             const std::vector<int32_t>& input_shape,
                             T* output_data) {
  if (!input_data || !output_data) {
    return -1;
  }
  auto input_count = shape_production(input_shape);
  if (act_type == RELU) {
    for (int64_t i = 0; i < input_count; i++) {
      output_data[i] = std::max(static_cast<T>(0), input_data[i]);
    }
    return 0;
  } else if (act_type == RELU6) {
    for (int64_t i = 0; i < input_count; i++) {
      output_data[i] = std::min(static_cast<T>(6),
                                std::max(static_cast<T>(0), input_data[i]));
    }
    return 0;
  }
  return -1;
}

int unary_activations(ActivationTypeCode act_type,
                      const int8_t* input_data,
                      const std::vector<int32_t>& input_shape,
                      float input_scale,
                      int8_t* output_data,
                      float output_scale);

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
