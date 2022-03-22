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
#include <vector>
#include "operation/math/utility.h"

namespace nnadapter {
namespace operation {
namespace math {

template <typename T>
static int dequantize(const T* input_data,
                      const std::vector<int32_t>& input_shape,
                      const std::vector<float>& input_scales,
                      float* output_data) {
  bool per_layer = input_scales.size() == 1;
  int quant_bits = sizeof(T) * 8;
  auto dtype_max = static_cast<int>((1 << (quant_bits - 1)) - 1);
  auto dtype_min = static_cast<int>(0 - dtype_max);
  auto input_data_count = production_of_shape(input_shape);
  for (int64_t i = 0; i < input_data_count; i++) {
    int scale_index = per_layer ? 0 : i;
    output_data[i] =
        std::min(std::max(static_cast<int>(input_data[i]), dtype_min),
                 dtype_max) *
        input_scales[scale_index];
  }
  return 0;
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
