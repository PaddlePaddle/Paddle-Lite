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
#include <vector>
#include "operation/math/dequantize.h"
#include "operation/math/quantize.h"
#include "operation/math/utility.h"

namespace nnadapter {
namespace operation {
namespace math {

template <typename T>
static int batch_normalization(const T* input_data,
                               const std::vector<int32_t>& input_shape,
                               float* scale_data,
                               float* bias_data,
                               float* mean_data,
                               float* variance_data,
                               float epsilon,
                               T* output_data) {
  if (!input_data || !scale_data || !bias_data || !mean_data ||
      !variance_data || !output_data) {
    return -1;
  }
  auto input_rank = input_shape.size();
  auto batch_size = input_shape[0];
  auto channel_size = input_shape[1];
  auto inner_size = shape_production(shape_slice(input_shape, 2, input_rank));
  for (int bs = 0; bs < batch_size; bs++) {
    for (int c = 0; c < channel_size; c++) {
      for (int i = 0; i < inner_size; i++) {
        *output_data++ = scale_data[c] * (*input_data++ - mean_data[c]) /
                             sqrt(variance_data[c] + epsilon) +
                         bias_data[c];
      }
    }
  }
  return 0;
}

int batch_normalization(const int8_t* input_data,
                        const std::vector<int32_t>& input_shape,
                        float input_scale,
                        float* scale_data,
                        float* bias_data,
                        float* mean_data,
                        float* variance_data,
                        float epsilon,
                        int8_t* output_data,
                        float output_scale);

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
