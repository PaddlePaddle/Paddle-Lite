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
#include <utility>
#include <vector>
#include "operation/math/dequantize.h"
#include "operation/math/quantize.h"
#include "operation/math/utility.h"

namespace nnadapter {
namespace operation {
namespace math {

template <typename T>
static int fully_connected(const T* input_data,
                           const std::vector<int32_t>& input_shape,
                           const T* weight_data,
                           const std::vector<int32_t>& weight_shape,
                           const T* bias_data,
                           FuseCode fuse_code,
                           T* output_data) {
  if (!input_data || !weight_data || !output_data) {
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
  for (int m = 0; m < batch_size; m++) {
    for (int n = 0; n < num_units; n++) {
      T output_value = bias_data ? bias_data[n] : 0;
      for (int k = 0; k < input_size; k++) {
        output_value +=
            input_data[m * input_size + k] * weight_data[n * input_size + k];
      }
      if (fuse_code == FUSE_RELU) {
        output_value = output_value > 0 ? output_value : 0;
      } else if (fuse_code == FUSE_RELU1) {
        output_value = std::min(std::max(static_cast<T>(0), output_value),
                                static_cast<T>(1));
      } else if (fuse_code == FUSE_RELU6) {
        output_value = std::min(std::max(static_cast<T>(0), output_value),
                                static_cast<T>(6));
      } else if (fuse_code == FUSE_NONE) {
      } else {
        return -1;
      }
      auto output_index = m * num_units + n;
      output_data[output_index] = output_value;
    }
  }
  return 0;
}

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
    float output_scale);

int fully_connected(const int8_t* input_data,
                    const std::vector<int32_t>& input_shape,
                    float input_scale,
                    const int8_t* weight_data,
                    const std::vector<int32_t>& weight_shape,
                    float weight_scale,
                    const int32_t* bias_data,
                    FuseCode fuse_code,
                    int8_t* output_data,
                    float output_scale);

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
