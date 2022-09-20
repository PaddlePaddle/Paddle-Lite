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
#include "operation/math/broadcast.h"
#include "operation/math/dequantize.h"
#include "operation/math/quantize.h"
#include "operation/math/utility.h"

namespace nnadapter {
namespace operation {
namespace math {

template <typename T>
static int elementwise(ElementwiseTypeCode eltwise_type,
                       const T* input0_data,
                       const std::vector<int32_t>& input0_shape,
                       const T* input1_data,
                       const std::vector<int32_t>& input1_shape,
                       FuseCode fuse_code,
                       T* output_data) {
  if (!input0_data || !input1_data || !output_data) {
    return -1;
  }
  auto output_shape = shape_broadcast(input0_shape, input1_shape);
  auto output_count = shape_production(output_shape);
  std::vector<T> broadcasted_input0_data(output_count);
  broadcast<T>(
      input0_data, input0_shape, output_shape, broadcasted_input0_data.data());
  std::vector<T> broadcasted_input1_data(output_count);
  broadcast<T>(
      input1_data, input1_shape, output_shape, broadcasted_input1_data.data());
  if (eltwise_type == ADD) {
    for (int64_t i = 0; i < output_count; i++) {
      output_data[i] = broadcasted_input0_data[i] + broadcasted_input1_data[i];
    }
  } else if (eltwise_type == SUB) {
    for (int64_t i = 0; i < output_count; i++) {
      output_data[i] = broadcasted_input0_data[i] - broadcasted_input1_data[i];
    }
  } else if (eltwise_type == MUL) {
    for (int64_t i = 0; i < output_count; i++) {
      output_data[i] = broadcasted_input0_data[i] * broadcasted_input1_data[i];
    }
  } else {
    return -1;
  }
  if (fuse_code == FUSE_RELU) {
    for (int64_t i = 0; i < output_count; i++) {
      auto output_value = output_data[i];
      output_data[i] = output_value > 0 ? output_value : 0;
    }
  } else if (fuse_code == FUSE_RELU1) {
    for (int64_t i = 0; i < output_count; i++) {
      auto output_value = output_data[i];
      output_data[i] = std::min(std::max(static_cast<T>(0), output_value),
                                static_cast<T>(1));
    }
  } else if (fuse_code == FUSE_RELU6) {
    for (int64_t i = 0; i < output_count; i++) {
      auto output_value = output_data[i];
      output_data[i] = std::min(std::max(static_cast<T>(0), output_value),
                                static_cast<T>(6));
    }
  } else if (fuse_code == FUSE_NONE) {
  } else {
    return -1;
  }
  return 0;
}

int elementwise(ElementwiseTypeCode eltwise_type,
                const int8_t* input0_data,
                const std::vector<int32_t>& input0_shape,
                float input0_scale,
                const int8_t* input1_data,
                const std::vector<int32_t>& input1_shape,
                float input1_scale,
                FuseCode fuse_code,
                int8_t* output_data,
                float output_scale);

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
