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

#include <vector>
#include "operation/math/utility.h"

namespace nnadapter {
namespace operation {
namespace math {

template <typename T>
static int expand(const T* input_data,
                  const std::vector<int32_t>& input_shape,
                  T* output_data,
                  const std::vector<int32_t>& output_shape) {
  std::vector<int> in_stride(input_shape.size(), 1);
  std::vector<int> out_stride(output_shape.size(), 1);
  for (int i = input_shape.size() - 2; i >= 0; --i) {
    in_stride[i] = input_shape[i + 1] * in_stride[i + 1];
  }
  for (int i = output_shape.size() - 2; i >= 0; --i) {
    out_stride[i] = output_shape[i + 1] * out_stride[i + 1];
  }
  int out_size = shape_production(output_shape);
  for (int out_id = 0; out_id < out_size; ++out_id) {
    int in_id = 0;
    for (int i = input_shape.size() - 1; i >= 0; --i) {
      int in_j = (out_id / out_stride[i]) % input_shape[i];
      in_id += in_j * in_stride[i];
    }
    output_data[out_id] = input_data[in_id];
  }
  return 0;
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
