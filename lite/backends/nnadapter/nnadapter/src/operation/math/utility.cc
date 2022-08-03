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

#include "operation/math/utility.h"

namespace nnadapter {
namespace operation {
namespace math {

std::vector<int32_t> shape_slice(const std::vector<int32_t>& input_shape,
                                 int start,
                                 int end) {
  int input_rank = input_shape.size();
  start = start < 0 ? 0 : (start > input_rank ? input_rank : start);
  end = end < start ? start : (end > input_rank ? input_rank : end);
  return std::vector<int32_t>(input_shape.data() + start,
                              input_shape.data() + end);
}

int64_t shape_production(const std::vector<int32_t>& input_shape) {
  auto input_rank = input_shape.size();
  int64_t production = 1;
  for (size_t i = 0; i < input_rank; i++) {
    auto dimension = input_shape[i];
    production *= dimension;
  }
  return production;
}

std::vector<int32_t> shape_broadcast(const std::vector<int32_t>& input0_shape,
                                     const std::vector<int32_t>& input1_shape) {
  int input0_rank = input0_shape.size();
  int input1_rank = input1_shape.size();
  int output_rank = input0_rank > input1_rank ? input0_rank : input1_rank;
  std::vector<int32_t> output_shape(output_rank, 0);
  for (int i = 0; i < output_rank; i++) {
    int input0_idx = i - output_rank + input0_rank;
    int input0_dim = input0_idx < 0 ? 1 : input0_shape[input0_idx];
    int input1_idx = i - output_rank + input1_rank;
    int input1_dim = input1_idx < 0 ? 1 : input1_shape[input1_idx];
    if (input0_dim != 1 && input1_dim != 1 && input0_dim != input1_dim) break;
    output_shape[i] = input0_dim > input1_dim ? input0_dim : input1_dim;
  }
  return output_shape;
}

std::vector<int64_t> shape_strides(const std::vector<int32_t>& input_shape) {
  int64_t input_rank = input_shape.size();
  std::vector<int64_t> input_strides(input_rank);
  input_strides[input_rank - 1] = 1;
  for (int64_t i = input_rank - 2; i >= 0; i--) {
    input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
  }
  return input_strides;
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
