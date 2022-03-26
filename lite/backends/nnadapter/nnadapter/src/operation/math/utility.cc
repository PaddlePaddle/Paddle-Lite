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

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
