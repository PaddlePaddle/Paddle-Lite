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

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

namespace nnadapter {
namespace operation {
namespace math {

template <typename T>
static int split(const T* input_data,
                 const std::vector<int32_t>& input_shape,
                 int32_t* sections,
                 uint32_t sections_count,
                 int32_t axis,
                 const std::vector<T*>& output_datas) {
  int stride = 0;
  for (size_t i = 0; i < sections_count; i++) {
    std::vector<int32_t> output_shape(input_shape);
    output_shape[axis] = sections[i];
    auto* output_data = output_datas[i];

    int n = 1;
    for (int i = 0; i < axis; i++) {
      n *= static_cast<int>(input_shape[i]);
    }
    int step_in = input_shape[axis];
    int step_out = output_shape[axis];
    for (int i = static_cast<int>(output_shape.size()) - 1; i > axis; i--) {
      step_in *= static_cast<int>(output_shape[i]);
      step_out *= static_cast<int>(output_shape[i]);
    }

    auto* in_ptr = input_data + stride;
    for (int i = 0; i < n; i++) {
      std::memcpy(output_data, in_ptr, sizeof(T) * step_out);
      in_ptr += step_in;
      output_data += step_out;
    }

    stride += step_out;
  }

  return 0;
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
