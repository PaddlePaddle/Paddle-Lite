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

#include <cstring>
#include <memory>
#include <vector>

namespace nnadapter {
namespace operation {
namespace math {

template <typename T>
static int concat(const std::vector<T*>& input_datas,
                  const std::vector<std::vector<int32_t>>& input_shapes,
                  const int axis,
                  T* output_data) {
  size_t num = input_datas.size();
  auto input_dim_0 = input_shapes[0];
  int64_t input_size = 1;
  int64_t num_cancats = 1;
  for (int i = axis + 1; i < input_dim_0.size(); i++) {
    input_size *= input_dim_0[i];
  }
  for (int i = 0; i < axis; i++) {
    num_cancats *= input_dim_0[i];
  }

  std::vector<int32_t> output_dims = input_dim_0;
  for (uint32_t i = 1; i < num; i++) {
    for (uint32_t j = 0; j < output_dims.size(); j++) {
      if (j == axis) {
        output_dims[j] += input_shapes[i][j];
      }
    }
  }
  auto* dst_ptr = output_data;
  const int out_axis = output_dims[axis];
  int64_t offset_axis = 0;
  int64_t out_sum = out_axis * input_size;
  for (int n = 0; n < num; n++) {
    auto dims = input_shapes[n];
    auto* src_ptr = input_datas[n];
    int64_t in_axis = dims[axis];
    auto* dout_ptr = dst_ptr + offset_axis * input_size;
    int64_t in_sum = in_axis * input_size;
    for (int i = 0; i < num_cancats; i++) {
      std::memcpy(dout_ptr, src_ptr, sizeof(T) * in_sum);
      dout_ptr += out_sum;
      src_ptr += in_sum;
    }
    offset_axis += in_axis;
  }
  return 0;
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
