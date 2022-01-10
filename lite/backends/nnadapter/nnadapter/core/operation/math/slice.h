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

namespace nnadapter {
namespace operation {
namespace math {

template <typename T>
static void slice(const T* input_data_ptr,
                  const std::vector<int32_t>& input_shapes,
                  uint32_t axes_count,
                  int32_t* axes,
                  int32_t* starts,
                  int32_t* ends,
                  T* output_data_ptr) {
  std::vector<int32_t> out_shapes(input_shapes);
  std::vector<int> real_starts(input_shapes.size(), 0);
  std::vector<int> real_ends(input_shapes.size(), 0);
  std::vector<int> real_step(input_shapes.size(), 0);
  for (size_t i = 0; i < input_shapes.size(); i++) {
    real_ends[i] = input_shapes[i];
  }
  for (size_t i = 0; i < axes_count; i++) {
    int dim_value = input_shapes[axes[i]];
    if (dim_value > 0) {
      int start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
      int end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
      start = std::max(start, 0);
      end = std::max(end, 0);
      end = std::min(end, dim_value);
      out_shapes[axes[i]] = end - start;
      real_starts[axes[i]] = start;
      real_ends[axes[i]] = end;
    }
  }
  const int length = input_shapes.size();
  std::vector<int> dst_step(length);
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    dst_step[i] = 1;
  }
  std::vector<int> src_step(length);
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    src_step[i] = 1;
  }
  int out_num = out_shapes[input_shapes.size() - 1];
  for (int i = input_shapes.size() - 2; i >= 0; i--) {
    dst_step[i] = out_shapes[i + 1] * dst_step[i + 1];
    src_step[i] = input_shapes[i + 1] * src_step[i + 1];
    out_num *= out_shapes[i];
  }

  for (int dst_id = 0; dst_id < out_num; dst_id++) {
    int src_id = 0;
    int index_id = dst_id;
    for (size_t j = 0; j < out_shapes.size(); j++) {
      int cur_id = index_id / dst_step[j];
      index_id = index_id % dst_step[j];
      src_id += (cur_id + real_starts[j]) * src_step[j];
    }
    output_data_ptr[dst_id] = input_data_ptr[src_id];
  }
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
