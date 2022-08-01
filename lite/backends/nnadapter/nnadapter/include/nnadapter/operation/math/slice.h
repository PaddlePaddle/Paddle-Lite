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
static int slice(const T* input_data,
                 const std::vector<int32_t>& input_shape,
                 uint32_t axes_count,
                 int32_t* axes,
                 int32_t* starts,
                 int32_t* ends,
                 int32_t* steps,
                 T* output_data) {
  std::vector<int32_t> output_shape(input_shape);
  std::vector<int> real_starts(input_shape.size(), 0);
  std::vector<int> real_ends(input_shape.size(), 0);
  std::vector<int> real_step(input_shape.size(), 0);
  for (size_t i = 0; i < input_shape.size(); i++) {
    real_ends[i] = input_shape[i];
  }
  for (size_t i = 0; i < axes_count; i++) {
    int dim_value = input_shape[axes[i]];
    if (dim_value > 0) {
      int start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
      int end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
      int step = std::abs(steps[i]);
      start = std::max(start, 0);
      end = std::max(end, 0);
      end = std::min(end, dim_value);
      output_shape[axes[i]] = (std::abs(end - start) + step - 1) / step;
      real_starts[axes[i]] = start;
      real_ends[axes[i]] = end;
      real_step[axes[i]] = step;
    }
  }
  const int length = input_shape.size();
  std::vector<int> dst_step(length);
  for (size_t i = 0; i < input_shape.size(); ++i) {
    dst_step[i] = 1;
  }
  std::vector<int> src_step(length);
  for (size_t i = 0; i < input_shape.size(); ++i) {
    src_step[i] = 1;
  }
  int out_num = output_shape[input_shape.size() - 1];
  for (int i = input_shape.size() - 2; i >= 0; i--) {
    dst_step[i] = output_shape[i + 1] * dst_step[i + 1];
    src_step[i] = input_shape[i + 1] * src_step[i + 1];
    out_num *= output_shape[i];
  }

  for (int dst_id = 0; dst_id < out_num; dst_id++) {
    int src_id = 0;
    int index_id = dst_id;
    for (size_t j = 0; j < output_shape.size(); j++) {
      int cur_id = index_id / dst_step[j];
      index_id = index_id % dst_step[j];
      src_id += (cur_id * real_step[j] + real_starts[j]) * src_step[j];
    }
    output_data[dst_id] = input_data[src_id];
  }
  return 0;
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
