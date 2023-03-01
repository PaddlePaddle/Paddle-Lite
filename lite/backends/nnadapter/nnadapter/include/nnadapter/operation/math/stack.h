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
#include "operation/math/utility.h"

namespace nnadapter {
namespace operation {
namespace math {

template <typename T>
static int stack(const std::vector<T*>& input_datas,
                 const std::vector<std::vector<int32_t>>& input_shapes,
                 int axis,
                 T* output_data) {
  if (axis < 0) axis += (input_shapes[0].size() + 1);
  int num = input_datas.size();
  auto* y_data = output_data;
  std::vector<const T*> x_datas(num);
  for (int i = 0; i < num; i++) x_datas[i] = input_datas[i];

  int pre = 1, post = 1;
  auto dim = input_shapes[0];
  for (int i = 0; i < axis; ++i) pre *= dim[i];
  for (int i = axis; i < dim.size(); ++i) post *= dim[i];

  auto x_data_arr = x_datas.data();

  int x_offset = 0;
  int y_offset = 0;
  for (int i = 0; i < pre; i++) {
    for (int j = 0; j < num; j++) {
      memcpy(y_data + y_offset, x_data_arr[j] + x_offset, post * sizeof(T));
      y_offset += post;
    }
    x_offset += post;
  }
  return 0;
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
