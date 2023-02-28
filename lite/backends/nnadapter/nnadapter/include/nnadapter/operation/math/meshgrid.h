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
#include "operation/math/utility.h"

namespace nnadapter {
namespace operation {
namespace math {

template <typename T>
static int meshgrid(const std::vector<T*>& input_datas,
                    const std::vector<std::vector<int32_t>>& input_shapes,
                    std::vector<T*> output_datas) {
  int32_t num = input_datas.size();
  std::vector<int32_t> shape(num);
  for (int32_t i = 0; i < num; ++i) {
    switch (input_shapes[i].size()) {
      case 1:
        shape[i] = input_shapes[i][0];
        break;
      default:
        return -1;
    }
  }
  for (int32_t i = 0; i < num; ++i) {
    T* dst = output_datas[i];

    std::vector<int32_t> view_shape(num, 1);
    view_shape[i] = shape[i];
    const T* src = input_datas[i];
    std::vector<int> bcast_dims(num);
    for (int32_t j = 0; j < num; j++) {
      bcast_dims[j] = shape[j];
    }
    bcast_dims[i] = 1;
    int inner_num = 1;
    int idx = num - 1;
    int outer_num = shape_production(shape_slice(view_shape, 0, idx));
    inner_num *= view_shape[idx];
    for (int j = 0; j < outer_num; ++j) {
      for (int k = 0; k < bcast_dims[idx]; ++k) {
        memcpy(dst + (j * bcast_dims[idx] + k) * inner_num,
               src + j * inner_num,
               sizeof(T) * inner_num);
      }
    }
    inner_num *= bcast_dims[idx];
    for (int idx = num - 2; idx >= 0; --idx) {
      int outer_num = shape_production(shape_slice(view_shape, 0, idx));
      inner_num *= view_shape[idx];
      for (int j = outer_num - 1; j >= 0; --j) {
        for (int k = bcast_dims[idx] - 1; k >= 0; --k) {
          memcpy(dst + (j * bcast_dims[idx] + k) * inner_num,
                 dst + j * inner_num,
                 sizeof(T) * inner_num);
        }
      }
      inner_num *= bcast_dims[idx];
    }
  }
  return 0;
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
