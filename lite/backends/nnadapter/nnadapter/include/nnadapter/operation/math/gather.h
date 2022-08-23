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
#include <vector>
#include "operation/math/utility.h"

namespace nnadapter {
namespace operation {
namespace math {

template <typename DATA_T, typename ID_T = int32_t>
static int gather(const DATA_T* input_data,
                  const std::vector<int32_t>& input_shape,
                  const ID_T* index_data,
                  const std::vector<int32_t>& index_shape,
                  int32_t axis,
                  DATA_T* output_data) {
  int inner_dim_size = 1;
  int outer_dim_size = 1;
  int input_size = shape_production(input_shape);
  int index_size = shape_production(index_shape);
  std::vector<int32_t> out_shape;
  for (int i = 0; i < axis; i++) {
    inner_dim_size *= input_shape[i];
    out_shape.push_back(input_shape[i]);
  }
  out_shape.push_back(index_size);
  for (int i = axis + 1; i < input_shape.size(); i++) {
    outer_dim_size *= input_shape[i];
    out_shape.push_back(input_shape[i]);
  }

  int output_index = 0;
  for (int i = 0; i < inner_dim_size; i++) {
    for (int j = 0; j < index_size; j++) {
      for (int k = 0; k < outer_dim_size; k++) {
        int index = k + index_data[j] * outer_dim_size +
                    (i * input_size / inner_dim_size);
        output_data[output_index] = input_data[index];
        output_index++;
      }
    }
  }

  return 0;
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
