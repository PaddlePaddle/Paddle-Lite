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

#include <cstring>
#include <memory>
#include <vector>

namespace nnadapter {
namespace operation {
namespace math {

template <typename T>
static int cum_sum(const T* input_data,
                   const std::vector<int>& input_shape,
                   T* output_data,
                   int axis,
                   bool reverse,
                   bool exclusive) {
  axis = axis < 0 ? axis + input_shape.size() : axis;
  int pre = 1;
  for (int i = 0; i < axis; i++) {
    pre *= input_shape[i];
  }
  int count = input_shape[axis];
  int post = 1;
  for (int i = axis + 1; i < input_shape.size(); i++) {
    post *= i;
  }
  if (reverse) {
    if (exclusive) {
      for (int i = 0; i < pre; i++) {
        for (int j = 0; j < post; j++) {
          int step = i * count * post + j;
          const T* src = input_data + step;
          T* dst = output_data + step;
          dst[(count - 1) * post] = 0;
          int p = 1;
          for (int k = count - 1; k > 0; k--, p++) {
            dst[(k - 1) * post] = src[k * post] + dst[k * post];
          }
        }
      }
    } else {
      for (int i = 0; i < pre; i++) {
        for (int j = 0; j < post; j++) {
          int step = i * count * post + j;
          const T* src = input_data + step;
          T* dst = output_data + step;
          dst[(count - 1) * post] = src[(count - 1) * post];
          int p = 1;
          for (int k = count - 2; k >= 0; k--, p++) {
            dst[k * post] = src[k * post] + dst[(k + 1) * post];
          }
        }
      }
    }
  } else {
    if (exclusive) {
      for (int i = 0; i < pre; i++) {
        for (int j = 0; j < post; j++) {
          int step = i * count * post + j;
          const T* src = input_data + step;
          T* dst = output_data + step;
          dst[0] = 0;
          for (int k = 0; k < count - 1; k++) {
            dst[(k + 1) * post] = src[k * post] + dst[k * post];
          }
        }
      }
    } else {
      for (int i = 0; i < pre; i++) {
        for (int j = 0; j < post; j++) {
          int step = i * count * post + j;
          const T* src = input_data + step;
          T* dst = output_data + step;
          dst[0] = src[0];
          for (int k = 1; k < count; k++) {
            dst[k * post] = src[k * post] + dst[(k - 1) * post];
          }
        }
      }
    }
  }
  return 0;
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
