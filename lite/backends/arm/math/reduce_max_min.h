/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <utility>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

enum class MaxMinType : bool { kMin = false, kMax = true };

template <typename DataType>
inline void reduce_one_line_max(const DataType* src, DataType* dst, int size) {
  DataType tmp = src[0];
  for (int i = 0; i < size; i++) {
    if (tmp <= src[i]) tmp = src[i];
  }
  *dst = tmp;
}

template <typename DataType>
inline void reduce_one_line_min(const DataType* src, DataType* dst, int size) {
  DataType tmp = src[0];
  for (int i = 0; i < size; i++) {
    if (tmp > src[i]) tmp = src[i];
  }
  *dst = tmp;
}

template <typename DataType>
void reduce_second_of_two(const DataType* src,
                          DataType* dst,
                          int first_in,
                          int second_in,
                          MaxMinType max_min_selector) {
  // max_min_selector == true, do reduce max; else do reduce min
  for (int j = 0; j < first_in; j++) {
    dst[j] = src[j * second_in];
    for (int k = 1; k < second_in; k++) {
      dst[j] = (src[j * second_in + k] <= dst[j]) ^
                       static_cast<bool>(max_min_selector)
                   ? src[j * second_in + k]
                   : dst[j];
    }
  }
}
template <typename DataType>
void reduce_first_of_two(const DataType* src,
                         DataType* dst,
                         int first_in,
                         int second_in,
                         MaxMinType max_min_selector) {
  // max_min_selector == true, do reduce max; else do reduce min
  for (int j = 0; j < second_in; j++) {
    dst[j] = src[j];
    for (int k = 1; k < first_in; k++) {
      dst[j] = (src[k * second_in + j] <= dst[j]) ^
                       static_cast<bool>(max_min_selector)
                   ? src[k * second_in + j]
                   : dst[j];
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
