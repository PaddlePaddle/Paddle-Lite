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

#include "lite/backends/arm/math/reduce_max_min.h"
#include <utility>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
void reduce_second_of_two<float>(const float* src,
                                 float* dst,
                                 int first_in,
                                 int second_in,
                                 MaxMinType max_min_selector) {
  // max_min_selector == true, do reduce max; else do reduce min
  for (int j = 0; j < second_in; j++) {
    dst[j * first_in] = src[j * first_in];
    for (int k = 1; k < first_in; k++) {
      dst[j * first_in] = (src[j * first_in + k] <= dst[j * first_in]) ^
                                  static_cast<bool>(max_min_selector)
                              ? src[j * first_in + k]
                              : dst[j * first_in];
    }
  }
}

template <>
void reduce_first_of_two<float>(const float* src,
                                float* dst,
                                int first_in,
                                int second_in,
                                MaxMinType max_min_selector) {
  // max_min_selector == true, do reduce max; else do reduce min
  for (int j = 0; j < first_in; j++) {
    dst[j] = src[j];
    for (int k = 1; k < second_in; k++) {
      dst[j] = (src[j + k * first_in] <= dst[j]) ^
                       static_cast<bool>(max_min_selector)
                   ? src[j + k * first_in]
                   : dst[j];
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
