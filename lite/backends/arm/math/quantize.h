// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <cmath>
#include <cstdint>
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

inline float GetScale(float threshold, int bit_length) {
  return threshold / ((1 << (bit_length - 1)) - 1);
}

float FindAbsMax(const float* input, int size);

template <typename T>
void QuantizeTensor(const float* input, T* output, int size, float scale) {
  auto quant_func = [scale](float x) {
    return static_cast<T>(std::round(x / scale));
  };
  std::transform(input, input + size, output, quant_func);
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
