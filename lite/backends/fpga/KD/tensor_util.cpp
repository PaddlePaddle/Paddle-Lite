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

#include <algorithm>

#include "lite/backends/fpga/KD/tensor_util.hpp"

namespace paddle {
namespace zynqmp {
float find_max(const Tensor& tensor) {
  float max = 0;
  Tensor& t = const_cast<Tensor&>(tensor);
  float* data = t.data<float>();
  for (int i = 0; i < t.shape().numel(); i++) {
    float value = data[i] > 0 ? data[i] : -data[i];
    max = std::max(value, max);
  }
  return max;
}
}  // namespace zynqmp
}  // namespace paddle
