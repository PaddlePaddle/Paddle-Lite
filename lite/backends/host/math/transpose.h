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
#include <vector>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace host {
namespace math {

template <typename T>
void Transpose(const Tensor &input,
               Tensor *output,
               const std::vector<int> &orders) {
  auto in_dims = input.dims();
  auto out_dims = output->dims();
  int num_axes = in_dims.size();
  int count = in_dims.production();

  const T *din = input.data<T>();
  T *dout = output->mutable_data<T>();
  std::vector<int> old_steps(
      {static_cast<int>(in_dims[1] * in_dims[2] * in_dims[3]),
       static_cast<int>(in_dims[2] * in_dims[3]),
       static_cast<int>(in_dims[3]),
       1});
  std::vector<int> new_steps(
      {static_cast<int>(out_dims[1] * out_dims[2] * out_dims[3]),
       static_cast<int>(out_dims[2] * out_dims[3]),
       static_cast<int>(out_dims[3]),
       1});

  for (int i = 0; i < count; ++i) {
    int old_idx = 0;
    int idx = i;
    for (int j = 0; j < num_axes; ++j) {
      int order = orders[j];
      old_idx += (idx / new_steps[j]) * old_steps[order];
      idx %= new_steps[j];
    }
    dout[i] = din[old_idx];
  }
}

}  // namespace math
}  // namespace host
}  // namespace lite
}  // namespace paddle
