/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include "operators/kernel/transpose_kernel.h"

namespace paddle_mobile {
namespace operators {

template <typename T>
void TransposeFunc(const int numel, const T* input, const vector<int> axis,
                   const vector<int> old_strides, const vector<int> new_strides,
                   T* output) {
  for (int i = 0; i < numel; ++i) {
    int old_idx = 0;
    int idx = i;
    for (int j = 0; j < axis.size(); ++j) {
      int order = axis[j];
      old_idx += (idx / new_strides[j]) * old_strides[order];
      idx %= new_strides[j];
    }
    output[i] = input[old_idx];
  }
}

template <>
void TransposeKernel<CPU, float>::Compute(const TransposeParam& param) const {
  const auto* input_x = param.InputX();
  const auto input_x_dims = input_x->dims();
  auto* out = param.Out();
  const auto axis = param.Axis();
  const auto* input_x_data = input_x->data<float>();
  auto* out_data = out->mutable_data<float>();

  size_t axis_size = axis.size();
  std::vector<int> new_dims;
  new_dims.reserve(axis_size);
  for (auto c : axis) {
    new_dims.push_back(input_x_dims[c]);
  }

  std::vector<int> old_strides;
  std::vector<int> new_strides;
  for (int i = 0; i < axis.size(); i++) {
    int temp_old = 1;
    int temp_new = 1;
    for (int j = i + 1; j < axis.size(); j++) {
      temp_old *= input_x_dims[j];
      temp_new *= new_dims[j];
    }
    old_strides.push_back(temp_old);
    new_strides.push_back(temp_new);
  }

  TransposeFunc<float>(input_x->numel(), input_x_data, axis, old_strides,
                       new_strides, out_data);
}

}  // namespace operators
}  // namespace paddle_mobile
