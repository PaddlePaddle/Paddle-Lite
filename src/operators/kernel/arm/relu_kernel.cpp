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

#include "operators/kernel/relu_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
void ReluKernel<CPU, float>::Compute(const ReluParam &param) const {
  const auto *input_x = param.InputX();
  auto *input_x_ptr = input_x->data<float>();
  auto *out = param.Out();
  auto *out_ptr = out->mutable_data<float>();
  for (int i = 0; i < input_x->numel(); i++) {
    out_ptr[i] = input_x_ptr[i] > 0 ? input_x_ptr[i] : 0;
  }
}
}  // namespace operators
}  // namespace paddle_mobile
