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

#ifdef INCREMENT_OP

#pragma once

#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename P>
void IncrementCompute(const IncrementParam<CPU> &param) {
  const framework::Tensor *input = param.InputX();
  framework::Tensor *out = param.Out();
  float step = param.Step();

  out->mutable_data<int64_t>();
  const int64_t *input_data = input->data<int64_t>();
  int64_t *out_data = out->data<int64_t>();
  *out_data = *input_data + step;
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
