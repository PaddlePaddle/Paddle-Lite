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

#ifdef FLATTEN_OP

#ifndef RESHAPE_OP
#define RESHAPE_OP
#endif

#pragma once

#include <operators/kernel/reshape_kernel.h>
#include <vector>
#include "operators/flatten_op.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename P>
void FlattenCompute(const FlattenParam<CPU> &param) {
  const auto *input_x = param.InputX();
  const auto axis = param.Axis();
  const auto &input_x_dims = input_x->dims();
  auto *out = param.Out();

  const auto &out_shape_v = GetOutputShape(axis, input_x_dims);
  const framework::DDim &out_dim = ValidateShape(out_shape_v, input_x_dims);

  out->Resize(out_dim);
  out->mutable_data<float>();
  framework::TensorCopy(*input_x, out);
  out->Resize(out_dim);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
