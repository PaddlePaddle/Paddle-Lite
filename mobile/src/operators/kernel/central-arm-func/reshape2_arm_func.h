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

#ifdef RESHAPE2_OP
#pragma once

#include <vector>
#include "operators/kernel/reshape_kernel.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename P>
void Reshape2Compute(const Reshape2Param<CPU> &param) {
  const auto *input_x = param.InputX();
  const auto &input_x_dims = input_x->dims();
  auto *out = param.Out();
  framework::DDim out_dims = out->dims();
  const auto *input_shape = param.InputShape();

  if (input_shape) {
    auto *shape_data = input_shape->data<int>();
    framework::Tensor cpu_shape_tensor;
    auto shape =
        std::vector<int>(shape_data, shape_data + input_shape->numel());
    out_dims = ValidateShape(shape, input_x->dims());
  } else {
    auto &shape = param.Shape();
    out_dims = ValidateShape(shape, input_x_dims);
  }

  bool inplace = param.Inplace();
  out->Resize(out_dims);
  if (!inplace) {
    out->mutable_data<float>();
    framework::TensorCopy(*input_x, out);
    out->Resize(out_dims);
  } else {
    out->ShareDataWith(*input_x);
    out->Resize(out_dims);
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
