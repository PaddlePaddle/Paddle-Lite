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

#ifdef MUL_OP

#pragma once

namespace paddle_mobile {
namespace operators {

template <typename P>
void MulCompute(const MulParam<CPU> &param) {
  const Tensor *input_x = param.InputX();
  const Tensor *input_y = param.InputY();
  Tensor *out = param.Out();

  const Tensor x_matrix =
      input_x->dims().size() > 2
          ? framework::ReshapeToMatrix(*input_x, param.XNumColDims())
          : *input_x;
  const Tensor y_matrix =
      input_y->dims().size() > 2
          ? framework::ReshapeToMatrix(*input_y, param.YNumColDims())
          : *input_y;
  auto out_dim = out->dims();
  if (out_dim.size() != 2) {
    out->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
  }
  if (param.InputX()->type() == type_id<int8_t>().name()) {
    out->mutable_data<int32_t>();
    math::MatMul<int8_t, int32_t>(x_matrix, false, y_matrix, false,
                                  static_cast<float>(1), out,
                                  static_cast<float>(0));
  } else {
    out->mutable_data<float>();
    math::MatMul<float, float>(x_matrix, false, y_matrix, false,
                               static_cast<float>(1), out,
                               static_cast<float>(0));
  }
  if (out_dim.size() != 2) {
    out->Resize(out_dim);
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
