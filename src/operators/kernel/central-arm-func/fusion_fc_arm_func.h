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

#ifdef FUSION_FC_OP

#pragma once

namespace paddle_mobile {
namespace operators {

template <typename P>
void FusionFcCompute(const FusionFcParam &param) {
  const Tensor *input_x = param.InputX();
  const Tensor *input_y = param.InputY();
  const Tensor *input_z = param.InputZ();
  auto *input_z_data = input_z->data<float>();
  int axis = param.Axis();
  Tensor *out = param.Out();
  auto *out_data = out->mutable_data<float>();
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
  PADDLE_MOBILE_ENFORCE(out_dim.size() == 2, " out_dim.size must be 2.");
  PADDLE_MOBILE_ENFORCE(input_z->dims().size() == 1, "inpu_z size must be 1");
  PADDLE_MOBILE_ENFORCE(out_dim[1] == input_z->dims()[0],
                        " out_dim.size must be 2.");
  axis = (axis == -1 ? out_dim.size() - input_z->dims().size() : axis);
  PADDLE_MOBILE_ENFORCE(axis == 1, " to fit broadcast, axis = 1. ")

  int64_t classes = input_z->numel();
  for (int i = 0; i < out_dim[0]; i++) {
    memory::Copy(out_data + i * classes, input_z_data, sizeof(float) * classes);
  }

  for (int i = 0; i < out->numel(); i++) {
    DLOG << out_data[i];
  }
  math::matmul<float>(x_matrix, false, y_matrix, false, static_cast<float>(1),
                      out, static_cast<float>(1));
  PADDLE_MOBILE_ENFORCE(out_dim.size() == 2, " out_dim.size must be 2.");
  //            if (out_dim.size() != 2) {
  //                out->Resize(out_dim);
  //            }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
