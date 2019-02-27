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

#ifdef PAD2D_OP

#include "operators/pad2d_op.h"
namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void Pad2dOp<Dtype, T>::InferShape() const {
  auto input_dims = this->param_.InputX()->dims();
  auto input_n = input_dims[0];
  auto input_c = input_dims[1];
  auto input_h = input_dims[2];
  auto input_w = input_dims[3];

  this->param_.Out()->Resize({input_n, input_c, input_h + 1, input_w + 1});
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(pad2d, ops::Pad2dOp);
#endif
#ifdef PADDLE_MOBILE_FPGA
REGISTER_OPERATOR_FPGA(pad2d, ops::Pad2dOp);
#endif

#endif
