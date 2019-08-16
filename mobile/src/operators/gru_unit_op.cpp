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

#ifdef GRU_UNIT_OP

#include "operators/gru_unit_op.h"

namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
void GruUnitOp<DeviceType, T>::InferShape() const {
  auto input_dims = this->param_.InputInput()->dims();
  auto hidden_prev_dims = this->param_.InputHiddenPrev()->dims();
  auto weight_dims = this->param_.InputWeight()->dims();
  int batch_size = input_dims[0];
  int input_size = input_dims[1];
  int frame_size = hidden_prev_dims[1];
  int weight_height = weight_dims[0];
  int weight_width = weight_dims[1];
  PADDLE_MOBILE_ENFORCE(
      (input_size == frame_size * 3),
      "The input_size must be 3 times of frame_size in GRUUnitOp.");
  PADDLE_MOBILE_ENFORCE(
      (weight_height == frame_size),
      "The shape of Weight matrix must be [frame_size, frame_size * 3].");
  PADDLE_MOBILE_ENFORCE(
      (weight_width == frame_size * 3),
      "The shape of Weight matrix must be [frame_size, frame_size * 3].");
  if (this->param_.InputBias()) {
    auto bias_dims = this->param_.InputBias()->dims();
    int bias_height = bias_dims[0];
    int bias_width = bias_dims[1];
    PADDLE_MOBILE_ENFORCE((bias_height == 1),
                          "The shape of Bias must be [1, frame_size * 3].");
    PADDLE_MOBILE_ENFORCE((bias_width == frame_size * 3),
                          "The shape of Bias must be [1, frame_size * 3].");
  }
  this->param_.OutGate()->Resize({batch_size, frame_size * 3});
  this->param_.OutResetHiddenPrev()->Resize({batch_size, frame_size});
  this->param_.OutHidden()->Resize({batch_size, frame_size});
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(gru_unit, ops::GruUnitOp);
#endif

#ifdef PADDLE_MOBILE_FPGA
#endif

#ifdef PADDLE_MOBILE_CL
#endif

#endif
