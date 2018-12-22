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

#ifdef GRU_OP

#include "operators/gru_op.h"
#include <vector>
#include "common/enforce.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void GruOp<Dtype, T>::InferShape() const {
  auto input_dims = this->param_.InputInput()->dims();
  auto weight_dims = this->param_.InputWeight()->dims();
  int input_size = input_dims[1];
  int frame_size = weight_dims[0];
  PADDLE_MOBILE_ENFORCE(
      (input_size == frame_size * 3),
      "The input_size must be 3 times of frame_size in GRUOp.");
  PADDLE_MOBILE_ENFORCE(
      (weight_dims[1] == frame_size * 3),
      "The shape of Weight matrix must be [frame_size, frame_size * 3].");
  if (this->param_.InputH0()) {
    auto h0_dims = this->param_.InputH0()->dims();
    PADDLE_MOBILE_ENFORCE((h0_dims[1] == frame_size),
                          "The width of H0 must be equal to frame_size.");
  }
  if (this->param_.InputBias()) {
    auto bias_dims = this->param_.InputBias()->dims();
    int bias_height = bias_dims[0];
    int bias_width = bias_dims[1];
    PADDLE_MOBILE_ENFORCE((bias_height == 1),
                          "The shape of Bias must be [1, frame_size * 3].");
    PADDLE_MOBILE_ENFORCE((bias_width == frame_size * 3),
                          "The shape of Bias must be [1, frame_size * 3].");
  }
  this->param_.OutBatchGate()->Resize(input_dims);
  this->param_.OutBatchResetHiddenPrev()->Resize({input_dims[0], frame_size});
  this->param_.OutBatchHidden()->Resize({input_dims[0], frame_size});
  this->param_.OutHidden()->Resize({input_dims[0], frame_size});
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(gru, ops::GruOp);
#endif
#ifdef PADDLE_MOBILE_FPGA
#endif

#endif
