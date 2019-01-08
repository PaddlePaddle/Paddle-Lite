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

#ifdef SEQUENCE_SOFTMAX_OP

#include "operators/sequence_ops/sequence_softmax_op.h"

namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
void SequenceSoftmaxOp<DeviceType, T>::InferShape() const {
  const auto *input_x = this->param_.InputX();
  const auto &x_lod = input_x->lod();

  this->param_.Out()->Resize(input_x->dims());
  this->param_.Out()->set_lod(input_x->lod());
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(sequence_softmax, ops::SequenceSoftmaxOp);
#endif

#endif  // SEQUENCE_SOFTMAX_OP
