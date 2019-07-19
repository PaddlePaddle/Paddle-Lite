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
#ifdef EXP_OP
#include "exp_op.h"
namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
void EXPOp<DeviceType, T>::InferShape() const {
  auto shape = this->param_.InputX()->dims();
  this->param_.Out()->Resize(shape);
}
}  // namespace operators
}  // namespace paddle_mobile
namespace ops = paddle_mobile::operators;

#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(exp, ops::EXPOp);
#endif

#ifdef PADDLE_MOBILE_CL
// REGISTER_OPERATOR_CL(exp, ops::EXPOp);
#endif

#endif
