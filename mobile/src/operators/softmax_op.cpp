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

#ifdef SOFTMAX_OP

#include "operators/softmax_op.h"

namespace paddle_mobile {
namespace operators {
template <typename DeviceType, typename T>
void SoftmaxOp<DeviceType, T>::InferShape() const {
  this->param_.Out()->Resize(this->param_.InputX()->dims());
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(softmax, ops::SoftmaxOp);
#endif
#if defined(PADDLE_MOBILE_FPGA) || defined(PADDLE_MOBILE_FPGA_KD)
REGISTER_OPERATOR_FPGA(softmax, ops::SoftmaxOp);
#endif
#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(softmax, ops::SoftmaxOp);
#endif

#endif
