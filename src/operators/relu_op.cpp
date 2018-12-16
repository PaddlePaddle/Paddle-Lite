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

#ifdef RELU_OP

#include "operators/relu_op.h"
namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void ReluOp<Dtype, T>::InferShape() const {
  auto input_dims = this->param_.InputX()->dims();
  this->param_.Out()->Resize(input_dims);
}

template <typename Dtype, typename T>
void Relu6Op<Dtype, T>::InferShape() const {
  auto input_dims = this->param_.InputX()->dims();
  this->param_.Out()->Resize(input_dims);
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(relu, ops::ReluOp);
REGISTER_OPERATOR_CPU(relu6, ops::Relu6Op);
#endif
#ifdef PADDLE_MOBILE_MALI_GPU
REGISTER_OPERATOR_MALI_GPU(relu, ops::ReluOp);
#endif
#ifdef PADDLE_MOBILE_FPGA
#endif
#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(relu, ops::ReluOp);
#endif

#endif
