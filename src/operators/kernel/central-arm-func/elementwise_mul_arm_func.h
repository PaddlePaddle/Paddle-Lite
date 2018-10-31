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

#ifdef ELEMENTWISEMUL_OP

#pragma once
#include "operators/math/elementwise_op_function.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename T>
struct MulFunctor {
  inline T operator()(T a, T b) const { return a * b; }
};

template <typename P>
void ElementwiseMulCompute(const ElementwiseMulParam<CPU> &param) {
  const Tensor *input_x = param.InputX();
  const Tensor *input_y = param.InputY();
  Tensor *Out = param.Out();
  Out->mutable_data<float>();
  int axis = param.Axis();
  ElementwiseComputeEx<MulFunctor<float>, float>(input_x, input_y, axis,
                                                 MulFunctor<float>(), Out);
}

template class ElementwiseMulKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
