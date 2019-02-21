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

#ifdef ELEMENTWISESUB_OP

#pragma once

#include "framework/data_type.h"
#include "operators/math/elementwise_op_function.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename T>
struct SubFunctor {
  inline T operator()(T a, T b) const { return a - b; }
};

struct SubOpFunctor {
  const framework::Tensor* x_;
  const framework::Tensor* y_;
  const int axis_;
  framework::Tensor* out_;

  SubOpFunctor(const framework::Tensor* x, const framework::Tensor* y,
               framework::Tensor* out, const int axis)
      : x_(x), y_(y), out_(out), axis_(axis) {}

  template <typename T>
  void apply() const {
    out_->mutable_data<T>();
    ElementwiseComputeEx<SubFunctor<T>, T>(x_, y_, axis_, SubFunctor<T>(),
                                           out_);
  }
};

template <typename P>
void ElementwiseSubCompute(const ElementwiseSubParam<CPU>& param) {
  const Tensor* input_x = param.InputX();
  const Tensor* input_y = param.InputY();
  Tensor* out = param.Out();

  int axis = param.Axis();
  framework::VisitDataType(framework::ToDataType(input_x->type()),
                           SubOpFunctor(input_x, input_y, out, axis));
}

template class ElementwiseSubKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
