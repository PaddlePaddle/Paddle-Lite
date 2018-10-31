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

#ifdef DROPOUT_OP

#include "operators/kernel/dropout_kernel.h"
#include <operators/math/transform.h>

namespace paddle_mobile {
namespace operators {

template <>
bool DropoutKernel<CPU, float>::Init(DropoutParam<CPU> *para) {
  return true;
}

template <typename T>
struct DropoutFunctor {
  explicit DropoutFunctor(T drop_pro) : dropout_pro_(drop_pro) {}
  inline T operator()(T in) const { return (1 - dropout_pro_) * in; }

 private:
  T dropout_pro_;
};

template <>
void DropoutKernel<CPU, float>::Compute(const DropoutParam<CPU> &param) {
  const auto *input_x = param.InputX();
  auto *input_x_ptr = input_x->data<float>();
  auto *out = param.Out();
  auto *out_ptr = out->mutable_data<float>();
  const float dropoutProb = param.DropoutProb();
  DropoutFunctor<float> func_(dropoutProb);
  math::Transform trans;
  trans(input_x_ptr, input_x_ptr + input_x->numel(), out_ptr, func_);
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
