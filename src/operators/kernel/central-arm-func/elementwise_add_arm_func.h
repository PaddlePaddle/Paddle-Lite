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

#ifdef ELEMENTWISEADD_OP

#pragma once

#include "operators/math/element_wise.h"
#include "operators/op_param.h"
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {

template <typename T>
inline void ElementwiseAddCompute(const ElementwiseAddParam<CPU> &param) {
  const framework::Tensor *input_x = param.InputX();
  const framework::Tensor *input_y = param.InputY();
  framework::Tensor *output = param.Out();
  int axis = param.Axis();

  math::AddElememtWise<IDENTITY>(input_x, input_y, axis, output);
}

template class ElementwiseAddKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
