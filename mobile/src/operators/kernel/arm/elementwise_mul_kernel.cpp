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

#include "operators/kernel/elementwise_mul_kernel.h"
#include "operators/kernel/central-arm-func/elementwise_mul_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ElementwiseMulKernel<CPU, float>::Init(ElementwiseMulParam<CPU> *param) {
  return true;
}

template <>
void ElementwiseMulKernel<CPU, float>::Compute(
    const ElementwiseMulParam<CPU> &param) {
  ElementwiseMulCompute<float>(param);
  param.Out()->set_lod(param.InputX()->lod());
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
