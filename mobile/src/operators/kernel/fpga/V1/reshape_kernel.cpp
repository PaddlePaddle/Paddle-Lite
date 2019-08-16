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

#ifdef RESHAPE_OP

#include "operators/kernel/reshape_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ReshapeKernel<FPGA, float>::Init(ReshapeParam<FPGA> *param) {
  param->Out()->ShareDataWith(*param->InputX());
  const int in_n = param->InputX()->dims()[0];
  const int in_c = param->InputX()->dims()[1];
  const int in_h = param->InputX()->dims()[2];
  const int in_w = param->InputX()->dims()[3];
  auto out = param->Out();
  out->Resize(framework::make_ddim({in_n, in_c * in_h * in_w}));
  return true;
}

template <>
void ReshapeKernel<FPGA, float>::Compute(const ReshapeParam<FPGA> &param) {}

}  // namespace operators
}  // namespace paddle_mobile

#endif
