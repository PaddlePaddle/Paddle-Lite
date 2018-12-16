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
#ifdef TRANSPOSE2_OP

#include "operators/kernel/transpose2_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool Transpose2Kernel<FPGA, float>::Init(Transpose2Param<FPGA> *param) {
  param->Out()->ShareDataWith(*param->InputX());
  return true;
}

template <>
void Transpose2Kernel<FPGA, float>::Compute(
    const Transpose2Param<FPGA> &param) {
  // Transpose2Compute<float>(param);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
