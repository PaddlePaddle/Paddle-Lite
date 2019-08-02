
/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef FLATTEN2_OP

#include "operators/kernel/flatten2_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool Flatten2Kernel<GPU_CL, float>::Init(
    paddle_mobile::operators::FlattenParam<paddle_mobile::GPU_CL> *param) {
  this->cl_helper_.AddKernel("flatten2", "flatten2_kernel.cl");
  return true;
}

template <>
void Flatten2Kernel<GPU_CL, float>::Compute(
    const paddle_mobile::operators::FlattenParam<paddle_mobile::GPU_CL>
        &param) {}

}  // namespace operators
}  // namespace paddle_mobile

#endif
