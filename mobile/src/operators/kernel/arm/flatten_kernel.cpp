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

#ifdef FLATTEN_OP

#include "operators/kernel/flatten_kernel.h"
#include "operators/kernel/central-arm-func/flatten_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool FlattenKernel<CPU, float>::Init(FlattenParam<CPU> *param) {
  return true;
}

template <>
void FlattenKernel<CPU, float>::Compute(const FlattenParam<CPU> &param) {
  FlattenCompute<float>(param);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
