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

#ifdef BOXCODER_OP

#include "operators/kernel/box_coder_kernel.h"
#include "operators/kernel/central-arm-func/box_coder_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool BoxCoderKernel<CPU, float>::Init(BoxCoderParam<CPU> *param) {
  return true;
}

template <>
void BoxCoderKernel<CPU, float>::Compute(const BoxCoderParam<CPU> &param) {
  BoxCoderCompute<float>(param);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
