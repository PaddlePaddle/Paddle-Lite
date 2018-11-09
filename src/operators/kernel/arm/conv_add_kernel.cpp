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
#ifdef FUSION_CONVADD_OP

#include "operators/kernel/conv_add_kernel.h"
#include "../central-arm-func/conv_add_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConvAddKernel<CPU, float>::Init(FusionConvAddParam<CPU> *param) {
  return true;
}

template <>
void ConvAddKernel<CPU, float>::Compute(const FusionConvAddParam<CPU> &param) {
  ConvAddCompute<float>(param);
}

template class ConvAddKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
