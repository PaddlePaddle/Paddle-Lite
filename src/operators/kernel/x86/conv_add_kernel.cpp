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

namespace paddle_mobile {
namespace operators {

template <>
bool ConvAddKernel<X86, float>::Init(FusionConvAddParam<X86> *param) {
  return true;
}

template <>
void ConvAddKernel<X86, float>::Compute(
    const FusionConvAddParam<X86> &param) const {
  // TODO
}

template class ConvAddKernel<X86, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
