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
bool DropoutKernel<X86, float>::Init(DropoutParam<X86> *para) {
  return true;
}

template <>
void DropoutKernel<X86, float>::Compute(const DropoutParam<X86> &param) const {
  // TODO
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
