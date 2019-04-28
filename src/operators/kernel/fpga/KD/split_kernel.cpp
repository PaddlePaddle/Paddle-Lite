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

#ifdef SPLIT_OP

#include "operators/kernel/split_kernel.h"
#include "operators/kernel/central-arm-func/split_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool SplitKernel<FPGA, float>::Init(SplitParam<FPGA> *param) {
  // param->Outs()->mutable_data<half>();
  std::vector<LoDTensor *> outs = param->Outs();
  for (int i = 0; i < outs.size(); i++) {
    outs[i]->mutable_data<half>();
  }
  return true;
}

template <>
void SplitKernel<FPGA, float>::Compute(const SplitParam<FPGA> &param) {
  // SplitCompute<float>(param);
}

template class SplitKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
