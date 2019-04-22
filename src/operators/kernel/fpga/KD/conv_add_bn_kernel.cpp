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

#ifdef FUSION_CONVADDBN_OP

#include "operators/kernel/conv_add_bn_kernel.h"
#include <cmath>

namespace paddle_mobile {
namespace operators {

template <>
bool ConvAddBNKernel<FPGA, float>::Init(FusionConvAddBNParam<FPGA>* param) {
  // bool relu_enabled = false;
  zynqmp::PE<ConvParam>& conv = param.context().convPE();
  ConvParam& p = conv.param();
  p.input = param->Input()->ZynqTensor();
  p.filter = param->Filter()->ZynqTensor();

  BatchnormParam* bn = new BatchnormParam();
  p.bn = bn;

  return true;
}

template <>
void ConvAddBNKernel<FPGA, float>::Compute(
    const FusionConvAddBNParam<FPGA>& param) {
  zynqmp::PE<ConvParam>& conv = param.context().convPE();
  conv.dispatch();
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
