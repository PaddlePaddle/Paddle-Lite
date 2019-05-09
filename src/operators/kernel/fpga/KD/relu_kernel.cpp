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

#ifdef RELU_OP

#include "fpga/KD/pes/relu_pe.hpp"
#include "operators/kernel/activation_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ReluKernel<FPGA, float>::Init(ReluParam<FPGA>* param) {
  param->Out()->mutable_data<half>();
  zynqmp::ReluPE& pe = param->context().pe<zynqmp::ReluPE>();
  zynqmp::InputParam& input_param = pe.param();
  input_param.input = param->InputX()->zynqmpTensor();
  input_param.output = param->Out()->zynqmpTensor();

  return true;
}

template <>
void ReluKernel<FPGA, float>::Compute(const ReluParam<FPGA>& param) {
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  zynqmp::ReluPE& pe = context.pe<zynqmp::ReluPE>();
  pe.dispatch();
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
