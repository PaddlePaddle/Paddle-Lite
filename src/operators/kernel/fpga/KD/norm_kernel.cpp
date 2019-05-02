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

#ifdef NORM_OP

#include "operators/kernel/norm_kernel.h"
#include "fpga/KD/pes/norm_pe.hpp"

namespace paddle_mobile {
namespace operators {

template <>
bool NormKernel<FPGA, float>::Init(NormParam<FPGA>* param) {
  param->Out()->mutable_data<half>();

  zynqmp::NormPE& pe = param->context().pe<zynqmp::NormPE>();
  zynqmp::NormParam& norm_param = pe.param();
  norm_param.input = param->InputX()->zynqmpTensor();
  norm_param.output = param->Out()->zynqmpTensor();

  pe.init();
  pe.apply();

  return true;
}

template <>
void NormKernel<FPGA, float>::Compute(const NormParam<FPGA>& param) {
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  zynqmp::NormPE& pe = context.pe<zynqmp::NormPE>();
  pe.dispatch();

  // param.Out()->zynqmpTensor()->saveToFile();
  param.Out()->zynqmpTensor()->printScale();
}
template class NormKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
