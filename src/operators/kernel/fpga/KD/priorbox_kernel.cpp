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

#ifdef PRIORBOX_OP

#include "fpga/KD/pes/prior_box_pe.hpp"
#include "operators/kernel/prior_box_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool PriorBoxKernel<FPGA, float>::Init(PriorBoxParam<FPGA>* param) {
  param->OutputBoxes()->mutable_data<half>();
  param->OutputVariances()->mutable_data<half>();

  zynqmp::PriorBoxPE& pe = param->context().pe<zynqmp::PriorBoxPE>();
  zynqmp::PriorBoxParam& priobox_param = pe.param();
  priobox_param.input = param->Input()->zynqmpTensor();
  priobox_param.image = param->InputImage()->zynqmpTensor();
  priobox_param.outputBoxes = param->OutputBoxes()->zynqmpTensor();
  priobox_param.outputVariances = param->OutputVariances()->zynqmpTensor();
  priobox_param.minSizes = param->MinSizes();
  priobox_param.maxSizes = param->MaxSizes();
  priobox_param.aspectRatios = param->AspectRatios();
  priobox_param.variances = param->Variances();
  priobox_param.minMaxAspectRatiosOrder = param->MinMaxAspectRatiosOrder();
  priobox_param.flip = param->Flip();
  priobox_param.clip = param->Clip();
  priobox_param.stepW = param->StepW();
  priobox_param.stepH = param->StepH();
  priobox_param.offset = param->Offset();

  pe.init();
  pe.apply();

  return true;
}

template <>
void PriorBoxKernel<FPGA, float>::Compute(const PriorBoxParam<FPGA>& param) {
  std::cout << "PriorBoxKernel\n";
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  zynqmp::PriorBoxPE& pe = context.pe<zynqmp::PriorBoxPE>();
  pe.dispatch();

  std::cout << "boxes Out scale:"
            << param.OutputBoxes()->zynqmpTensor()->scale()[0] << std::endl;
  std::cout << "variances Out scale:"
            << param.OutputVariances()->zynqmpTensor()->scale()[0] << std::endl;
}

template class PriorBoxKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
