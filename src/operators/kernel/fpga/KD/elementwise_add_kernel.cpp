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
#ifdef FUSION_ELEMENTWISEADDRELU_OP

#include "operators/kernel/elementwise_add_kernel.h"
#include "fpga/KD/pes/elementwise_add_pe.hpp"

using ElementwiseAddPE = paddle_mobile::zynqmp::ElementwiseAddPE;

namespace paddle_mobile {
namespace operators {

template <>
bool ElementwiseAddKernel<FPGA, float>::Init(ElementwiseAddParam<FPGA>* param) {
  param->Out()->mutable_data<half>();

  ElementwiseAddPE& pe = param->context().pe<ElementwiseAddPE>();
  zynqmp::ElementwiseAddParam& ew_param = pe.param();
  ew_param.inputs = {
      param->InputX()->zynqmpTensor(),
      param->InputY()->zynqmpTensor(),
  };
  ew_param.output = param->Out()->zynqmpTensor();
  ew_param.relu.enabled = false;

  pe.init();
  pe.apply();
  return true;
}

template <>
void ElementwiseAddKernel<FPGA, float>::Compute(
    const ElementwiseAddParam<FPGA>& param) {
  std::cout << "ElementwiseAddKernel\n";
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  ElementwiseAddPE& pe = context.pe<ElementwiseAddPE>();
  pe.dispatch();

  std::string path =
      "ew_" + std::to_string(param.Out()->zynqmpTensor()->id()) + ".txt";
  // param.Out()->zynqmpTensor()->saveToFile(path);
  std::cout << "Out scale:" << param.Out()->zynqmpTensor()->scale()[0]
            << std::endl;
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
