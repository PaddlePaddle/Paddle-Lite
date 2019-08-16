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

#ifdef SOFTMAX_OP

#include "operators/kernel/softmax_kernel.h"
#include "fpga/KD/pes/softmax_pe.hpp"
#include "operators/kernel/central-arm-func/softmax_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool SoftmaxKernel<FPGA, float>::Init(SoftmaxParam<FPGA>* param) {
  param->Out()->mutable_data<half>();

  zynqmp::SoftmaxPE& pe = param->context().pe<zynqmp::SoftmaxPE>();
  zynqmp::SoftmaxParam& fc_param = pe.param();
  fc_param.input = param->InputX()->zynqmpTensor();
  fc_param.output = param->Out()->zynqmpTensor();
  pe.init();
  pe.apply();
  return true;
}

template <>
void SoftmaxKernel<FPGA, float>::Compute(const SoftmaxParam<FPGA>& param) {
  std::cout << "SoftmaxKernel\n";
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  zynqmp::SoftmaxPE& pe = context.pe<zynqmp::SoftmaxPE>();
  pe.dispatch();

  param.Out()->zynqmpTensor()->invalidate();
  std::string path =
      "softmax_" + std::to_string(param.Out()->zynqmpTensor()->id()) + ".txt";
  param.Out()->zynqmpTensor()->saveToFile(path);
  std::cout << "Out scale:" << param.Out()->zynqmpTensor()->scale()[0]
            << std::endl;
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
