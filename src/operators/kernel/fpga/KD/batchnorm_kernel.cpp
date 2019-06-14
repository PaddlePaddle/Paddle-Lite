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

#ifdef BATCHNORM_OP

#include "operators/kernel/batchnorm_kernel.h"
#include "fpga/KD/pes/batchnorm_pe.hpp"

namespace paddle_mobile {
namespace operators {

template <>
bool BatchNormKernel<FPGA, float>::Init(BatchNormParam<FPGA>* param) {
  param->OutputY()->mutable_data<half>();

  zynqmp::BatchnormPE& pe = param->context().pe<zynqmp::BatchnormPE>();
  zynqmp::BatchnormParam& bn_param = pe.param();

  bn_param.input = param->InputX()->zynqmpTensor();
  bn_param.output = param->OutputY()->zynqmpTensor();
  bn_param.bias = param->InputBias()->zynqmpTensor();
  bn_param.scale = param->InputScale()->zynqmpTensor();
  bn_param.mean = param->InputMean()->zynqmpTensor();
  bn_param.variance = param->InputVariance()->zynqmpTensor();
  bn_param.epsilon = param->Epsilon();
  bn_param.relu.enabled = false;  // TODO(chonwhite)

  pe.init();
  pe.apply();

  return true;
}

template <>
void BatchNormKernel<FPGA, float>::Compute(const BatchNormParam<FPGA>& param) {
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  zynqmp::BatchnormPE& pe = context.pe<zynqmp::BatchnormPE>();
  pe.dispatch();
  // std::cout << "bn\n";
  // param.OutputY()->zynqmpTensor()->printScale();
  // param.OutputY()->zynqmpTensor()->saveToFile("bn.txt");
}
template class BatchNormKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
