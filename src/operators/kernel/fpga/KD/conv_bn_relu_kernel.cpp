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

#ifdef FUSION_CONVBNRELU_OP

#include "operators/kernel/conv_bn_relu_kernel.h"
#include "fpga/KD/pes/conv_pe.hpp"

#include <math.h>

using ConvPE = paddle_mobile::zynqmp::ConvPE;

namespace paddle_mobile {
namespace operators {

template <>
bool ConvBNReluKernel<FPGA, float>::Init(FusionConvBNReluParam<FPGA>* param) {
  param->Output()->mutable_data<half>();

  ConvPE& pe = param->context().pe<ConvPE>();
  zynqmp::ConvParam& conv_param = pe.param();
  zynqmp::BatchnormParam* bn_param = new zynqmp::BatchnormParam();
  bn_param->bias = param->InputBias()->zynqmpTensor();
  bn_param->scale = param->InputScale()->zynqmpTensor();
  bn_param->mean = param->InputMean()->zynqmpTensor();
  bn_param->variance = param->InputVariance()->zynqmpTensor();
  bn_param->epsilon = param->Epsilon();
  conv_param.input = param->Input()->zynqmpTensor();
  conv_param.output = param->Output()->zynqmpTensor();
  conv_param.filter = param->Filter()->zynqmpTensor();
  conv_param.batchnorm = bn_param;
  conv_param.relu.enabled = true;
  conv_param.groups = param->Groups();
  conv_param.strides = param->Strides();
  conv_param.paddings = param->Paddings();
  pe.init();
  pe.apply();
  return true;
}
template <>
void ConvBNReluKernel<FPGA, float>::Compute(
    const FusionConvBNReluParam<FPGA>& param) {
  std::cout << "ConvBNReluKernel\n";
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  ConvPE& pe = context.pe<ConvPE>();
  pe.dispatch();

  std::string path =
      "bnr_" + std::to_string(param.Output()->zynqmpTensor()->id()) + ".txt";
  // param.Output()->zynqmpTensor()->saveToFile(path);
  std::cout << "Out scale:" << param.Output()->zynqmpTensor()->scale()[0]
            << std::endl;

  if (isinf(param.Output()->zynqmpTensor()->scale()[0])) {
    // zynqmp::ConvParam& conv_param = pe.param();
    std::cout << "invalid cale !!!!!!!!!!!!" << std::endl;
    // std::cout << conv_param.convArgs.conv_arg[0].kernel.width << std::endl;
    exit(-1);
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
