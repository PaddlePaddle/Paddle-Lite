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

#ifdef FUSION_CONVADDRELU_OP

#include "operators/kernel/conv_add_relu_kernel.h"
#include "fpga/KD/pes/conv_pe.hpp"
#include "fpga/KD/pes/conv_process.hpp"

namespace paddle_mobile {
namespace operators {

template <>
bool ConvAddReluKernel<FPGA, float>::Init(FusionConvAddReluParam<FPGA>* param) {
  param->Output()->mutable_data<half>();

  zynqmp::ConvPE& pe = param->context().pe<zynqmp::ConvPE>();
  zynqmp::ConvParam& conv_param = pe.param();
  zynqmp::BatchnormParam* bn_param = new zynqmp::BatchnormParam();

  conv_param.input = param->Input()->zynqmpTensor();
  conv_param.output = param->Output()->zynqmpTensor();
  conv_param.filter = param->Filter()->zynqmpTensor();
  // conv_param.batchnorm = bn_param;
  conv_param.relu.enabled = true;
  conv_param.groups = param->Groups();
  conv_param.strides = param->Strides();
  conv_param.paddings = param->Paddings();

  fill_scale_bias_const(&conv_param);

  Tensor* bias = param->Bias();
  conv_param.bias()->copyFrom(bias->zynqmpTensor());

  pe.init();
  pe.apply();
  return true;
}

template <>
void ConvAddReluKernel<FPGA, float>::Compute(
    const FusionConvAddReluParam<FPGA>& param) {
  // std::cout << "ConvAddReluKernel\n";
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  zynqmp::ConvPE& pe = context.pe<zynqmp::ConvPE>();
  pe.dispatch();

  std::cout << "Out scale:" << param.Output()->zynqmpTensor()->scale()[0]
            << std::endl;

  param.Output()->zynqmpTensor()->saveToFile("convaddrelu.txt");
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
