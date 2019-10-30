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

#ifdef FUSION_CONVBNLEAKYRELU_OP

#include "operators/kernel/conv_bn_leakyrelu_kernel.h"
#include "fpga/KD/pes/conv_pe.hpp"
#include "fpga/KD/pes/conv_process.hpp"

#include <math.h>

using ConvPE = paddle_mobile::zynqmp::ConvPE;

namespace paddle_mobile {
namespace operators {

template <>
bool ConvBNLeakyReluKernel<FPGA, float>::Init(FusionConvBNLeakyReluParam<FPGA>* param) {
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
  conv_param.relu.enabled = true;
  conv_param.relu.leaky_relu_factor = param->Alpha();
  conv_param.groups = param->Groups();
  conv_param.strides = param->Strides();
  conv_param.paddings = param->Paddings();
  conv_param.dilations = param->Dilations();

  // conv_param.filter->saveToFile("conv_bn_leakyrelu_filter_", true);
  combine_bn_params(bn_param, &conv_param);
  pe.init();
  pe.apply();
  
  delete bn_param;

  return true;
}
template <>
void ConvBNLeakyReluKernel<FPGA, float>::Compute(
    const FusionConvBNLeakyReluParam<FPGA>& param) {
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  ConvPE& pe = context.pe<ConvPE>();
  pe.dispatch();

  param.Input()->zynqmpTensor()->printScale();
  param.Output()->zynqmpTensor()->printScale();
  // param.Input()->zynqmpTensor()->saveToFile("conv_bn_leakyrelu_input_", true);

  // param.Output()->zynqmpTensor()->saveToFile("conv_bn_leakyrelu_output_", true);
#ifdef PADDLE_MOBILE_DEBUG
  zynqmp::Debugger::get_instance().registerOutput(
      "conv_bn_leakyrelu", param.Output()->zynqmpTensor());
#endif
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
