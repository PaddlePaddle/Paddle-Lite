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

#ifdef FUSION_CONVBN_OP

#include "operators/kernel/conv_bn_kernel.h"
#include "fpga/KD/pes/conv_pe.hpp"

using ConvPE = paddle_mobile::zynqmp::ConvPE;

namespace paddle_mobile {
namespace operators {

template <>
bool ConvBNKernel<FPGA, float>::Init(FusionConvBNParam<FPGA>* param) {
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
  // conv_param.batchnorm = bn_param;
  conv_param.relu.enabled = false;
  conv_param.groups = param->Groups();
  conv_param.strides = param->Strides();
  conv_param.paddings = param->Paddings();

  combine_bn_params(bn_param, &conv_param);

  pe.init();
  pe.apply();
  return true;
}

template <>
void ConvBNKernel<FPGA, float>::Compute(const FusionConvBNParam<FPGA>& param) {
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  ConvPE& pe = context.pe<ConvPE>();
  pe.dispatch();
#ifdef PADDLE_MOBILE_DEBUG
  zynqmp::Debugger::get_instance().registerOutput(
      "conv_bn", param.Output()->zynqmpTensor());
#endif
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
