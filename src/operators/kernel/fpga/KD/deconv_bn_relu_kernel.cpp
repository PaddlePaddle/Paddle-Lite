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

// #ifdef FUSION_DECONVBNRELU_OP

#include "framework/operator.h"
#include "operators/op_param.h"
#include "operators/kernel/deconv_bn_relu_kernel.h"

#include "fpga/KD/float16.hpp"
#include "fpga/KD/pes/transposed_conv_pe.hpp"

using TransposedConvPE = paddle_mobile::zynqmp::TransposedConvPE;

namespace paddle_mobile {
namespace operators {

template <>
bool DeconvBNReluKernel<FPGA, float>::Init(FusionDeconvBNReluParam<FPGA>* param) {
  param->Output()->mutable_data<half>();

  TransposedConvPE& pe = param->context().pe<TransposedConvPE>();
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

  conv_param.activeParam.type = zynqmp::TYPE_RELU;
  conv_param.groups = param->Groups();
  conv_param.strides = param->Strides();
  conv_param.paddings = param->Paddings();
  conv_param.dilations = param->Dilations();

  combine_bn_params(bn_param, &conv_param);

  pe.init();
  pe.apply();
  return true;
}

template <>
void DeconvBNReluKernel<FPGA, float>::Compute(const FusionDeconvBNReluParam<FPGA>& param) {
  // param.Input()->zynqmpTensor()->saveToFile("deconvin", true);
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  TransposedConvPE& pe = context.pe<TransposedConvPE>();
  // pe.param().input->saveToFile("deconvin-convin", true);
  pe.dispatch();
  // param.Output()->zynqmpTensor()->saveToFile("deconvout", true);
  
#ifdef PADDLE_MOBILE_DEBUG
  zynqmp::Debugger::get_instance().registerOutput(
      "deconv", param.Output()->zynqmpTensor());
#endif
}

}  // namespace operators
}  // namespace paddle_mobile

// #endif
