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

#ifdef FUSION_DWCONVBNRELU_OP

#include "operators/kernel/dwconv_bn_relu_kernel.h"
#include "fpga/KD/pes/depthwise_conv_pe.hpp"

namespace paddle_mobile {
namespace operators {

template <>
bool DWConvBNReluKernel<FPGA, float>::Init(
    FusionDWConvBNReluParam<FPGA>* param) {
  param->Output()->mutable_data<half>();

  zynqmp::DepthwiseConvPE& pe = param->context().pe<zynqmp::DepthwiseConvPE>();

  zynqmp::DepthwiseConvParam& depthwise_conv_param = pe.param();

  zynqmp::BatchnormParam* bn_param = new zynqmp::BatchnormParam();
  bn_param->bias = param->InputBias()->zynqmpTensor();
  bn_param->scale = param->InputScale()->zynqmpTensor();
  bn_param->mean = param->InputMean()->zynqmpTensor();
  bn_param->variance = param->InputVariance()->zynqmpTensor();
  bn_param->epsilon = param->Epsilon();

  depthwise_conv_param.input = param->Input()->zynqmpTensor();
  depthwise_conv_param.output = param->Output()->zynqmpTensor();
  depthwise_conv_param.filter = param->Filter()->zynqmpTensor();
  depthwise_conv_param.relu.enabled = true;
  depthwise_conv_param.groups = param->Groups();
  depthwise_conv_param.strides = param->Strides();
  depthwise_conv_param.paddings = param->Paddings();

  combine_bn_params(bn_param, &depthwise_conv_param);
  pe.init();
  pe.apply();
  delete bn_param;

  return true;
}

template <>
void DWConvBNReluKernel<FPGA, float>::Compute(
    const FusionDWConvBNReluParam<FPGA>& param) {
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  zynqmp::DepthwiseConvPE& pe = context.pe<zynqmp::DepthwiseConvPE>();
  pe.dispatch();

  param.Output()->zynqmpTensor()->printScale();
#ifdef PADDLE_MOBILE_DEBUG
  zynqmp::Debugger::get_instance().registerOutput(
      "dwconv_bn_relu", param.Output()->zynqmpTensor());
#endif
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
