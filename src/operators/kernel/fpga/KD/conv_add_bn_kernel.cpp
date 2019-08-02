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

#ifdef FUSION_CONVADDBN_OP

#include "operators/kernel/conv_add_bn_kernel.h"
#include <cmath>

namespace paddle_mobile {
namespace operators {

using ConvPE = paddle_mobile::zynqmp::ConvPE;
using DepthwiseConvPE = paddle_mobile::zynqmp::DepthwiseConvPE;

template <>
bool ConvAddBNKernel<FPGA, float>::Init(FusionConvAddBNParam<FPGA>* param) {
  param->Output()->mutable_data<half>();
  const int groups = param->Groups();
  const int channel = param->Input()->dims()[1];
  if (groups == channel) {
    DepthwiseConvPE& pe = param->context().pe<DepthwiseConvPE>();

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
    depthwise_conv_param.relu.enabled = false;
    depthwise_conv_param.groups = param->Groups();
    depthwise_conv_param.strides = param->Strides();
    depthwise_conv_param.paddings = param->Paddings();

    combine_add_bn_params(bn_param, param->Bias()->zynqmpTensor(),
                          &depthwise_conv_param);

    pe.init();
    pe.apply();
    delete bn_param;

  } else {
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
    conv_param.relu.enabled = false;
    conv_param.groups = param->Groups();
    conv_param.strides = param->Strides();
    conv_param.paddings = param->Paddings();

    combine_add_bn_params(bn_param, param->Bias()->zynqmpTensor(), &conv_param);

    pe.init();
    pe.apply();
    delete bn_param;
  }

  return true;
}

template <>
void ConvAddBNKernel<FPGA, float>::Compute(
    const FusionConvAddBNParam<FPGA>& param) {
  const int groups = param.Groups();
  const int channel = param.Output()->dims()[1];
  if (groups == channel) {
    zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
    DepthwiseConvPE& pe = context.pe<DepthwiseConvPE>();
    pe.dispatch();
  } else {
    zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
    ConvPE& pe = context.pe<ConvPE>();
    pe.dispatch();
  }
#ifdef PADDLE_MOBILE_DEBUG
  zynqmp::Debugger::get_instance().registerOutput(
      "conv_add_bn", param.Output()->zynqmpTensor());
#endif
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
