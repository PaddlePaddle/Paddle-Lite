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

  const int groups = param->Groups();
  const int channel = param->Input()->dims()[1];
  if (groups == channel) {
    zynqmp::DepthwiseConvPE& pe =
        param->context().pe<zynqmp::DepthwiseConvPE>();

    zynqmp::DepthwiseConvParam& depthwise_conv_param = pe.param();

    depthwise_conv_param.input = param->Input()->zynqmpTensor();
    depthwise_conv_param.output = param->Output()->zynqmpTensor();
    depthwise_conv_param.filter = param->Filter()->zynqmpTensor();
    depthwise_conv_param.relu.enabled = true;
    depthwise_conv_param.groups = param->Groups();
    depthwise_conv_param.strides = param->Strides();
    depthwise_conv_param.paddings = param->Paddings();

    fill_scale_bias_const(&depthwise_conv_param);
    Tensor* bias = param->Bias();
    bias->zynqmpTensor()->flush();
    bias->zynqmpTensor()->setDataLocation(zynqmp::CPU);
    depthwise_conv_param.bias()->copyFrom(bias->zynqmpTensor());
    pe.init();
    pe.apply();

  } else {
    zynqmp::ConvPE& pe = param->context().pe<zynqmp::ConvPE>();
    zynqmp::ConvParam& conv_param = pe.param();

    conv_param.input = param->Input()->zynqmpTensor();
    conv_param.output = param->Output()->zynqmpTensor();
    conv_param.filter = param->Filter()->zynqmpTensor();
    conv_param.relu.enabled = true;
    conv_param.groups = param->Groups();
    conv_param.strides = param->Strides();
    conv_param.paddings = param->Paddings();

    fill_scale_bias_const(&conv_param);
    Tensor* bias = param->Bias();
    bias->zynqmpTensor()->flush();
    bias->zynqmpTensor()->setDataLocation(zynqmp::CPU);
    conv_param.bias()->copyFrom(bias->zynqmpTensor());
    pe.init();
    pe.apply();
  }

  return true;
}

template <>
void ConvAddReluKernel<FPGA, float>::Compute(
    const FusionConvAddReluParam<FPGA>& param) {
  const int groups = param.Groups();
  const int channel = param.Output()->dims()[1];
  if (groups == channel) {
    zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
    zynqmp::DepthwiseConvPE& pe = context.pe<zynqmp::DepthwiseConvPE>();
    pe.dispatch();
  } else {
    zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
    zynqmp::ConvPE& pe = context.pe<zynqmp::ConvPE>();
    pe.dispatch();
  }

  param.Output()->zynqmpTensor()->printScale();
#ifdef PADDLE_MOBILE_DEBUG
  zynqmp::Debugger::get_instance().registerOutput(
      "conv_add_relu", param.Output()->zynqmpTensor());
#endif
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
