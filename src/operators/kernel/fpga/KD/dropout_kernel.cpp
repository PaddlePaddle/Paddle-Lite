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

#ifdef DROPOUT_OP

#include "operators/kernel/dropout_kernel.h"
#include "fpga/KD/pes/scale_pe.hpp"

namespace paddle_mobile {
namespace operators {

template <>
bool DropoutKernel<FPGA, float>::Init(DropoutParam<FPGA>* param) {
  param->Out()->mutable_data<half>();

  zynqmp::ScalePE& pe = param->context().pe<zynqmp::ScalePE>();
  zynqmp::ScaleParam& scale_param = pe.param();
  scale_param.input = param->InputX()->zynqmpTensor();
  scale_param.output = param->Out()->zynqmpTensor();

  int channel = scale_param.input->shape().channel();
  zynqmp::Tensor* scale = new zynqmp::Tensor();
  zynqmp::Tensor* bias = new zynqmp::Tensor();
  zynqmp::Shape shape(zynqmp::N, {channel});
  float* scale_data = scale->mutableData<float>(zynqmp::FP32, shape);
  float* bias_data = bias->mutableData<float>(zynqmp::FP32, shape);

  float scale_value = 1 - param->DropoutProb();
  for (int i = 0; i < channel; ++i) {
    scale_data[i] = scale_value;
    bias_data[i] = 0.0f;
  }
  scale->flush();
  bias->flush();

  scale_param.bias = bias;
  scale_param.scale = scale;

  pe.init();
  pe.apply();

  return true;
}

template <>
void DropoutKernel<FPGA, float>::Compute(const DropoutParam<FPGA>& param) {
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  zynqmp::ScalePE& pe = context.pe<zynqmp::ScalePE>();
  pe.dispatch();

  param.Out()->zynqmpTensor()->printScale();
  // param.InputX()->zynqmpTensor()->saveToFile("dropout_in.txt");
  // param.Out()->zynqmpTensor()->saveToFile("dropout_out.txt");
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
