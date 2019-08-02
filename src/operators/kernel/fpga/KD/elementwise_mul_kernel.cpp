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

#ifdef ELEMENTWISEMUL_OP

#include "operators/kernel/elementwise_mul_kernel.h"
#include "fpga/KD/pes/scale_pe.hpp"
// #include "operators/kernel/central-arm-func/elementwise_mul_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ElementwiseMulKernel<FPGA, float>::Init(ElementwiseMulParam<FPGA>* param) {
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

  if (param->InputY()->numel() == 1) {
    Tensor* y =
        const_cast<Tensor*>(reinterpret_cast<const Tensor*>(param->InputY()));
    y->mutable_data<float>();
    float scale_value = param->InputY()->zynqmpTensor()->data<float>()[0];
    for (int i = 0; i < channel; ++i) {
      scale_data[i] = scale_value;
      bias_data[i] = 0.0f;
    }
  } else {
    scale->copyFrom(param->InputY()->zynqmpTensor());
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
void ElementwiseMulKernel<FPGA, float>::Compute(
    const ElementwiseMulParam<FPGA>& param) {
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  zynqmp::ScalePE& pe = context.pe<zynqmp::ScalePE>();
  pe.dispatch();
#ifdef PADDLE_MOBILE_DEBUG
  zynqmp::Debugger::get_instance().registerOutput("ew_mul",
                                                  param.Out()->zynqmpTensor());
#endif
}

template class ElementwiseMulKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
