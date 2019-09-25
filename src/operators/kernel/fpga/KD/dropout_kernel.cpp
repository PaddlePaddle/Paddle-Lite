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

void cpu_compute(const DropoutParam<FPGA>& param) {
    zynqmp::Tensor* input = param.InputX()->zynqmpTensor();
    zynqmp::Tensor* output = param.Out()->zynqmpTensor();
    zynqmp::Tensor float_input;
    float* input_data = float_input.mutableData<float>(zynqmp::FP32, input->shape());
    input->syncToCPU();
    float_input.copyFrom(input);

    zynqmp::float16* data_out = param.Out()->zynqmpTensor()->data<zynqmp::float16>();
    float max = 0;
    float scale_value = 1 - param.DropoutProb();
    for (int i = 0; i < input->shape().numel(); ++i) {
      float value = input_data[i] * scale_value;
      data_out[i] = zynqmp::float_to_half(value);

      if (value < 0) {
        value = -value;
      }
      if (value > max) {
        max = value;
      }
    }

    output->scale()[0] = max / 127.0f;
    output->scale()[1] = 127.0f / max;
    output->flush();
}

template <>
void DropoutKernel<FPGA, float>::Compute(const DropoutParam<FPGA>& param) {
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  if (param.InputX()->numel() >= 2048) {
    cpu_compute(param);
  } else {
    zynqmp::ScalePE& pe = context.pe<zynqmp::ScalePE>();
    pe.dispatch();
  }

#ifdef PADDLE_MOBILE_DEBUG
  zynqmp::Debugger::get_instance().registerOutput("dropout",
                                                  param.Out()->zynqmpTensor());
#endif

}

}  // namespace operators
}  // namespace paddle_mobile

#endif
