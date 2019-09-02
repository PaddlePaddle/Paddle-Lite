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
#include "fpga/KD/float16.hpp"
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
  // float* scale_data = scale->mutableData<float>(zynqmp::FP32, shape);
  // float* bias_data = bias->mutableData<float>(zynqmp::FP32, shape);
  zynqmp::float16* scale_data = scale->mutableData<zynqmp::float16>(zynqmp::FP16, shape);
  zynqmp::float16* bias_data = bias->mutableData<zynqmp::float16>(zynqmp::FP16, shape);

  if (param->InputY()->numel() == 1) {
    // float* scale_data = scale->mutableData<float>(zynqmp::FP32, shape);
    // float* bias_data = bias->mutableData<float>(zynqmp::FP32, shape);
    Tensor* y =
        const_cast<Tensor*>(reinterpret_cast<const Tensor*>(param->InputY()));
    y->mutable_data<float>();
    float scale_value = param->InputY()->zynqmpTensor()->data<float>()[0];
    zynqmp::float16 one = zynqmp::float_to_half(0);
    for (int i = 0; i < channel; ++i) {
      scale_data[i] = zynqmp::float_to_half(scale_value);
      bias_data[i] = one;
    }
    scale_param.bias = bias;
    scale_param.scale = scale;
  } else {
    // numel 不为1的时候，Y来之上一个节点，可能是是float16
    // scale->copyFrom(param->InputY()->zynqmpTensor());
    zynqmp::float16 one = zynqmp::float_to_half(0);
    for (int i = 0; i < channel; ++i) {
      bias_data[i] = one;
    }

    scale_param.bias = bias;
    scale_param.scale = param->InputY()->zynqmpTensor();
  }
  scale->flush();
  bias->flush();

  pe.init();
  pe.apply();

  return true;
}

void cpu_compute(const ElementwiseMulParam<FPGA>& param) {
    zynqmp::Tensor* input = param.InputX()->zynqmpTensor();
    zynqmp::Tensor* input_y = param.InputY()->zynqmpTensor();
    zynqmp::Tensor* output = param.Out()->zynqmpTensor();
    zynqmp::Tensor float_input;
    float* input_data = float_input.mutableData<float>(zynqmp::FP32, input->shape());
    input->syncToCPU();
    float_input.copyFrom(input);

    zynqmp::Tensor float_input_y;
    float* input_y_data = float_input_y.mutableData<float>(zynqmp::FP32, input_y->shape());
    input_y->syncToCPU();
    float_input_y.copyFrom(input_y);

    // zynqmp::Tensor float_out;
    // float* data_out = float_out.mutableData<float>(zynqmp::FP32, output->shape());
    zynqmp::float16* data_out = param.Out()->zynqmpTensor()->data<zynqmp::float16>();

    int wh = input->shape().width() * input->shape().height();
    float max = 0;

    for (int i = 0; i < wh; i++) {
      for (int j = 0; j < input_y->shape().numel(); j++) {
        int index = i * input->shape().channel() + j;
        float value = input_data[index] * input_y_data[j];
        data_out[index] = zynqmp::float_to_half(value);

        if (value < 0) {
          value = -value;
        }
        if (value > max) {
          max = value;
        }
      }
    }

    // output->copyFrom(&float_out);
    output->flush();
    output->scale()[0] = max / 127.0f;
    output->scale()[1] = 127.0f / max;
}


template <>
void ElementwiseMulKernel<FPGA, float>::Compute(
    const ElementwiseMulParam<FPGA>& param) {
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  zynqmp::ScalePE& pe = context.pe<zynqmp::ScalePE>();

  if (param.InputY()->numel() >= 2048) {
    cpu_compute(param);
  } else {
    pe.dispatch();
  }

  // param.InputX()->zynqmpTensor()->saveToFile("in.txt");
  // param.InputY()->zynqmpTensor()->saveToFile("in_y.txt");
  // param.Out()->zynqmpTensor()->saveToFile("out.txt");
 
#ifdef PADDLE_MOBILE_DEBUG
  zynqmp::Debugger::get_instance().registerOutput("ew_mul",
                                                  param.Out()->zynqmpTensor());
#endif

}

template class ElementwiseMulKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
