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

#ifdef SIGMOID_OP

#include "fpga/KD/float16.hpp"
#include "operators/kernel/activation_kernel.h"

using float16 = paddle_mobile::zynqmp::float16;

namespace paddle_mobile {
namespace operators {

template <>
bool SigmoidKernel<FPGA, float>::Init(SigmoidParam<FPGA> *param) {
  param->Out()->mutable_data<float>();
  param->Out()->zynqmpTensor()->setAligned(false);
  param->Out()->zynqmpTensor()->setDataLocation(zynqmp::CPU);
  return true;
}
template <>
void SigmoidKernel<FPGA, float>::Compute(const SigmoidParam<FPGA> &param) {
  const auto *input_x = param.InputX();
  auto out = param.Out();
  int numel = input_x->numel();

  float16 *in_data = input_x->zynqmpTensor()->data<float16>();
  float *out_data = out->zynqmpTensor()->data<float>();
  input_x->zynqmpTensor()->syncToCPU();
  // input_x->zynqmpTensor()->saveToFile("sin.txt");
  float max = 0.0f;
  for (int i = 0; i < numel; i++) {
    /* code */
    float value = zynqmp::half_to_float(in_data[i]);
    value = 1 / (1 + exp(-value));
    out_data[i] = value;
    // out_data[i] = zynqmp::float_to_half(value);
    max = std::max(std::abs(value), max);
  }
  out->zynqmpTensor()->scale()[0] = max / 127.0;
  out->zynqmpTensor()->scale()[1] = 127.0 / max;
  out->zynqmpTensor()->flush();
  out->zynqmpTensor()->printScale();

#ifdef PADDLE_MOBILE_DEBUG
  zynqmp::Debugger::get_instance().registerOutput("sigmoid",
                                                  param.Out()->zynqmpTensor());
#endif

  exit(-1);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
