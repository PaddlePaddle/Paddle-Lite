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

#ifdef SCALE_OP

#include "operators/kernel/scale_kernel.h"
#include "fpga/KD/float16.hpp"
#include "fpga/KD/pes/scale_pe.hpp"

namespace paddle_mobile {
namespace operators {

template <>
bool ScaleKernel<FPGA, float>::Init(ScaleParam<FPGA>* param) {
  param->Out()->mutable_data<half>();
  return true;
}

template <>
void ScaleKernel<FPGA, float>::Compute(const ScaleParam<FPGA>& param) {
  zynqmp::Tensor* input = param.InputX()->zynqmpTensor();
  zynqmp::Tensor* output = param.Out()->zynqmpTensor();
  input->invalidate();

  for (int i = 0; i < input->shape().numel(); i++) {
    output->data<zynqmp::float16>()[i] = input->data<zynqmp::float16>()[i];
  }
  output->flush();
}

template class ScaleKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
