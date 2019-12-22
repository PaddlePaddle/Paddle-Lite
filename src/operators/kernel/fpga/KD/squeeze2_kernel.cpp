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

#ifdef SQUEEZE2_OP

#include "operators/kernel/squeeze2_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool Squeeze2Kernel<FPGA, float>::Init(Squeeze2Param<FPGA> *param) {
  param->Out()->mutable_data<half>();
  return true;
}

template <>
void Squeeze2Kernel<FPGA, float>::Compute(const Squeeze2Param<FPGA> &param) {
  const auto *input_x = param.InputX();
  const auto &input_x_dims = input_x->dims();
  auto *out = param.Out();

  framework::DDim out_dims = out->dims();

  input_x->zynqmpTensor()->syncToCPU();
  out->zynqmpTensor()->copyFrom(input_x->zynqmpTensor());
  out->zynqmpTensor()->unalignImage();

  out->Resize(out_dims);
  
  out->zynqmpTensor()->setAligned(input_x->zynqmpTensor()->aligned());
  out->zynqmpTensor()->copyScaleFrom(input_x->zynqmpTensor());


#ifdef PADDLE_MOBILE_DEBUG
  zynqmp::Debugger::get_instance().registerOutput("squeeze2",
                                                  param.Out()->zynqmpTensor());
#endif
}
template class Squeeze2Kernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
