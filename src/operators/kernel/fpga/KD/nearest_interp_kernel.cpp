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

#ifdef NEAREST_INTERP_OP

#include "operators/kernel/nearest_interp_kernel.h"
#include "fpga/KD/pes/resize.hpp"

namespace paddle_mobile {
namespace operators {

template <>
bool NearestInterpKernel<FPGA, float>::Init(NearestInterpParam<FPGA>* param) {
  param->Out()->mutable_data<half>();

  zynqmp::ResizePE& pe = param->context().pe<zynqmp::ResizePE>();
  zynqmp::ResizeParam& norm_param = pe.param();
  norm_param.input = param->InputX()->zynqmpTensor();
  norm_param.output = param->Out()->zynqmpTensor();

  pe.init();
  pe.apply();

  return true;
}

template <>
void NearestInterpKernel<FPGA, float>::Compute(
    const NearestInterpParam<FPGA>& param) {
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  zynqmp::ResizePE& pe = context.pe<zynqmp::ResizePE>();
  pe.dispatch();
  // param.InputX()->zynqmpTensor()->saveToFile("resize_in.txt");
  // param.Out()->zynqmpTensor()->saveToFile("resize.txt");
  param.Out()->zynqmpTensor()->printScale();
  // float* scale = param.Out()->zynqmpTensor()->scale();
  // std::cout << "scale:" << scale[0] << " inv:" << scale[1] << std::endl;
}

template class NearestInterpKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
