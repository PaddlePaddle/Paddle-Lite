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

#ifdef CONCAT_OP

#include "operators/kernel/concat_kernel.h"
#include "fpga/KD/pes/concat_pe.hpp"

namespace paddle_mobile {
namespace operators {

template <>
bool ConcatKernel<FPGA, float>::Init(ConcatParam<FPGA>* param) {
  auto inputs = param->Inputs();
  auto out = param->Out();
  auto image_num = inputs.size();
  out->mutable_data<half>();

  zynqmp::ConcatPE& pe = param->context().pe<zynqmp::ConcatPE>();
  zynqmp::ConcatParam& concat_param = pe.param();
  std::vector<zynqmp::Tensor*> input_tensors;
  for (size_t i = 0; i < image_num; i++) {
    input_tensors.push_back(inputs[i]->zynqmpTensor());
  }
  concat_param.inputs = input_tensors;
  concat_param.output = out->zynqmpTensor();
  concat_param.axis = param->Axis();

  pe.init();
  pe.apply();
  return true;
}

template <>
void ConcatKernel<FPGA, float>::Compute(const ConcatParam<FPGA>& param) {
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  zynqmp::ConcatPE& pe = context.pe<zynqmp::ConcatPE>();
  pe.dispatch();

#ifdef PADDLE_MOBILE_DEBUG
  zynqmp::Debugger::get_instance().registerOutput("concat",
                                                  param.Out()->zynqmpTensor());
#endif
}
template class ConcatKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
