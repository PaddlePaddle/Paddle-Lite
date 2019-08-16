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
#include "operators/kernel/fetch_kernel.h"
#include "fpga/KD/pes/output_pe.hpp"

namespace paddle_mobile {
namespace operators {

using OutputPE = zynqmp::OutputPE;

template <>
bool FetchKernel<FPGA, float>::Init(FetchParam<FPGA>* param) {
  auto input = param->InputX();
  int col = param->Col();
  auto output = &(param->Out()->at(col));
  output->Resize(input->dims());
  output->mutable_data<float>();

  zynqmp::Context& context = const_cast<zynqmp::Context&>(param->context_);
  OutputPE& pe = context.pe<OutputPE>();
  zynqmp::OutputParam& out_param = pe.param();
  out_param.input = input->zynqmpTensor();
  out_param.output = output->zynqmpTensor();

  pe.init();
  pe.apply();
  return true;
}

template <>
void FetchKernel<FPGA, float>::Compute(const FetchParam<FPGA>& param) {
  std::cout << "FetchKernel\n";
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  OutputPE& pe = context.pe<OutputPE>();
  pe.dispatch();

  int col = param.Col();
  auto output = &(param.Out()->at(col));
  output->zynqmpTensor()->saveToFile("fetch_out.txt");
}
template class FetchKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile
