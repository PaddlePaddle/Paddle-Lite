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

#include "operators/kernel/feed_kernel.h"
#include "fpga/KD/pes/input_pe.hpp"

using InputParam = paddle_mobile::zynqmp::InputParam;
using InputPE = paddle_mobile::zynqmp::InputPE;

namespace paddle_mobile {
namespace operators {

template <>
bool FeedKernel<FPGA, float>::Init(FeedParam<FPGA>* param) {
  int col = param->Col();
  auto input = const_cast<LoDTensor*>(&param->InputX()->at(col));

  InputPE& pe = param->context().pe<InputPE>();
  InputParam& input_param = pe.param();
  input->mutable_data<float>();
  zynqmp::Tensor* input_tensor = input->zynqmpTensor();
  input_param.input = input_tensor;
  param->Out()->mutable_data<half>();
  auto out = param->Out()->zynqmpTensor();
  input_param.output = out;
  pe.init();

  return true;
}

template <>
void FeedKernel<FPGA, float>::Compute(const FeedParam<FPGA>& param) {
  std::cout << "FeedKernel\n";
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  InputPE& pe = context.pe<InputPE>();

  int col = param.Col();
  auto input = const_cast<LoDTensor*>(&param.InputX()->at(col));
  InputParam& input_param = pe.param();
  input->mutable_data<float>();
  zynqmp::Tensor* input_tensor = input->zynqmpTensor();
  input_param.input = input_tensor;
  param.Out()->Resize(input->dims());
  param.Out()->mutable_data<half>();
  auto out = param.Out()->zynqmpTensor();
  input_param.output = out;
  pe.dispatch();

  param.Out()->zynqmpTensor()->saveToFile("feed_out.txt");
}
template class FeedKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile
