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
#ifdef TRANSPOSE_OP

#include "operators/kernel/transpose_kernel.h"
// #include "operators/kernel/central-arm-func/transpose_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <typename P>
void TransposeCompute(const TransposeParam<FPGA>& param) {
  const auto* input_x = param.InputX();
  const auto input_x_dims = input_x->dims();
  auto* out = param.Out();
  const auto axis = param.Axis();
  const auto* input_x_data = input_x->data<half>();
  auto* out_data = out->mutable_data<half>();
  zynqmp::Tensor ot;
  half* od = ot.mutableData<half>(zynqmp::FP16, out->zynqmpTensor()->shape());

  int num = input_x_dims[1];
  int channel = input_x_dims[2];

  // DLOG << "num::" << num << "  channel::" << channel;

  int index = 0;
  for (int n = 0; n < num; n++) {
    for (int c = 0; c < channel; c++) {
      out_data[c * num + n] = input_x_data[n * channel + c];
      index++;
    }
  }
}

template <>
bool TransposeKernel<FPGA, float>::Init(TransposeParam<FPGA>* param) {
  auto input = param->InputX();
  auto output = param->Out();
  auto axis = param->Axis();
  auto dim = input->dims();
  // output->ShareDataWith(*input);

  auto dim_v = vectorize(dim);

  for (int i = 0; i < axis.size(); i++) {
    dim_v[i] = dim[axis[i]];
  }
  output->Resize(framework::make_ddim(dim_v));
  output->mutable_data<half>();

  if (param->InputX()->dims().size() == 4) {
    param->Out()->ShareDataWith(*param->InputX());
  }
  return true;
}

template <>
void TransposeKernel<FPGA, float>::Compute(const TransposeParam<FPGA>& param) {
  auto input = param.InputX();
  auto output = param.Out();
  input->zynqmpTensor()->unalignImage();
  if (param.InputX()->dims().size() != 4) {
    TransposeCompute<float>(param);
    auto out = param.Out();
    auto out_data = out->data<half>();
  } else {
    output->zynqmpTensor()->copyFrom(input->zynqmpTensor());
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
