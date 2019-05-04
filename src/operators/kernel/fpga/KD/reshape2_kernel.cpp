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

#ifdef RESHAPE2_OP

#include "operators/kernel/reshape2_kernel.h"
#include "operators/kernel/reshape_kernel.h"
// #include "fpga/KD/pes/reshape2_pe.hpp"

namespace paddle_mobile {
namespace operators {

template <>
bool Reshape2Kernel<FPGA, float>::Init(Reshape2Param<FPGA> *param) {
  param->Out()->mutable_data<half>();
  return true;
}

template <>
void Reshape2Kernel<FPGA, float>::Compute(const Reshape2Param<FPGA> &param) {
  const auto *input_x = param.InputX();
  const auto &input_x_dims = input_x->dims();
  auto *out = param.Out();

  framework::DDim out_dims = out->dims();
  const auto *input_shape = param.InputShape();
  if (input_shape) {
    auto *shape_data = input_shape->data<int>();
    framework::Tensor cpu_shape_tensor;
    auto shape =
        std::vector<int>(shape_data, shape_data + input_shape->numel());
    out_dims = ValidateShape(shape, input_x->dims());
  }

  bool inplace = param.Inplace();
  // out->Resize(out_dims);
  if (!inplace) {
    out->mutable_data<half>();
    framework::TensorCopy(*input_x, out);  // TODO(chonwhite) is it right?
    out->Resize(out_dims);
  } else {
    out->ShareDataWith(*input_x);
    out->Resize(out_dims);
  }
  out->zynqmpTensor()->scale()[0] = input_x->zynqmpTensor()->scale()[0];
  out->zynqmpTensor()->scale()[1] = input_x->zynqmpTensor()->scale()[1];
  std::cout << "Out scale:" << out->zynqmpTensor()->scale()[0] << std::endl;
}
template class Reshape2Kernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
