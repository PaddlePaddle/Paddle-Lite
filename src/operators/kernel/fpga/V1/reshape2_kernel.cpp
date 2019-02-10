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
#include "framework/ddim.h"

namespace paddle_mobile {
namespace operators {

template <>
bool Reshape2Kernel<FPGA, float>::Init(Reshape2Param<FPGA> *param) {
  auto input = const_cast<LoDTensor *>(param->InputX());
  auto output = param->Out();
  auto shape = param->Shape();
  output->ShareDataWith(*input);

  auto num_in = framework::product(input->dims());
  auto num_shape = framework::product(framework::make_ddim(shape));
  PADDLE_MOBILE_ENFORCE(num_shape != 0, "0 index is not supported");

  for (int i = 0; i < shape.size(); i++) {
    if (shape[i] == -1) {
      shape[i] = static_cast<int>(-num_in / num_shape);
      break;
    }
  }
  output->Resize(framework::make_ddim(shape));
  DLOG << "input: " << input;
  DLOG << "output: " << output;

  return true;
}

template <>
void Reshape2Kernel<FPGA, float>::Compute(const Reshape2Param<FPGA> &param) {
  auto input = const_cast<LoDTensor *>(param.InputX());
  auto output = param.Out();
  auto shape = param.Shape();

  if (output->type() != typeid(half)) {
    DLOG << "wrong type";
  }

  auto num_in = framework::product(input->dims());
  auto num_shape = framework::product(framework::make_ddim(shape));
  PADDLE_MOBILE_ENFORCE(num_shape != 0, "0 index is not supported");

  for (int i = 0; i < shape.size(); i++) {
    if (shape[i] == -1) {
      shape[i] = static_cast<int>(-num_in / num_shape);
      break;
    }
  }
  output->Resize(framework::make_ddim(shape));
  if (output->type() != typeid(half)) {
    DLOG << "wrong type";
    DLOG << output;
  }
  //
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
