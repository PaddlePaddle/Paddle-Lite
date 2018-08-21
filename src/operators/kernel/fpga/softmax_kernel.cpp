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

#ifdef SOFTMAX_OP

#include "../softmax_kernel.h"
#include "../central-arm-func/softmax_arm_func.h"
#include "common/types.h"
#include "fpga/api.h"
#include "operators/math/softmax.h"
namespace paddle_mobile {
namespace operators {

template <>
bool SoftmaxKernel<FPGA, float>::Init(SoftmaxParam<FPGA> *param) {
  const Tensor *input = param->InputX();
  if (input->type() == typeid(half)) {
    auto input_ptr = input->data<half>();
    auto output_ptr = param->Out();
    fpga::BypassArgs args;
    args.convert_type = fpga::DATA_FP16_TO_FP32;
    args.layout_type = fpga::LAYOUT_HWC_TO_CHW;
    args.image.address = (void *)(input_ptr);
    args.image.height = input->dims()[0];
    args.image.width = input->dims()[1];
    args.image.channels = 1;
    args.output.address = output_ptr;
    param->SetFpgaArgs(args);
  }

  return true;
}

template <>
void SoftmaxKernel<FPGA, float>::Compute(
    const SoftmaxParam<FPGA> &param) const {
  // SoftmaxCompute<float>(param);
}

template class SoftmaxKernel<FPGA, float>;
}  // namespace operators
}  // namespace paddle_mobile

#endif
