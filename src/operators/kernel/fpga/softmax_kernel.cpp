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
  auto input_ptr = input->data<float>();
  auto output_ptr = param->Out();
  Tensor *floatInput = new Tensor(*input);
  fpga::BypassArgs args;
  args.input_layout_type = fpga::LAYOUT_HWC;
  args.output_layout_type = fpga::LAYOUT_CHW;
  args.input_data_type = fpga::DATA_TYPE_FP16;
  args.output_data_type = fpga::DATA_TYPE_FP32;
  args.image.address = (void *)(input_ptr);
  args.image.height = (uint32_t)input->dims()[0];
  args.image.width = (uint32_t)input->dims()[1];
  args.image.channels = 1;
  args.output.address = (void *)floatInput->mutable_data<float>();

  param->SetFloatInput(floatInput);
  param->SetFpgaArgs(args);
  return true;
}

template <>
void SoftmaxKernel<FPGA, float>::Compute(
    const SoftmaxParam<FPGA> &param) const {
  DLOG << "======================================= FPGA SoftMAX "
          "===============================================";
  const Tensor *in_x = param.FloatInput();
  Tensor *out = param.Out();
  fpga::fpga_flush((void *)in_x->data<float>(), in_x->memory_size());
  fpga::PerformBypass(param.FpgaArgs());
  fpga::fpga_invalidate(out->data<float>(), out->memory_size());

  auto x_dims = in_x->dims();
  out->Resize(x_dims);
  math::SoftmaxFuntor<CPU, float>()(in_x, out);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
