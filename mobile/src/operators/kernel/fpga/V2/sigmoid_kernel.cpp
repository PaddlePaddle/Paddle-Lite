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

#ifdef SIGMOID_OP

#include "operators/kernel/activation_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool SigmoidKernel<FPGA, float>::Init(SigmoidParam<FPGA> *param) {
  auto input = const_cast<LoDTensor *>(param->InputX());
  auto input_ptr = input->data<int8_t>();
  paddle_mobile::fpga::ActivationType activation_enable =
      paddle_mobile::fpga::SIGMOID;
  int16_t leaky_relu_negative_slope =
      fpga::fp32_2_fp16(input->scale[0] / 127.0);
  auto out = param->Out();
  fpga::format_ofm(out);

  fpga::BypassArgs args = {fpga::DATA_TYPE_INT8};
  args.input_data_type = fpga::DATA_TYPE_INT8;
  args.output_data_type = fpga::DATA_TYPE_INT8;
  args.image.address = input_ptr;
  args.image.height = 1;
  args.image.width = 1;
  args.image.channels = input->fpga_data_num;
  args.output.address = out->data<int8_t>();
  args.output.scale_address = out->scale;
  args.output.activation.activation_type = activation_enable;
  args.output.activation.leaky_relu_negative_slope = leaky_relu_negative_slope;
  param->SetFpgaArgs(args);
  return true;
}

template <>
void SigmoidKernel<FPGA, float>::Compute(const SigmoidParam<FPGA> &param) {
  fpga::PerformBypass(param.FpgaArgs());
  param.Out()->scale[0] = 1.0;
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
