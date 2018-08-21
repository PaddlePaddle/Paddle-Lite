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

#ifdef FUSION_CONVADDRELU_OP

#include "operators/kernel/conv_add_relu_kernel.h"
#include "fpga/quantization.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConvAddReluKernel<FPGA, float>::Init(FusionConvAddReluParam<FPGA> *param) {
  bool relu_enabled = true;
  const Tensor *input = param->Input();
  auto input_ptr = input->data<half>();
  const Tensor *bias = param->Bias();
  auto bias_ptr = bias->data<float>();
  Tensor *filter = param->Filter();
  Tensor *out = param->Output();
  auto out_ptr = out->mutable_data<half>();

  PADDLE_MOBILE_ENFORCE(out->dims()[1] == bias->dims()[0],
                        "Output channel should be equal to bias number");
  int channel = out->dims()[1];
  float *bs_ptr = (float *)fpga::fpga_malloc(2 * channel * sizeof(float));
  for (int i = 0; i < channel; i++) {
    bs_ptr[i * 2] = 1;
    bs_ptr[i * 2 + 1] = bias_ptr[i];
  }

  fpga::quantize_filter(filter);
  auto filter_ptr = filter->data<int8_t>();

  fpga::ConvArgs convArgs;
  convArgs.relu_enabled = relu_enabled;
  convArgs.filter_address = (void *)filter_ptr;
  convArgs.filter_num = filter->dims()[0];
  convArgs.group_num = param->Groups();
  convArgs.sb_address = (void *)bs_ptr;
  convArgs.kernel.stride_h = param->Strides()[0];
  convArgs.kernel.stride_w = param->Strides()[1];
  convArgs.kernel.height = filter->dims()[2];
  convArgs.kernel.width = filter->dims()[3];
  convArgs.image.address = (void *)input_ptr;
  convArgs.image.channels = input->dims()[1];
  convArgs.image.height = input->dims()[2];
  convArgs.image.width = input->dims()[3];

  convArgs.image.pad_height = param->Paddings()[0];
  convArgs.image.pad_width = param->Paddings()[1];
  convArgs.image.scale_address = input->fpga_args().scale_pointer();
  convArgs.output.address = (void *)out_ptr;
  convArgs.output.scale_address = out->fpga_args().scale_pointer();
  param->SetFpgaArgs(convArgs);
  return true;
}

template <>
void ConvAddReluKernel<FPGA, float>::Compute(
    const FusionConvAddReluParam<FPGA> &param) const {
  fpga::ComputeFpgaConv(param.FpgaArgs());
}
template class ConvAddReluKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
