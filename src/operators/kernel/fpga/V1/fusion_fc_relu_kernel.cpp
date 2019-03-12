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
#ifdef FUSION_FCRELU_OP

#include "operators/kernel/fc_relu_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool FusionFcReluKernel<FPGA, float>::Init(FusionFcReluParam<FPGA> *param) {
  // bool relu_enabled = false;
  paddle_mobile::fpga::ActivationType activation_enable =
      paddle_mobile::fpga::LEAKYRELU;
  int16_t leaky_relu_negative_slope = 0;
  auto input_x = const_cast<LoDTensor *>(param->InputX());
  auto filter = const_cast<Tensor *>(param->InputY());
  const Tensor *input_z = param->InputZ();
  auto input_z_ptr = input_z->data<float>();
  auto out = param->Out();

  // PADDLE_MOBILE_ENFORCE(input_x->dims()[1] == filter->dims()[0],
  //                      "Image channel should be equal to weight number");
  int channel = (uint32_t)out->dims()[1];
  auto bs_ptr =
      (float *)fpga::fpga_malloc(2 * channel * sizeof(float));  // NOLINT
  for (int i = 0; i < channel; i++) {
    bs_ptr[i + channel] = 1;
    bs_ptr[i] = input_z_ptr[i];
  }
  int num = (uint32_t)filter->dims()[1];
  int chw = (uint32_t)filter->dims()[0];
  PADDLE_MOBILE_ENFORCE(
      chw == input_x->numel(),
      "Filter element num should be equal to IFM element num");
  int height = (uint32_t)input_x->dims()[2];
  int width = (uint32_t)input_x->dims()[3];
  int filter_channel = chw / height / width;

  out->Resize(framework::make_ddim({1, channel, 1, 1}));
  filter->Resize(framework::make_ddim({num, filter_channel, height, width}));
  float max_value = fpga::filter_find_max(filter);
  fpga::format_fc_filter(filter, max_value);

  int element_num_per_div = fpga::get_filter_num_per_div(filter, 1);
  fpga::format_bias_scale_array(&bs_ptr, element_num_per_div, channel);
  fpga::format_fp16_ofm(out);

  fpga::SplitConvArgs conv_arg = {0};
  fpga::fill_split_arg(&conv_arg, input_x, out, filter, activation_enable,
                       leaky_relu_negative_slope, 1, 1, 1, 0, 0, bs_ptr);
  param->SetFpgaArgs(conv_arg);
  return true;
}

template <>
void FusionFcReluKernel<FPGA, float>::Compute(
    const FusionFcReluParam<FPGA> &param) {
  fpga::ComputeFpgaConv(param.FpgaArgs());
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
