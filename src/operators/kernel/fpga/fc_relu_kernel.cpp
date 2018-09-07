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
  bool relu_enabled = true;
  auto *input_x = const_cast<Tensor *>(param->InputX());
  auto input_x_ptr = input_x->data<float>();
  auto *filter = const_cast<Tensor *>(param->InputY());
  const Tensor *input_z = param->InputZ();
  auto input_z_ptr = input_z->data<float>();
  Tensor *out = param->Out();
  PADDLE_MOBILE_ENFORCE(input_x->dims()[1] == filter->dims()[0],
                        "Image channel should be equal to weight number");
  int channel = (uint32_t)out->dims()[1];
  auto *bs_ptr = (float *)fpga::fpga_malloc(2 * channel * sizeof(float));
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

  filter->Resize(framework::make_ddim({num, filter_channel, height, width}));
  float max_value = fpga::filter_find_max(filter);
  fpga::format_filter(filter, max_value, 1);
  auto filter_ptr = filter->data<float>();

  int element_num_per_div = fpga::get_element_num_per_div(filter, 1);
  fpga::format_bias_scale_array(&bs_ptr, element_num_per_div, channel);

  auto out_ptr = out->mutable_data<float>();

  fpga::WrapperConvArgs convArgs;
  convArgs.group_num = 1;
  convArgs.split_num = (uint32_t)fpga::get_plit_num(filter);
  convArgs.filter_num = (uint32_t)filter->dims()[0];
  convArgs.output.address = out_ptr;
  convArgs.output.scale_address = out->scale;
  convArgs.args = (fpga::ConvArgs *)fpga::fpga_malloc(convArgs.split_num *
                                                      sizeof(fpga::ConvArgs));
  param->SetFpgaArgs(convArgs);

  int element_num = fpga::get_aligned_filter_element_num(
      filter->dims()[1] * filter->dims()[2] * filter->dims()[3]);

  int n = convArgs.split_num;
  for (int i = 0; i < n; i++) {
    convArgs.args[i].relu_enabled = relu_enabled;
    convArgs.args[i].group_num = 1;
    convArgs.args[i].kernel.stride_h = 1;
    convArgs.args[i].kernel.stride_w = 1;
    convArgs.args[i].kernel.height = (uint32_t)filter->dims()[2];
    convArgs.args[i].kernel.width = (uint32_t)filter->dims()[3];
    convArgs.args[i].image.address = input_x_ptr;
    convArgs.args[i].image.channels = (uint32_t)input_x->dims()[1];
    convArgs.args[i].image.height = (uint32_t)input_x->dims()[2];
    convArgs.args[i].image.width = (uint32_t)input_x->dims()[3];
    convArgs.args[i].image.pad_height = 0;
    convArgs.args[i].image.pad_width = 0;
    convArgs.args[i].filter_address = &((int8_t *)filter_ptr)[i * element_num];
    convArgs.args[i].sb_address = &((int8_t *)bs_ptr)[i * element_num];
    convArgs.args[i].filter_num =
        (uint32_t)(i == n - 1 ? fpga::get_aligned_filter_num(
                                    channel - (n - 1) * element_num_per_div)
                              : element_num_per_div);
    convArgs.args[i].image.scale_address =
        (float *)fpga::fpga_malloc(2 * sizeof(float));
  }
  return true;
}
template <>
void FusionFcReluKernel<FPGA, float>::Compute(
    const FusionFcReluParam<FPGA> &param) const {
  fpga::ComputeFpgaConv(param.FpgaArgs());
};

}  // namespace operators
}  // namespace paddle_mobile
#endif
