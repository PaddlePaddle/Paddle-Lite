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

#ifdef FUSION_CONVADDBN_OP

#include "operators/kernel/conv_add_bn_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConvAddBNKernel<FPGA, float>::Init(FusionConvAddBNParam<FPGA> *param) {
  bool relu_enabled = false;
  auto *input = const_cast<Tensor *>(param->Input());
  auto input_ptr = input->data<float>();
  const Tensor *bias = param->Bias();
  auto bias_ptr = bias->data<float>();
  auto *filter = const_cast<Tensor *>(param->Filter());

  Tensor *out = param->Output();

  auto bn_mean_ptr = param->InputMean()->data<float>();
  auto bn_var_ptr = param->InputVariance()->data<float>();
  auto bn_scale_ptr = param->InputScale()->data<float>();
  auto bn_bias_ptr = param->InputBias()->data<float>();
  const float epsilon = param->Epsilon();
  PADDLE_MOBILE_ENFORCE(out->dims()[1] == bias->dims()[0] &&
                            bias->dims()[0] == param->InputBias()->dims()[0],
                        "Output channel should be equal to bias number");

  const int channel = out->dims()[1];
  auto *bs_ptr =
      reinterpret_cast<float *>(fpga::fpga_malloc(2 * channel * sizeof(float)));
  auto *new_scale = new Tensor();
  auto *new_bias = new Tensor();
  auto new_scale_ptr = new_scale->mutable_data<float>({channel});
  auto new_bias_ptr = new_bias->mutable_data<float>({channel});

  for (int i = 0; i < channel; i++) {
    new_scale_ptr[i] = bn_scale_ptr[i] /
                       static_cast<float>(pow((bn_var_ptr[i] + epsilon), 0.5));
    new_bias_ptr[i] =
        bn_bias_ptr[i] + (bias_ptr[i] - bn_mean_ptr[i]) * new_scale_ptr[i];
    bs_ptr[i + channel] = new_scale_ptr[i];
    bs_ptr[i] = new_bias_ptr[i];
  }
  param->SetNewScale(new_scale);
  param->SetNewBias(new_bias);

  float max_value = fpga::filter_find_max(filter);
  fpga::format_filter(filter, max_value, param->Groups());
  auto filter_ptr = filter->data<float>();

  int element_num_per_div =
      fpga::get_element_num_per_div(filter, param->Groups());
  fpga::format_bias_scale_array(&bs_ptr, element_num_per_div, channel);

  fpga::format_ofm(out);
  auto out_ptr = out->mutable_data<float>();

  fpga::WrapperConvArgs convArgs;
  convArgs.group_num = (uint32_t)param->Groups();
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
    convArgs.args[i].group_num = (uint32_t)param->Groups();
    convArgs.args[i].kernel.stride_h = (uint32_t)param->Strides()[0];
    convArgs.args[i].kernel.stride_w = (uint32_t)param->Strides()[1];
    convArgs.args[i].kernel.height = (uint32_t)filter->dims()[2];
    convArgs.args[i].kernel.width = (uint32_t)filter->dims()[3];
    convArgs.args[i].image.address = input_ptr;
    convArgs.args[i].image.channels = (uint32_t)input->dims()[1];
    convArgs.args[i].image.height = (uint32_t)input->dims()[2];
    convArgs.args[i].image.width = (uint32_t)input->dims()[3];
    convArgs.args[i].image.pad_height = (uint32_t)param->Paddings()[0];
    convArgs.args[i].image.pad_width = (uint32_t)param->Paddings()[1];
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
void ConvAddBNKernel<FPGA, float>::Compute(
    const FusionConvAddBNParam<FPGA> &param) const {
  fpga::ComputeFpgaConv(param.FpgaArgs());
}
template class ConvAddBNKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
