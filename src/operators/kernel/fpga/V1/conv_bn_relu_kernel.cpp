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

#ifdef FUSION_CONVBNRELU_OP

#include "operators/kernel/conv_bn_relu_kernel.h"
#include <cmath>
namespace paddle_mobile {
namespace operators {
template <>
bool ConvBNReluKernel<FPGA, float>::Init(FusionConvBNReluParam<FPGA> *param) {
  paddle_mobile::fpga::ActivationType activation_enable =
      paddle_mobile::fpga::LEAKYRELU;
  int16_t leaky_relu_negative_slope = 0;
  auto input = const_cast<Tensor *>(param->Input());
  auto filter = const_cast<Tensor *>(param->Filter());
  auto out = param->Output();
  auto bn_mean_ptr = param->InputMean()->data<float>();
  auto bn_var_ptr = param->InputVariance()->data<float>();
  auto bn_scale_ptr = param->InputScale()->data<float>();
  auto bn_bias_ptr = param->InputBias()->data<float>();
  const float epsilon = param->Epsilon();
  PADDLE_MOBILE_ENFORCE(out->dims()[1] == param->InputBias()->dims()[0],
                        "Output channel should be equal to bias number");
  const int channel = out->dims()[1];
  auto bs_ptr =
      (float *)fpga::fpga_malloc(2 * channel * sizeof(float));  // NOLINT
  auto new_scale = new Tensor();
  auto new_bias = new Tensor();
  auto new_scale_ptr = new_scale->mutable_data<float>({channel});
  auto new_bias_ptr = new_bias->mutable_data<float>({channel});
  for (int i = 0; i < channel; i++) {
    new_scale_ptr[i] = bn_scale_ptr[i] /
                       static_cast<float>(pow((bn_var_ptr[i] + epsilon), 0.5));
    new_bias_ptr[i] = bn_bias_ptr[i] + (0 - bn_mean_ptr[i]) * new_scale_ptr[i];
    bs_ptr[i + channel] = new_scale_ptr[i];
    bs_ptr[i] = new_bias_ptr[i];
  }
  const int groups = param->Groups();
  if (groups == channel) {
    fpga::format_dwconv_data(filter, out, new_scale_ptr, &new_bias_ptr);
    fpga::DWconvArgs dwconv_arg = {0};
    fpga::fill_dwconv_arg(&dwconv_arg, input, out, filter, activation_enable,
                          leaky_relu_negative_slope, param->Strides()[0],
                          param->Strides()[1], param->Paddings()[0],
                          param->Paddings()[1], new_bias_ptr);
    param->SetFpgaArgs(dwconv_arg);
  } else {
    fpga::format_conv_data(filter, out, &bs_ptr, param->Groups());
    fpga::SplitConvArgs conv_arg = {0};
    fpga::fill_split_arg(&conv_arg, input, out, filter, activation_enable,
                         leaky_relu_negative_slope, param->Groups(),
                         param->Strides()[0], param->Strides()[1],
                         param->Paddings()[0], param->Paddings()[1], bs_ptr);
    param->SetFpgaArgs(conv_arg);
  }
  delete new_scale;
  delete new_bias;
  return true;
}
template <>
void ConvBNReluKernel<FPGA, float>::Compute(
    const FusionConvBNReluParam<FPGA> &param) {
  if (param.Groups() == param.Output()->dims()[1]) {
    fpga::ComputeDWConv(param.FpgaDwconvArgs());
  } else {
    fpga::ComputeFpgaConv(param.FpgaArgs());
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
