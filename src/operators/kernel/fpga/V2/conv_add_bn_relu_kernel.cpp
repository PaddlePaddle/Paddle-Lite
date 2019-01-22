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

#ifdef FUSION_CONVADDBNRELU_OP

#include "operators/kernel/conv_add_bn_relu_kernel.h"
#include <cmath>

namespace paddle_mobile {
namespace operators {

template <>
bool ConvAddBNReluKernel<FPGA, float>::Init(
    FusionConvAddBNReluParam<FPGA> *param) {
  bool relu_enabled = true;
  auto input = const_cast<Tensor *>(param->Input());
  const Tensor *bias = param->Bias();
  auto bias_ptr = bias->data<float>();
  auto filter = const_cast<Tensor *>(param->Filter());
  auto out = param->Output();
  auto bn_mean_ptr = param->InputMean()->data<float>();
  auto bn_var_ptr = param->InputVariance()->data<float>();
  auto bn_scale_ptr = param->InputScale()->data<float>();
  auto bn_bias_ptr = param->InputBias()->data<float>();
  const float epsilon = param->Epsilon();
  PADDLE_MOBILE_ENFORCE(out->dims()[1] == bias->dims()[0] &&
                            bias->dims()[0] == param->InputBias()->dims()[0],
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
    new_bias_ptr[i] =
        bn_bias_ptr[i] + (bias_ptr[i] - bn_mean_ptr[i]) * new_scale_ptr[i];
    bs_ptr[i + 2] = new_scale_ptr[i];
    bs_ptr[i] = new_bias_ptr[i];
  }
  param->SetNewScale(new_scale);
  param->SetNewBias(new_bias);

  fpga::format_conv_data(filter, out, &bs_ptr, param->Groups());

  fpga::SplitConvArgs conv_arg = {0};
  fpga::fill_split_arg(&conv_arg, input, out, filter, relu_enabled,
                       param->Groups(), param->Strides()[0],
                       param->Strides()[1], param->Paddings()[0],
                       param->Paddings()[1], bs_ptr);
  param->SetFpgaArgs(conv_arg);
  return true;
}

template <>
void ConvAddBNReluKernel<FPGA, float>::Compute(
    const FusionConvAddBNReluParam<FPGA> &param) {
  fpga::ComputeFpgaConv(param.FpgaArgs());
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
