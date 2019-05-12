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
  // paddle_mobile::fpga::ActivationType activation_enable =
  //    paddle_mobile::fpga::LEAKYRELU;
  auto input = const_cast<LoDTensor *>(param->Input());
  auto bias = param->Bias();
  auto bias_ptr = bias->data<float>();
  auto filter = const_cast<LoDTensor *>(param->Filter());
  auto out = param->Output();
  const int groups = param->Groups();
  float Si = input->scale[0];
  float So = out->scale[0];
  float Sf = fpga::filter_find_max(filter);
  vector<int> paddings = param->Paddings();
  vector<int> strides = param->Strides();
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
      reinterpret_cast<float *>(fpga::fpga_malloc(2 * channel * sizeof(float)));
  auto new_scale = new Tensor();
  auto new_bias = new Tensor();
  auto new_scale_ptr = new_scale->mutable_data<float>({channel});
  auto new_bias_ptr = new_bias->mutable_data<float>({channel});

  for (int i = 0; i < channel; i++) {
    new_scale_ptr[i] = bn_scale_ptr[i] /
                       static_cast<float>(pow((bn_var_ptr[i] + epsilon), 0.5));
    new_bias_ptr[i] =
        bn_bias_ptr[i] + (bias_ptr[i] - bn_mean_ptr[i]) * new_scale_ptr[i];
    bs_ptr[i + channel] = new_scale_ptr[i] * Si / So * Sf / 127.0;
    bs_ptr[i] = new_bias_ptr[i] * 127.0 / So;
    if (groups == channel) {
      new_scale_ptr[i] = new_scale_ptr[i] * Si / So;
      new_bias_ptr[i] = new_bias_ptr[i] * 127.0f / So;
    }
  }

  if (groups == channel) {
    fpga::format_dwconv_data(filter, out, new_scale_ptr, &new_bias_ptr);
    fpga::DWconvArgs dwconv_arg = {0};
    fpga::fill_dwconv_arg(&dwconv_arg, input, out, filter, relu_enabled,
                          strides[0], strides[1], paddings[0], paddings[1],
                          new_bias_ptr);
    param->SetFpgaArgs(dwconv_arg);
    fpga::fpga_free(bs_ptr);
  } else {
    fpga::format_conv_data(filter, out, &bs_ptr, param->Groups());
    fpga::SplitConvArgs conv_arg = {0};
    fpga::fill_split_arg(&conv_arg, input, out, filter, relu_enabled,
                         param->Groups(), strides[0], strides[1], paddings[0],
                         paddings[1], bs_ptr);
    param->SetFpgaArgs(conv_arg);
  }
  delete new_scale;
  delete new_bias;
  return true;
}

template <>
void ConvAddBNReluKernel<FPGA, float>::Compute(
    const FusionConvAddBNReluParam<FPGA> &param) {
  if (param.Groups() == param.Output()->dims()[1]) {
    fpga::ComputeDWConv(param.FpgaDwconvArgs());
  } else {
    fpga::ComputeFpgaConv(param.FpgaArgs());
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
