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
#include "fpga/api/fpga_api.h"
#include "fpga/fpga_quantilization.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConvAddBNKernel<FPGA, float>::Init(FusionConvAddBNParam *param) {
  bool relu_enabled = false;
  const Tensor *input = param->Input();
  auto input_ptr = input->data<half>();
  const Tensor *bias = param->Bias();
  auto bias_ptr = bias->data<float>();
  Tensor *filter = param->Filter();

  Tensor *out = param->Output();
  auto out_ptr = out->mutable_data<half>();
  auto bn_mean_ptr = param->InputMean()->data<float>();
  auto bn_var_ptr = param->InputVariance()->data<float>();
  auto bn_scale_ptr = param->InputScale()->data<float>();
  auto bn_bias_ptr = param->InputBias()->data<float>();
  const float epsilon = param->Epsilon();
  PADDLE_MOBILE_ENFORCE(input->dims()[1] == bias->dims()[0] &&
                            bias->dims()[0] == param->InputBias()->dims()[0],
                        "Image channel should be equal to bias number");

  const int channel = input->dims()[1];
  float *bs_ptr =
      reinterpret_cast<float *>(fpga::fpga_malloc(2 * channel * sizeof(float)));
  Tensor *new_scale = new Tensor();
  Tensor *new_bias = new Tensor();
  auto new_scale_ptr = new_scale->mutable_data<float>({channel});
  auto new_bias_ptr = new_bias->mutable_data<float>({channel});

  for (int i = 0; i < channel; i++) {
    new_scale_ptr[i] = bn_scale_ptr[i] /
                       static_cast<float>(pow((bn_var_ptr[i] + epsilon), 0.5));
    new_bias_ptr[i] =
        bn_bias_ptr[i] + (bias_ptr[i] - bn_mean_ptr[i]) * new_scale_ptr[i];
    bs_ptr[i * 2] = new_scale_ptr[i];
    bs_ptr[i * 2 + 1] = new_bias_ptr[i];
  }
  param->SetNewScale(new_scale);
  param->SetNewBias(new_bias);

  fpga::quantify_filter(filter);

  auto filter_ptr = filter->data<float>();
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
void ConvAddBNKernel<FPGA, float>::Compute(
    const FusionConvAddBNParam &param) const {
  fpga::ComputeFpgaConv(param.FpgaArgs());
}
template class ConvAddBNKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
