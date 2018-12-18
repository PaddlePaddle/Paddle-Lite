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

#ifdef FUSION_DECONVADD_OP

#include "operators/kernel/deconv_add_kernel.h"
#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <>
bool DeconvAddKernel<FPGA, float>::Init(FusionDeconvAddParam<FPGA> *param) {
  bool relu_enabled = false;
  auto input = const_cast<Tensor *>(param->Input());
  const Tensor *bias = param->Bias();
  auto bias_ptr = bias->data<float>();
  auto filter = const_cast<Tensor *>(param->Filter());
  auto out = param->Output();

  PADDLE_MOBILE_ENFORCE(out->dims()[1] == bias->dims()[0],
                        "Output channel should be equal to bias number");
  int channel = out->dims()[1];

  int sub_conv_n = param->Strides()[0];
  auto bs_ptr = (float *)fpga::fpga_malloc(2 * channel * sub_conv_n *
                                           sizeof(float));  // NOLINT

  for (int i = 0; i < channel * sub_conv_n; i++) {
    bs_ptr[i + sub_conv_n * channel] = 1;
    bs_ptr[i] = bias_ptr[i % (channel)];
  }

  PADDLE_MOBILE_ENFORCE(param->Strides()[1] == param->Strides()[0],
                        "stride_width should be equal to stride_height ");
  PADDLE_MOBILE_ENFORCE(filter->dims()[2] == filter->dims()[3],
                        "filter width should be equal to filter height ");
  PADDLE_MOBILE_ENFORCE(((filter->dims()[2] % param->Strides()[0]) == 0),
                        "filter axis should be the multiple of stride axis ");

  float max_value = fpga::filter_find_max(filter);
  fpga::format_deconv_filter(filter, max_value, param->Groups(),
                             param->Strides()[0]);

  int element_num_per_div =
      fpga::get_deconv_filter_num_per_div(filter, param->Groups(), sub_conv_n);

  //
  fpga::format_bias_scale_array(&bs_ptr, element_num_per_div,
                                channel * sub_conv_n);

  fpga::format_fp16_ofm(out);

  fpga::DeconvArgs deconv_arg = {0};
  fpga::fill_deconv_arg(&deconv_arg, input, out, filter, relu_enabled,
                        param->Groups(), param->Strides()[0],
                        param->Strides()[1], param->Paddings()[0],
                        param->Paddings()[1], bs_ptr);
  param->SetFpgaArgs(deconv_arg);

  return true;
}

template <>
void DeconvAddKernel<FPGA, float>::Compute(
    const FusionDeconvAddParam<FPGA> &param) {
  fpga::ComputeFpgaDeconv(param.FpgaArgs());
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
