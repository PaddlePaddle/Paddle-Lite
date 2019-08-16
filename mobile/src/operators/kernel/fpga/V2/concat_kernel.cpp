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

#ifdef CONCAT_OP

#include "operators/kernel/concat_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConcatKernel<FPGA, float>::Init(ConcatParam<FPGA> *param) {
  auto inputs = param->Inputs();
  auto out = param->Out();
  auto image_num = inputs.size();
  auto images_in =
      (int8_t **)fpga::fpga_malloc(image_num * sizeof(int8_t *));  // NOLINT
  auto scales_in =
      (float **)fpga::fpga_malloc(image_num * sizeof(float *));  // NOLINT
  auto channel_num =
      (uint32_t *)fpga::fpga_malloc(image_num * sizeof(uint32_t));  // NOLINT

  auto height = inputs[0]->dims()[2];
  auto width = inputs[0]->dims()[3];
  for (int i = 0; i < image_num; i++) {
    auto input = inputs[i];
    PADDLE_MOBILE_ENFORCE(
        input->dims()[2] == height && input->dims()[3] == width,
        "Image height & width should be unified");
    images_in[i] = input->data<int8_t>();
    channel_num[i] = (uint32_t)inputs[i]->dims()[1];  // NOLINT
    scales_in[i] = input->scale;
  }
  fpga::format_concat_output(out, height, width, image_num, channel_num);

  fpga::ConcatArgs concatArgs = {0};
  concatArgs.image_num = image_num;
  concatArgs.images_in = images_in;
  concatArgs.scales_in = scales_in;
  concatArgs.image_out = out->data<int8_t>();
  concatArgs.scale_out = out->scale;
  concatArgs.channel_num = channel_num;
  concatArgs.height = height;
  concatArgs.width = width;
  param->SetFpgaArgs(concatArgs);
  return true;
}

template <>
void ConcatKernel<FPGA, float>::Compute(const ConcatParam<FPGA> &param) {
  ComputeFPGAConcat(param.FpgaArgs());
}
template class ConcatKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
