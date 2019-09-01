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

#include "operators/kernel/pad2d_kernel.h"
namespace paddle_mobile {
namespace operators {
template <>
bool Pad2DKernel<FPGA, float>::Init(Pad2DParam<FPGA> *param) {
  Tensor *output = param->Out();
  fpga::format_fp16_ofm(output);
  return true;
}
void pad2dFunc(const framework::Tensor *input, framework::Tensor *output) {
  auto input_data = (input->data<half>());
  auto output_data = (output->data<half>());
  auto input_c = input->dims()[1];
  auto input_h = input->dims()[2];
  auto input_w = input->dims()[3];
  auto output_c = output->dims()[1];
  auto output_w = output->dims()[3];
  auto copysize = input_c * input_w;
  for (int h = 0; h < input_h; ++h) {
    auto input_offset = h * input_c * input_w;
    auto output_offset = h * paddle_mobile::fpga::align_to_x(
                                 output_c * output_w, IMAGE_ALIGNMENT);
    memcpy((output_data + output_offset), (input_data + input_offset),
           copysize * sizeof(half));
  }
}
template <>
void Pad2DKernel<FPGA, float>::Compute(const Pad2DParam<FPGA> &param) {
  auto in_x = param.InputX();
  auto out = param.Out();
  fpga::fpga_invalidate((void *)in_x->data<half>(),  // NOLINT
                        in_x->numel() * sizeof(half));
  pad2dFunc(in_x, out);
  (out->scale)[0] = (in_x->scale)[0];
  (out->scale)[1] = (in_x->scale)[1];
  DLOG << (out->scale)[0];
  DLOG << (out->scale)[1];
  size_t outputSize =
      out->dims()[2] *
      paddle_mobile::fpga::align_to_x((out->dims()[1]) * (out->dims()[3]),
                                      IMAGE_ALIGNMENT) *
      sizeof(half);
  fpga::fpga_flush(out->data<half>(), outputSize);
}
}  // namespace operators
}  // namespace paddle_mobile
