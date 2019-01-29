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

#include "operators/kernel/fetch_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool FetchKernel<FPGA, float>::Init(FetchParam<FPGA> *param) {
  Tensor *output = param->Out();
  // fpga::format_fp16_ofm(output);
  return true;
}

template <>
void FetchKernel<FPGA, float>::Compute(const FetchParam<FPGA> &param) {
  param.Out()->ShareDataWith(*(param.InputX()));
  /*auto input =
          reinterpret_cast<Tensor *>(const_cast<Tensor *>(param.InputX()));
  fpga::format_image(input);
  auto input_ptr = input->data<float>();
  Tensor *output = param.Out();
  auto output_ptr = output->data<float>();

  fpga::BypassArgs args = {fpga::DATA_TYPE_FP16};

  args.input_data_type = fpga::DATA_TYPE_FP16;
  args.output_data_type = fpga::DATA_TYPE_FP32;
  args.input_layout_type = fpga::LAYOUT_CHW;
  args.output_layout_type = fpga::LAYOUT_HWC;
  args.image.address = reinterpret_cast<void *>(input_ptr);
  args.image.channels = (uint32_t)input->dims()[1];
  args.image.height = (input->dims().size() == 4) ? (uint32_t)input->dims()[2] :
  1; args.image.width = (input->dims().size() == 4) ? (uint32_t)input->dims()[3]
  : 1; args.image.pad_height = 0; args.image.pad_width = 0; args.output.address
  = output_ptr; args.output.scale_address = output->scale;
  fpga::PerformBypass(args);*/
}

template class FetchKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile
