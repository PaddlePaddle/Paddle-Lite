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

#include "operators/kernel/feed_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool FeedKernel<FPGA, float>::Init(FeedParam<FPGA> *param) {
  Tensor *output = param->Out();
  fpga::format_fp16_ofm(output);
  return true;
}

template <>
void FeedKernel<FPGA, float>::Compute(const FeedParam<FPGA> &param) {
  auto input =
      reinterpret_cast<Tensor *>(const_cast<LoDTensor *>(param.InputX()));
  fpga::format_image(input);
  auto input_ptr = input->data<float>();
  Tensor *output = param.Out();
  auto output_ptr = output->data<float>();

  fpga::BypassArgs args = {fpga::DATA_TYPE_FP32};

  args.input_data_type = fpga::DATA_TYPE_FP32;
  args.output_data_type = fpga::DATA_TYPE_FP16;
  args.input_layout_type = fpga::LAYOUT_CHW;
  args.output_layout_type = fpga::LAYOUT_HWC;
  args.image.address = reinterpret_cast<void *>(input_ptr);
  args.image.channels = (uint32_t)input->dims()[1];
  args.image.height = (uint32_t)input->dims()[2];
  args.image.width = (uint32_t)input->dims()[3];
  args.image.pad_height = 0;
  args.image.pad_width = 0;
  args.output.address = output_ptr;
  args.output.scale_address = output->scale;
  fpga::PerformBypass(args);
}
template class FeedKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile
