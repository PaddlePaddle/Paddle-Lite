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
  auto input = const_cast<LoDTensor *>(param->InputX());
  int col = param->Col();
  auto output = &(param->Out()->at(col));
  if (input->type() == typeid(float)) {
    return true;
  }
  output->init(typeid(float));
  output->Resize(input->dims());
  fpga::format_fp32_ofm(output);

  fpga::BypassArgs args = {fpga::DATA_TYPE_FP16};

  args.input_data_type = fpga::DATA_TYPE_FP16;
  args.output_data_type = fpga::DATA_TYPE_FP32;
  args.input_layout_type = fpga::LAYOUT_CHW;
  args.output_layout_type = fpga::LAYOUT_HWC;
  args.image.address = input->data<half>();
  args.image.channels = (uint32_t)product(input->dims());
  args.image.height = 1;
  args.image.width = 1;
  args.image.pad_height = 0;
  args.image.pad_width = 0;
  args.output.address = output->data<float>();
  args.output.scale_address = output->scale;
  param->fpga_bypass_args = args;

  return true;
}

template <>
void FetchKernel<FPGA, float>::Compute(const FetchParam<FPGA> &param) {
  auto input = param.InputX();
  if (input->type() == typeid(float)) {
    int col = param.Col();
    auto output = &(param.Out()->at(col));
    output->ShareDataWith(*input);
    return;
  }
  fpga::PerformBypass(param.fpga_bypass_args);
  fpga::fpga_invalidate(param.fpga_bypass_args.output.address,
                        param.fpga_bypass_args.image.channels * sizeof(float));

  // TODO: DEalign: get rid of extra 0
}

template class FetchKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile
