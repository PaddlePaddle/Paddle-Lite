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
  auto output = param->Out();
  int col = param->Col();
  auto input = const_cast<LoDTensor *>(&param->InputX()->at(col));
  input->init(typeid(float));
  input->Resize(output->dims());

  if (output->dims().size() != 4) {
    return true;
  }

  fpga::format_fp16_ofm(output);
  return true;
}

template <>
void FeedKernel<FPGA, float>::Compute(const FeedParam<FPGA> &param) {
  auto output = param.Out();
  int col = param.Col();
  auto input = const_cast<LoDTensor *>(&param.InputX()->at(col));
  std::type_index input_type = input->type();

  if (input_type == typeid(float)) {
    input->init(typeid(float));
  } else {  // input_type == typeid(int8_t)
    input->init(typeid(int8_t));
  }
  input->Resize(output->dims());

  if (output->dims().size() != 4) {
    size_t size = output->numel() * sizeof(float);
    auto output_ptr = output->data<float>();
    auto input_ptr = input->data<float>();
    auto external_ptr = reinterpret_cast<float *>(input->external_data);
    float *p_data = external_ptr == nullptr ? input_ptr : external_ptr;
    memcpy(output_ptr, p_data, size);
    input->external_data = nullptr;
    return;
  }

  fpga::format_image(input);
  auto output_ptr = output->data<half>();
  fpga::BypassArgs args = {fpga::DATA_TYPE_FP32};
  if (input_type == typeid(float)) {
    auto input_ptr = input->data<float>();
    auto external_ptr = reinterpret_cast<float *>(input->external_data);
    float *p_data = external_ptr == nullptr ? input_ptr : external_ptr;

    args.input_data_type = fpga::DATA_TYPE_FP32;
    args.output_data_type = fpga::DATA_TYPE_FP16;
    args.input_layout_type = fpga::LAYOUT_CHW;
    args.output_layout_type = fpga::LAYOUT_HWC;
    args.image.address = p_data;
    args.image.channels = (uint32_t)input->dims()[1];
    args.image.height = (uint32_t)input->dims()[2];
    args.image.width = (uint32_t)input->dims()[3];
    args.image.pad_height = 0;
    args.image.pad_width = 0;
    args.output.address = output_ptr;
    args.output.scale_address = output->scale;
    fpga::PerformBypass(args);
    input->external_data = nullptr;
  } else {  // input_type == typeid(int8_t)
    auto input_ptr = input->data<int8_t>();
    auto external_ptr = reinterpret_cast<int8_t *>(input->external_data);
    int8_t *p_data = external_ptr == nullptr ? input_ptr : external_ptr;

    args.input_data_type = fpga::DATA_TYPE_INT8;
    args.output_data_type = fpga::DATA_TYPE_FP16;
    args.input_layout_type = fpga::LAYOUT_CHW;
    args.output_layout_type = fpga::LAYOUT_HWC;
    args.image.address = p_data;
    args.image.channels = (uint32_t)input->dims()[1];
    args.image.height = (uint32_t)input->dims()[2];
    args.image.width = (uint32_t)input->dims()[3];
    args.image.pad_height = 0;
    args.image.pad_width = 0;
    args.output.address = output_ptr;
    args.output.scale_address = output->scale;
    fpga::PerformBypass(args);
    input->external_data = nullptr;
  }
}
template class FeedKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile
