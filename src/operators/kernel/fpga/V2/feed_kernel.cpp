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
  DLOG << "col = " << col;
  auto input = const_cast<LoDTensor *>(&param->InputX()->at(col));

  if (output->dims().size() != 4) {
    input->init(type_id<float>().hash_code());
    return true;
  }
  input->init(type_id<int8_t>().hash_code());
  input->Resize(output->dims());
  fpga::format_ofm(output);
  return true;
}

template <>
void FeedKernel<FPGA, float>::Compute(const FeedParam<FPGA> &param) {
  auto output = param.Out();
  int col = param.Col();
  auto input = const_cast<LoDTensor *>(&param.InputX()->at(col));
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
  output->ShareDataWith(*input);
}
template class FeedKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile
