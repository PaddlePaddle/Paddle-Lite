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
  return true;
}

template <>
void ConcatKernel<FPGA, float>::Compute(const ConcatParam<FPGA> &param) const {
  auto inputs = param.Inputs();
  auto *out = param.Out();
  int64_t axis = param.Axis();
  out->mutable_data<half>();

  DDim out_dim = out->dims();
  int pixels = out_dim[1] * out_dim[2];
  auto out_channel = out_dim[3];

  auto out_offset = 0;
  for (int i = 0; i < inputs.size(); ++i) {
    auto input = inputs[i];
    auto channels = input->dims()[3];
    out_offset += channels;
    auto src = input->data<half>();
    for (int j = 0; j < pixels; ++j) {
      auto dst = out->mutable_data<half>() + out_offset;
      memory::Copy(dst, src, sizeof(half));
    }
  }
}
template class ConcatKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
