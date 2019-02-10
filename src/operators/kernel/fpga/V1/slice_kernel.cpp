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

#ifdef SLICE_OP

#include "operators/kernel/slice_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool SliceKernel<FPGA, float>::Init(SliceParam<FPGA>* param) {
  auto output = param->output_;
  fpga::format_fp16_ofm(output);
  DLOG << "input: " << param->input_;
  DLOG << "output: " << param->output_;
  if (param->input_->type() != typeid(half)) {
    DLOG << "wrong type";
  }
  return true;
}
template <>
void SliceKernel<FPGA, float>::Compute(const SliceParam<FPGA>& param) {
  // Only support slicing in channel dimension

  auto input = param.input_;
  DLOG << input;
  int HW = input->dims()[2] * input->dims()[3];
  int channel = input->dims()[1];
  auto input_ptr = input->data<half>();
  auto output_ptr = param.output_->data<half>();

  int start = param.starts_[0], end = param.ends_[0];
  start = start < 0 ? start + channel : start;
  end = end < 0 ? end + channel : end;
  start = start > channel ? channel : start;
  end = end > channel ? channel : end;
  int len = end - start;

  for (int i = 0; i < HW; i++) {
    memcpy(output_ptr + len * i, input_ptr + i * channel + start, len);
  }
}
}  // namespace operators
}  // namespace paddle_mobile
#endif
