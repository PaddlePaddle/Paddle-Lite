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

#include "operators/kernel/dequantize_kernel.h"

namespace paddle_mobile {
namespace operators {

template<>
bool DequantizeKernel<X86, float>::Init(DequantizeParam<X86> *param) {
  return true;
}

template<>
void DequantizeKernel<X86, float>::Compute(
    const DequantizeParam<X86> &param) const {
  // TODO
  const Tensor *input = param.input_;
  Tensor *output = param.out_; 
  float activation_scale = param.activation_scale_->data<float>()[0];
  float weight_scale = param.weight_scale_;
  const int32_t *x = input->data<const int32_t>();
  float *y = output->mutable_data<float>();
  for (size_t i = 0; i < output->numel(); ++i) {
    y[i] = x[i] / activation_scale / weight_scale;
  }
}

}  // namespace paddle_mobile
}  // namespace operators
