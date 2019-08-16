/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

//
// Created by hujie09 on 2019-07-16.
//

#ifdef EXP_OP
#pragma once

#include <math.h>
#include <operators/kernel/exp_kernel.h>
namespace paddle_mobile {
namespace operators {
template <>
bool EXPKernel<CPU, float>::Init(
    paddle_mobile::operators::EXPParam<paddle_mobile::CPU> *param) {
  return true;
}

template <>
void EXPKernel<CPU, float>::Compute(
    const paddle_mobile::operators::EXPParam<paddle_mobile::CPU> &param) {
  const auto input_ = param.InputX();
  auto output = param.Out();
  float *output_data = output->mutable_data<float>();
  const float *input_data = input_->data<float>();
  for (int i = 0; i < output->numel(); ++i, output_data++, input_data++) {
    *output_data = exp(*input_data);
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // EXP_OP
