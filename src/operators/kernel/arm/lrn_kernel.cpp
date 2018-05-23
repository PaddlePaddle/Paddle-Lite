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

#pragma once

#include "operators/kernel/lrn_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
void LrnKernel<CPU, float>::Compute(const LrnParam &param) const {
  const Tensor *input_x = param.InputX();
  auto x_dims = input_x->dims();
  /// data_format = NCHW
  const int N = x_dims[0];
  const int C = x_dims[1];
  const int H = x_dims[2];
  const int W = x_dims[3];
  Tensor *out = param.Out();
  out->mutable_data<float>();
  const int n = param.N();
  const float alpha = param.Alpha();
  const float beta = param.Beta();
  const float k = param.K();
  LRNFunctor<float> lrnFunctor;
  lrnFunctor(*input_x, out, N, C, H, W, n, k, alpha, beta);
}

template class LrnKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile
