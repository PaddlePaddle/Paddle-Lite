/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#pragma once

#include "operators/kernel/lrn_kernel.h"

namespace paddle_mobile {
namespace operators {

template <> void LrnKernel<CPU, float>::Compute(const LrnParam &param) const {
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

} // namespace operators
} // namespace paddle_mobile
