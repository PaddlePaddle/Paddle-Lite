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

#include "operators/kernel/mul_kernel.h"

namespace paddle_mobile {
namespace operators {

template<>
void
MulKernel<CPU, float, MulParam>::Compute(const MulParam &param) const {
  const Tensor *input_x = param.InputX();
  const Tensor *input_y = param.InputY();
  Tensor *out = param.Out();
  out->mutable_data<float>();
  const Tensor x_matrix =
      input_x->dims().size() > 2
      ? framework::ReshapeToMatrix(*input_x, param.XNumColDims())
      : *input_x;
  const Tensor y_matrix =
      input_y->dims().size() > 2
      ? framework::ReshapeToMatrix(*input_y, param.YNumColDims())
      : *input_y;
  auto out_dim = out->dims();
  if (out_dim.size() != 2) {
    out->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
  }
  math::matmul<float>(x_matrix, false, y_matrix, false,
                      static_cast<float>(1), out,
                      static_cast<float>(0));
  if (out_dim.size() != 2) {
    out->Resize(out_dim);
  }
}

template class MulKernel<CPU, float, MulParam>;

} // namespace operators
} // namespace paddle
