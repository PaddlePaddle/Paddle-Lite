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

#include "concat_op.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void ConcatOp<Dtype, T>::InferShape() const {
  auto inputs = param_.Inputs();
  const size_t n = inputs.size();

  std::vector<DDim> inputs_dims;
  inputs_dims.reserve(n);
  for (int i = 0; i < n; i++) {
    inputs_dims.push_back(inputs[i]->dims());
  }

  auto axis = static_cast<size_t>(param_.Axis());

  if (n == 1) {
    DLOG << "Warning: concat op have only one input, "
            "may waste memory";
  }

  /// add all dim[axis] and check other dims if equal.
  auto out_dims = inputs_dims[0];
  int in_zero_dims_size = out_dims.size();
  for (size_t i = 1; i < n; i++) {
    for (size_t j = 0; j < in_zero_dims_size; j++) {
      if (j == axis) {
        out_dims[axis] += inputs_dims[i][j];
      } else {
        assert(out_dims[j] == inputs_dims[i][j]);
      }
    }
  }

  if (out_dims[axis] < 0) {
    out_dims[axis] = -1;
  }

  param_.Out()->Resize(out_dims);
}
template class ConcatOp<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile
