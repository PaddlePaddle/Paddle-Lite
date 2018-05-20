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

#include "mul_op.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void MulOp<Dtype, T>::InferShape() const {
  auto x_dims = param_.InputX()->dims();
  auto y_dims = param_.InputY()->dims();
  int x_num_col_dims = param_.XNumColDims();
  int y_num_col_dims = param_.YNumColDims();

  assert(x_dims.size() > x_num_col_dims);
  assert(y_dims.size() > y_num_col_dims);

  /// (1,2,3,4) , x_num_col_dims = 2  -> (2,12)
  auto x_mat_dims = framework::flatten_to_2d(x_dims, x_num_col_dims);
  auto y_mat_dims = framework::flatten_to_2d(y_dims, y_num_col_dims);

  assert(x_mat_dims[1] == y_mat_dims[0]);

  std::vector<int64_t> output_dims;
  output_dims.reserve(
      static_cast<size_t>(x_num_col_dims + y_dims.size() - y_num_col_dims));

  for (int i = 0; i < x_num_col_dims; ++i) {
    output_dims.push_back(x_dims[i]);
  }

  for (int i = y_num_col_dims; i < y_dims.size(); ++i) {
    output_dims.push_back(y_dims[i]);
  }

  framework::DDim ddim = framework::make_ddim(output_dims);
  param_.Out()->Resize(ddim);
}
template class MulOp<CPU, float>;
}  // namespace operators
}  // namespace paddle_mobile
