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

#ifdef MUL_OP

#pragma once

namespace paddle_mobile {
namespace operators {

// 1、如果x,y维度都是2维，
// x = [[1,2],   y = [[5,6],
//      [3,4]]        [7,8]]
// 运算结果为正常矩阵相乘。结果 out =
//  [[1*5+2*7,1*6+2*8],[3*5+4*7, 3*6+4*8]]
//
// 2、如果x的维度大于2或者y的维度大于2,x的维度(2,3,4) ,y的维度(4,1,2)
// x = [[[1,2,3,4],
//       [2,3,4,5],
//       [3,4,5,6]],
//      [[1,2,3,4],
//       [2,3,4,5],
//       [3,4,5,6]]]
// y = [[[1,2]],
//      [[3,4]],
//      [[5,6]],
//      [[7,8]]]
// 需要借助x_num_col_dims和y_num_col_dims将x和y的维度转换为2维
// 从模型中读到参数,x_num_col_dims = 2,y_num_col_dims = 1,左开右闭
// (1) 将x = (2,3,4)的index [0,x_num_col_dims)部分2,3相乘，得到6，
//     [x_num_col_dims,xdim.size())部分4相乘，得到4，
//     将Tensor x的dims重写成(6,4)
// (2) 将y = (4,1,2)的index [0,y_num_col_dims)部分4相乘，得到4，
//     [y_num_col_dims,ydim.size())部分1,2相乘，得到2,
//     将Tensor y的dims重写成(4,2)
// 并不影响x,y在内存中的分布。
// x = [[1,2,3,4],             y = [[1,2],
//      [2,3,4,5],                  [3,4],
//      [3,4,5,6],   矩阵乘法        [5,6],
//      [1,2,3,4],                  [7,8]]
//      [2,3,4,5],
//      [3,4,5,6]]
// 结果x(6行4列)乘y(4行2列)，按1中矩阵相乘，结果out(6行2列)

template <typename P>
void MulCompute(const MulParam<CPU> &param) {
  const Tensor *input_x = param.InputX();
  const Tensor *input_y = param.InputY();
  Tensor *out = param.Out();

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
  if (param.InputX()->type() == typeid(int8_t)) {
    out->mutable_data<int32_t>();
    math::matmul<int8_t, int32_t>(x_matrix, false, y_matrix, false,
                                  static_cast<float>(1), out,
                                  static_cast<float>(0));
  } else {
    out->mutable_data<float>();
    math::matmul<float, float>(x_matrix, false, y_matrix, false,
                               static_cast<float>(1), out,
                               static_cast<float>(0));
  }
  if (out_dim.size() != 2) {
    out->Resize(out_dim);
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
