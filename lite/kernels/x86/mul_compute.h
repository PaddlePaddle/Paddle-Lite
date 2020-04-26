// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include "lite/backends/x86/math/blas.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"
#include "lite/fluid/eigen.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = lite::fluid::EigenMatrix<T, MajorType, IndexType>;

// using Tensor = framework::Tensor;
inline lite::Tensor ReshapeToMatrix(const lite::Tensor& src, int num_col_dims) {
  int rank = src.dims().size();
  if (rank == 2) {
    return src;
  }
  lite::Tensor res;
  res.ShareDataWith(src);
  res.Resize(src.dims().Flatten2D(num_col_dims));
  return res;
}

template <typename T>
class MulCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::MulParam;

  void Run() override {
    auto& param = *param_.get_mutable<operators::MulParam>();

    auto* z = param.output;

    auto* x = param.x;
    auto* y = param.y;

    Tensor x_matrix, y_matrix;
    if (x->dims().size() > 2) {
      x_matrix = ReshapeToMatrix(*x, param.x_num_col_dims);
    } else {
      x_matrix = *x;
    }

    if (y->dims().size() > 2) {
      y_matrix = ReshapeToMatrix(*y, param.y_num_col_dims);

    } else {
      y_matrix = *y;
    }

    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> mat_test(
        x_matrix.mutable_data<T>(), x_matrix.dims()[0], x_matrix.dims()[1]);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> mat1_test(
        y_matrix.mutable_data<T>(), y_matrix.dims()[0], y_matrix.dims()[1]);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> out_test(
        z->mutable_data<T>(), z->dims()[0], z->dims()[1]);
    out_test = mat_test * mat1_test;
  }

  virtual ~MulCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
