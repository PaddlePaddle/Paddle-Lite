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
namespace paddle {
namespace lite_metal {
namespace kernels {
namespace x86 {

/**
 * Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
 * original x_dim is returned.
 */
static lite_metal::DDim RowMatrixFromVector(const lite_metal::DDim &x_dim) {
  if (x_dim.size() > 1) {
    return x_dim;
  }
  return lite_metal::DDim({1, x_dim[0]});
}

/**
 * Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
 * original y_dim is returned.
 */
static lite_metal::DDim ColumnMatrixFromVector(const lite_metal::DDim &y_dim) {
  if (y_dim.size() > 1) {
    return y_dim;
  }
  return lite_metal::DDim({y_dim[0], 1});
}

template <typename T>
class MatMulCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::MatMulParam;

  void Run() override {
    auto &context = ctx_->As<X86Context>();
    auto &param = *param_.get_mutable<operators::MatMulParam>();

    auto *x = param.X;
    auto *y = param.Y;
    auto *out = param.Out;
    out->template mutable_data<T>();

    auto blas = lite_metal::x86::math::GetBlas<lite_metal::TargetType::kX86, T>(context);
    auto mat_dim_a = lite_metal::x86::math::CreateMatrixDescriptor(
        RowMatrixFromVector(x->dims()), 0, param.transpose_X);
    auto mat_dim_b = lite_metal::x86::math::CreateMatrixDescriptor(
        ColumnMatrixFromVector(y->dims()), 0, param.transpose_Y);
    auto scale = static_cast<T>(param.alpha);
    blas.MatMul(*x, mat_dim_a, *y, mat_dim_b, scale, out, T(0));
  }

  virtual ~MatMulCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
