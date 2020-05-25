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
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class SearchAlignedMatMulCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::MatMulParam;

  void Run() override {
    auto& context = ctx_->As<X86Context>();
    auto& param = *param_.get_mutable<operators::MatMulParam>();

    auto x = param.X;
    auto y = param.Y;
    auto out = param.Out;
    bool x_transpose = param.transpose_X;
    bool y_transpose = param.transpose_Y;
    float alpha = param.alpha;
    const auto x_dims = x->dims();
    const auto y_dims = y->dims();
    const auto& x_lod = x->lod();
    const auto& y_lod = y->lod();
    const auto& x_lod_0 = x_lod[0];
    const auto& y_lod_0 = y_lod[0];

    int seq_num = x_lod_0.size() - 1;
    int x_inner_size = x_dims[1];
    int y_inner_size = y_dims[1];
    int x_batch_size = x_lod_0[1];
    int y_batch_size = y_lod_0[1];
    int M = x_transpose ? x_inner_size : x_batch_size;
    int N = y_transpose ? y_batch_size : y_inner_size;
    int X_K = x_transpose ? x_batch_size : x_inner_size;
    int Y_K = y_transpose ? y_inner_size : y_batch_size;
    CHECK_EQ(X_K, Y_K) << "K of Input(X) and Input(Y) is not equal";
    int K = X_K;

    lite::x86::math::MatDescriptor mat_dim_a;
    mat_dim_a.height_ = M;
    mat_dim_a.width_ = K;
    mat_dim_a.stride_ = x_batch_size * x_inner_size;
    mat_dim_a.batch_size_ = seq_num;
    mat_dim_a.trans_ = x_transpose;
    lite::x86::math::MatDescriptor mat_dim_b;
    mat_dim_b.height_ = K;
    mat_dim_b.width_ = N;
    mat_dim_b.stride_ = y_batch_size * y_inner_size;
    mat_dim_b.batch_size_ = seq_num;
    mat_dim_b.trans_ = y_transpose;
    auto blas = lite::x86::math::GetBlas<lite::TargetType::kX86, T>(context);
    blas.MatMul(*x, mat_dim_a, *y, mat_dim_b, static_cast<T>(alpha), out, T(0));
  }

  virtual ~SearchAlignedMatMulCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
