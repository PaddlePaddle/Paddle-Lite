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

#define INIT_PARAM                                                           \
  auto& ctx = this->ctx_->template As<X86Context>();                         \
  auto& param = *param_.get_mutable<operators::MatMulParam>();               \
  auto x_dims = param.X->dims();                                             \
  auto y_dims = param.Y->dims();                                             \
  int m, n, k;                                                               \
  int lda, ldb, ldc;                                                         \
  bool x_transpose = param.transpose_X;                                      \
  bool y_transpose = param.transpose_Y;                                      \
  if ((x_dims.size() >= 2 && y_dims.size() >= 2) &&                          \
      (x_dims.size() != 2 || y_dims.size() != 2)) {                          \
    if (!x_transpose) {                                                      \
      m = x_dims[x_dims.size() - 2];                                         \
      k = x_dims[x_dims.size() - 1];                                         \
      lda = k;                                                               \
    } else {                                                                 \
      m = x_dims[x_dims.size() - 1];                                         \
      k = x_dims[x_dims.size() - 2];                                         \
      lda = m;                                                               \
    }                                                                        \
    if (!y_transpose) {                                                      \
      n = y_dims[y_dims.size() - 1];                                         \
      ldb = n;                                                               \
      CHECK_EQ(k, y_dims[y_dims.size() - 2])                                 \
          << "k must be equal y_dims[y_dims.size() - 2]";                    \
    } else {                                                                 \
      n = y_dims[y_dims.size() - 2];                                         \
      ldb = k;                                                               \
      CHECK_EQ(k, y_dims[y_dims.size() - 1])                                 \
          << "k must be equal y_dims[y_dims.size() - 1]";                    \
    }                                                                        \
    ldc = n;                                                                 \
    if (x_dims.size() > 2 && y_dims.size() > 2) {                            \
      auto sum_x = x_dims.count(0, x_dims.size() - 2);                       \
      auto sum_y = y_dims.count(0, y_dims.size() - 2);                       \
      CHECK_EQ(sum_x, sum_y)                                                 \
          << "sum_x(x_dims[0]+..x_dims[size()-2]) must be equal with "       \
             "sum_y(y_dims[0]+..y_dims[size()-2])";                          \
    }                                                                        \
  } else if ((x_dims.size() == 2 && y_dims.size() == 2) ||                   \
             (x_dims.size() == 2 && y_dims.size() == 1)) {                   \
    if (!x_transpose) {                                                      \
      m = x_dims[0];                                                         \
      k = x_dims[1];                                                         \
      lda = k;                                                               \
    } else {                                                                 \
      m = x_dims[1];                                                         \
      k = x_dims[0];                                                         \
      lda = m;                                                               \
    }                                                                        \
    if (!y_transpose) {                                                      \
      if (y_dims.size() > 1) {                                               \
        n = y_dims[1];                                                       \
      } else {                                                               \
        n = 1;                                                               \
      }                                                                      \
      ldb = n;                                                               \
      CHECK_EQ(k, y_dims[0]) << "k must be equal y_dims[0]";                 \
    } else {                                                                 \
      if (y_dims.size() > 1) {                                               \
        n = y_dims[0];                                                       \
        CHECK_EQ(k, y_dims[1]) << "k must be equal y_dims[1]";               \
      } else {                                                               \
        n = 1;                                                               \
        CHECK_EQ(k, y_dims[0]) << "k must be equal y_dims[0]";               \
      }                                                                      \
      ldb = k;                                                               \
    }                                                                        \
    ldc = n;                                                                 \
  } else if (x_dims.size() >= 2 && y_dims.size() == 1) {                     \
    n = 1;                                                                   \
    k = y_dims[0];                                                           \
    if (!x_transpose) {                                                      \
      m = x_dims.count(0, x_dims.size() - 1);                                \
      CHECK_EQ(k, x_dims[x_dims.size() - 1])                                 \
          << "k must be equal x_dims[x_dims.size() - 1]";                    \
    } else {                                                                 \
      m = x_dims.count(1, x_dims.size() - 1);                                \
      CHECK_EQ(k, x_dims[0]) << "k must be equal x_dims[0]";                 \
    }                                                                        \
    lda = k;                                                                 \
    ldb = n;                                                                 \
    ldc = n;                                                                 \
  } else if (y_dims.size() >= 2 && x_dims.size() == 1) {                     \
    m = 1;                                                                   \
    k = x_dims[0];                                                           \
    if (!y_transpose) {                                                      \
      n = y_dims.count(1, y_dims.size());                                    \
      CHECK_EQ(k, y_dims[0]) << "k must be equal y_dims[0]";                 \
    } else {                                                                 \
      n = y_dims.count(0, y_dims.size() - 1);                                \
      CHECK_EQ(k, y_dims[y_dims.size() - 1])                                 \
          << "k must be equal y_dims[y_dims.size() - 1]";                    \
    }                                                                        \
    lda = k;                                                                 \
    ldb = n;                                                                 \
    ldc = n;                                                                 \
  } else if (x_dims.size() == 1 && y_dims.size() == 1) {                     \
    m = 1;                                                                   \
    n = 1;                                                                   \
    k = x_dims[0];                                                           \
    if (x_transpose == true && y_transpose == true) {                        \
      m = x_dims[0];                                                         \
      k = 1;                                                                 \
      n = y_dims[0];                                                         \
    } else if (x_transpose == false && y_transpose == false) {               \
      CHECK_EQ(x_dims[0], y_dims[0]) << "x_dims[0] must be equal y_dims[0]"; \
    } else {                                                                 \
      LOG(FATAL) << "not supported x_dims(" << x_dims << ") and y_dims("     \
                 << y_dims << ")"                                            \
                 << ", when x_transpose is " << x_transpose                  \
                 << " and y_transpose is " << y_transpose;                   \
    }                                                                        \
    lda = k;                                                                 \
    ldb = n;                                                                 \
    ldc = n;                                                                 \
  } else {                                                                   \
    LOG(FATAL) << "This x_dims: " << x_dims << " and y_dims: " << y_dims     \
               << " doesn't support!";                                       \
  }

template <typename T>
class MatMulV2Compute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::MatMulParam;

  void Run() override {
    INIT_PARAM;
    const auto* x_data = param.X->template data<T>();
    const auto* y_data = param.Y->template data<T>();
    auto* o_data = param.Out->template mutable_data<T>();
    auto o_dims = param.Out->dims();
    auto alpha = param.alpha;

    auto blas = lite::x86::math::GetBlas<lite::TargetType::kX86, T>(ctx);

    if ((x_dims.size() >= 2 && y_dims.size() >= 2) &&
        (x_dims.size() != 2 || y_dims.size() != 2)) {
      // x: [B, ..., M, K], y: [B, ..., K, N], out: [B, ..., M, N]
      // x: [B, M, K], y: [K, N], out: [B, M, N]
      // or
      // x: [M, K], y: [B, ..., K, N], out: [B, ..., M, N]
      // x: [M, K], y: [B, K, N], out: [B, M, N]
      int x_inner = x_dims[x_dims.size() - 2] * x_dims[x_dims.size() - 1];
      int y_inner = y_dims[y_dims.size() - 2] * y_dims[y_dims.size() - 1];
      int out_inner = o_dims[o_dims.size() - 2] * o_dims[o_dims.size() - 1];

      if (x_dims.size() > 2 && y_dims.size() > 2) {
        for (size_t i = 0; i < x_dims.count(0, x_dims.size() - 2); ++i) {
          blas.GEMM(x_transpose,
                    y_transpose,
                    m,
                    n,
                    k,
                    alpha,
                    x_data + i * x_inner,
                    lda,
                    y_data + i * y_inner,
                    ldb,
                    0.f,
                    o_data + i * out_inner,
                    ldc);
        }
      } else if (x_dims.size() > 2 && y_dims.size() == 2) {
        for (size_t i = 0; i < x_dims.count(0, x_dims.size() - 2); ++i) {
          blas.GEMM(x_transpose,
                    y_transpose,
                    m,
                    n,
                    k,
                    alpha,
                    x_data + i * x_inner,
                    lda,
                    y_data,
                    ldb,
                    0.f,
                    o_data + i * out_inner,
                    ldc);
        }
      } else if (x_dims.size() == 2 && y_dims.size() > 2) {
        for (size_t i = 0; i < y_dims.count(0, y_dims.size() - 2); ++i) {
          blas.GEMM(x_transpose,
                    y_transpose,
                    m,
                    n,
                    k,
                    alpha,
                    x_data,
                    lda,
                    y_data + i * y_inner,
                    ldb,
                    0.f,
                    o_data + i * out_inner,
                    ldc);
        }
      }
    } else if (x_dims.size() == 2 && y_dims.size() == 2) {
      // x: [M, K], y: [K, N], out: [M, N]
      blas.GEMM(x_transpose,
                y_transpose,
                m,
                n,
                k,
                alpha,
                x_data,
                lda,
                y_data,
                ldb,
                0.f,
                o_data,
                ldc);
    } else if (x_dims.size() >= 2 && y_dims.size() == 1) {
      // x: [B, M, K], y: [K], out: [B, M]
      blas.GEMM(x_transpose,
                false,
                m,
                n,
                k,
                alpha,
                x_data,
                lda,
                y_data,
                ldb,
                0.f,
                o_data,
                ldc);
    } else if (y_dims.size() >= 2 && x_dims.size() == 1) {
      // y: [B, K, N], x: [K], out: [B, N]
      blas.GEMM(false,
                y_transpose,
                m,
                n,
                k,
                alpha,
                x_data,
                lda,
                y_data,
                ldb,
                0.f,
                o_data,
                ldc);
    } else if (x_dims.size() == 1 && y_dims.size() == 1) {
      // x: [K], y: [K], out: [1]
      if (x_transpose == false && y_transpose == false) {
        o_data[0] = 0.;
        for (size_t i = 0; i < x_dims[0]; ++i) {
          o_data[0] += x_data[i] * y_data[i] * alpha;
        }
      } else if (x_transpose == true && y_transpose == true) {
        blas.GEMM(false,
                  false,
                  m,
                  n,
                  k,
                  alpha,
                  x_data,
                  lda,
                  y_data,
                  ldb,
                  0.f,
                  o_data,
                  ldc);
      } else {
        LOG(FATAL) << "not supported x_dims.(" << x_dims << ") and y_dims("
                   << y_dims << ")"
                   << ", and x_transpose: " << x_transpose
                   << ", y_transpose: " << y_transpose;
      }
    } else {
      LOG(FATAL) << "not supported x_dims(" << x_dims << ") and y_dims("
                 << y_dims << ")";
    }
  }

  virtual ~MatMulV2Compute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
