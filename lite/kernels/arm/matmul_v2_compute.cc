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

#include "lite/kernels/arm/matmul_v2_compute.h"
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {
#define INIT_PARAM                                                           \
  auto& ctx = this->ctx_->template As<ARMContext>();                         \
  auto& param = Param<param_t>();                                            \
  auto x_dims = param.X->dims();                                             \
  auto y_dims = param.Y->dims();                                             \
  bool x_transpose = param.transpose_X;                                      \
  bool y_transpose = param.transpose_Y;                                      \
  if (last_x_shape_ == x_dims && last_y_shape_ == y_dims) {                  \
    return;                                                                  \
  }                                                                          \
  if ((x_dims.size() >= 2 && y_dims.size() >= 2) &&                          \
      (x_dims.size() != 2 || y_dims.size() != 2)) {                          \
    if (!x_transpose) {                                                      \
      m_ = x_dims[x_dims.size() - 2];                                        \
      k_ = x_dims[x_dims.size() - 1];                                        \
      lda_ = k_;                                                             \
    } else {                                                                 \
      m_ = x_dims[x_dims.size() - 1];                                        \
      k_ = x_dims[x_dims.size() - 2];                                        \
      lda_ = m_;                                                             \
    }                                                                        \
    if (!y_transpose) {                                                      \
      n_ = y_dims[y_dims.size() - 1];                                        \
      ldb_ = n_;                                                             \
      CHECK_EQ(k_, y_dims[y_dims.size() - 2])                                \
          << "k_ must be equal y_dims[y_dims.size() - 2]";                   \
    } else {                                                                 \
      n_ = y_dims[y_dims.size() - 2];                                        \
      ldb_ = k_;                                                             \
      CHECK_EQ(k_, y_dims[y_dims.size() - 1])                                \
          << "k_ must be equal y_dims[y_dims.size() - 1]";                   \
    }                                                                        \
    ldc_ = n_;                                                               \
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
      m_ = x_dims[0];                                                        \
      k_ = x_dims[1];                                                        \
      lda_ = k_;                                                             \
    } else {                                                                 \
      m_ = x_dims[1];                                                        \
      k_ = x_dims[0];                                                        \
      lda_ = m_;                                                             \
    }                                                                        \
    if (!y_transpose) {                                                      \
      if (y_dims.size() > 1) {                                               \
        n_ = y_dims[1];                                                      \
      } else {                                                               \
        n_ = 1;                                                              \
      }                                                                      \
      ldb_ = n_;                                                             \
      CHECK_EQ(k_, y_dims[0]) << "k_ must be equal y_dims[0]";               \
    } else {                                                                 \
      if (y_dims.size() > 1) {                                               \
        n_ = y_dims[0];                                                      \
        CHECK_EQ(k_, y_dims[1]) << "k_ must be equal y_dims[1]";             \
      } else {                                                               \
        n_ = 1;                                                              \
        CHECK_EQ(k_, y_dims[0]) << "k_ must be equal y_dims[0]";             \
      }                                                                      \
      ldb_ = k_;                                                             \
    }                                                                        \
    ldc_ = n_;                                                               \
  } else if (x_dims.size() >= 2 && y_dims.size() == 1) {                     \
    n_ = 1;                                                                  \
    k_ = y_dims[0];                                                          \
    if (!x_transpose) {                                                      \
      m_ = x_dims.count(0, x_dims.size() - 1);                               \
      CHECK_EQ(k_, x_dims[x_dims.size() - 1])                                \
          << "k_ must be equal x_dims[x_dims.size() - 1]";                   \
    } else {                                                                 \
      m_ = x_dims.count(1, x_dims.size() - 1);                               \
      CHECK_EQ(k_, x_dims[0]) << "k_ must be equal x_dims[0]";               \
    }                                                                        \
    lda_ = k_;                                                               \
    ldb_ = n_;                                                               \
    ldc_ = n_;                                                               \
  } else if (y_dims.size() >= 2 && x_dims.size() == 1) {                     \
    m_ = 1;                                                                  \
    k_ = x_dims[0];                                                          \
    if (!y_transpose) {                                                      \
      n_ = y_dims.count(1, y_dims.size());                                   \
      CHECK_EQ(k_, y_dims[0]) << "k_ must be equal y_dims[0]";               \
    } else {                                                                 \
      n_ = y_dims.count(0, y_dims.size() - 1);                               \
      CHECK_EQ(k_, y_dims[y_dims.size() - 1])                                \
          << "k_ must be equal y_dims[y_dims.size() - 1]";                   \
    }                                                                        \
    k_ = y_dims[y_dims.size() - 1];                                          \
    lda_ = k_;                                                               \
    ldb_ = n_;                                                               \
    ldc_ = n_;                                                               \
    CHECK_EQ(k_, x_dims[0]) << "k_ must be equal y_dims[0]";                 \
  } else if (x_dims.size() == 1 && y_dims.size() == 1) {                     \
    m_ = 1;                                                                  \
    n_ = 1;                                                                  \
    k_ = x_dims[0];                                                          \
    if (x_transpose == true && y_transpose == true) {                        \
      m_ = x_dims[0];                                                        \
      k_ = 1;                                                                \
      n_ = y_dims[0];                                                        \
    } else if (x_transpose == false && y_transpose == false) {               \
      CHECK_EQ(x_dims[0], y_dims[0]) << "x_dims[0] must be equal y_dims[0]"; \
    } else {                                                                 \
      LOG(FATAL) << "not supported x_dims(" << x_dims << ") and y_dims("     \
                 << y_dims << ")"                                            \
                 << ", when x_transpose is " << x_transpose                  \
                 << " and y_transpose is " << y_transpose;                   \
    }                                                                        \
    lda_ = k_;                                                               \
    ldb_ = n_;                                                               \
    ldc_ = n_;                                                               \
  }

void MatMulV2Compute::PrepareForRun() {
  INIT_PARAM
  last_x_shape_ = x_dims;
  last_y_shape_ = y_dims;
}

void MatMulV2Compute::Run() {
  auto& param = Param<param_t>();

  const auto* x_data = param.X->data<float>();
  const auto* y_data = param.Y->data<float>();
  auto* o_data = param.Out->mutable_data<float>();

  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  auto o_dims = param.Out->dims();
  bool x_transpose = param.transpose_X;
  bool y_transpose = param.transpose_Y;
  float alpha = param.alpha;
  auto& ctx = this->ctx_->template As<ARMContext>();

  operators::ActivationParam act_param;
  act_param.has_active = false;

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
        lite::arm::math::sgemm(x_transpose,
                               y_transpose,
                               m_,
                               n_,
                               k_,
                               alpha,
                               x_data + i * x_inner,
                               lda_,
                               y_data + i * y_inner,
                               ldb_,
                               0.f,
                               o_data + i * out_inner,
                               ldc_,
                               nullptr,
                               false,
                               act_param,
                               &ctx);
      }
    } else if (x_dims.size() > 2 && y_dims.size() == 2) {
      for (size_t i = 0; i < x_dims.count(0, x_dims.size() - 2); ++i) {
        lite::arm::math::sgemm(x_transpose,
                               y_transpose,
                               m_,
                               n_,
                               k_,
                               alpha,
                               x_data + i * x_inner,
                               lda_,
                               y_data,
                               ldb_,
                               0.f,
                               o_data + i * out_inner,
                               ldc_,
                               nullptr,
                               false,
                               act_param,
                               &ctx);
      }
    } else if (x_dims.size() == 2 && y_dims.size() > 2) {
      for (size_t i = 0; i < y_dims.count(0, y_dims.size() - 2); ++i) {
        lite::arm::math::sgemm(x_transpose,
                               y_transpose,
                               m_,
                               n_,
                               k_,
                               alpha,
                               x_data,
                               lda_,
                               y_data + i * y_inner,
                               ldb_,
                               0.f,
                               o_data + i * out_inner,
                               ldc_,
                               nullptr,
                               false,
                               act_param,
                               &ctx);
      }
    }
  } else if (x_dims.size() == 2 && y_dims.size() == 2) {
    // x: [M, K], y: [K, N], out: [M, N]
    lite::arm::math::sgemm(x_transpose,
                           y_transpose,
                           m_,
                           n_,
                           k_,
                           alpha,
                           x_data,
                           lda_,
                           y_data,
                           ldb_,
                           0.f,
                           o_data,
                           ldc_,
                           nullptr,
                           false,
                           act_param,
                           &ctx);
  } else if (x_dims.size() >= 2 && y_dims.size() == 1) {
    // x: [B, M, K], y: [K], out: [B, M]
    lite::arm::math::sgemm(x_transpose,
                           false,
                           m_,
                           n_,
                           k_,
                           alpha,
                           x_data,
                           lda_,
                           y_data,
                           ldb_,
                           0.f,
                           o_data,
                           ldc_,
                           nullptr,
                           false,
                           act_param,
                           &ctx);
  } else if (y_dims.size() >= 2 && x_dims.size() == 1) {
    // y: [B, K, N], x: [K], out: [B, N]
    lite::arm::math::sgemm(false,
                           y_transpose,
                           m_,
                           n_,
                           k_,
                           alpha,
                           x_data,
                           lda_,
                           y_data,
                           ldb_,
                           0.f,
                           o_data,
                           ldc_,
                           nullptr,
                           false,
                           act_param,
                           &ctx);
  } else if (x_dims.size() == 1 && y_dims.size() == 1) {
    // x: [K], y: [K], out: [1]
    lite::arm::math::sgemm(x_transpose,
                           y_transpose,
                           m_,
                           n_,
                           k_,
                           alpha,
                           x_data,
                           lda_,
                           y_data,
                           ldb_,
                           0.f,
                           o_data,
                           ldc_,
                           nullptr,
                           false,
                           act_param,
                           &ctx);
  } else {
    LOG(FATAL) << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
               << ")";
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(matmul_v2,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::MatMulV2Compute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
