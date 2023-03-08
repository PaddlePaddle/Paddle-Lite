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
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {
#define DELETE_DIM_ONE(old_dims, now_dims)                          \
  if (old_dims.size() == 1) {                                       \
    now_dims.push_back(old_dims[0]);                                \
  } else {                                                          \
    for (int i = 0; i < old_dims.size(); i++) {                     \
      if (old_dims[i] == 1) {                                       \
        while ((i + 1) < old_dims.size() && old_dims[i + 1] == 1) { \
          i++;                                                      \
        }                                                           \
        while (i + 1 < old_dims.size()) {                           \
          now_dims.push_back(old_dims[i + 1]);                      \
          i++;                                                      \
        }                                                           \
        break;                                                      \
      } else {                                                      \
        now_dims.push_back(old_dims[i]);                            \
      }                                                             \
    }                                                               \
  }

#define INIT_PARAM                                                           \
  auto& ctx = this->ctx_->template As<ARMContext>();                         \
  auto& param = Param<param_t>();                                            \
  auto x_dims = param.X->dims();                                             \
  auto y_dims = param.Y->dims();                                             \
  std::vector<int64_t> x_shape, y_shape;                                     \
  int k = 0;                                                                 \
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
    lda_ = k_;                                                               \
    ldb_ = n_;                                                               \
    ldc_ = n_;                                                               \
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
  } else {                                                                   \
    LOG(FATAL) << "This x_dims: " << x_dims << " and y_dims: " << y_dims     \
               << " doesn't support!";                                       \
  }

template <>
void MatMulV2Compute<PRECISION(kFloat), PRECISION(kFloat)>::ReInitWhenNeeded() {
  auto& param1 = Param<param_t>();
  INIT_PARAM
  last_x_shape_ = x_dims;
  last_y_shape_ = y_dims;
}

template <>
void MatMulV2Compute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
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
    if (x_transpose == false && y_transpose == false) {
      o_data[0] = 0.;
      for (size_t i = 0; i < x_dims[0]; ++i) {
        o_data[0] += x_data[i] * y_data[i] * alpha;
      }
    } else if (x_transpose == true && y_transpose == true) {
      lite::arm::math::sgemm(false,
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
    } else {
      LOG(FATAL) << "not supported x_dims.(" << x_dims << ") and y_dims("
                 << y_dims << ")"
                 << ", and x_transpose: " << x_transpose
                 << ", y_transpose: " << y_transpose;
    }
  } else {
    LOG(FATAL) << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
               << ")";
  }
}

template <>
void MatMulV2Compute<PRECISION(kInt8), PRECISION(kFloat)>::ReInitWhenNeeded() {
  INIT_PARAM
  const float alpha = param.alpha;
  scale_.resize(n_);
  scale_one.resize(m_);

  if (param.weight_scale.size() == 1) {
    param.output_scale = param.input_scale * param.weight_scale[0] * alpha;
    for (int i = 0; i < n_; i++) {
      scale_[i] = param.output_scale;
    }
  } else {
    for (int i = 0; i < n_; i++) {
      param.output_scale = param.input_scale * param.weight_scale[i] * alpha;
      scale_[i] = param.output_scale;
    }
  }
  for (int i = 0; i < m_; i++) {
    scale_one[i] = 1;
  }
  last_x_shape_ = x_dims;
  last_y_shape_ = y_dims;
}

void matmulv2_add_n_scale_bias(float* o_data, float* scale, int m, int n) {
  int n_tail = n % 4;
  int n_inner = n - n_tail;
  for (int i = 0; i < m; i++) {
    float* o_m_data = o_data + i * n;
    float* scale_ptr = scale;
    for (int j = 0; j < n_inner; j += 4) {
      float32x4_t vdin = vld1q_f32(o_m_data);
      float32x4_t vscale = vld1q_f32(scale_ptr);
      float32x4_t vsum = vmulq_f32(vscale, vdin);
      vst1q_f32(o_m_data, vsum);
      o_m_data += 4;
      scale_ptr += 4;
    }
    for (int j = n_inner; j < n; j++) {
      *o_m_data *= *scale_ptr;
      scale_ptr++;
      o_m_data++;
    }
  }
}

#define MatMulV2ComputeKernel(func)                                        \
  if ((x_dims.size() >= 2 && y_dims.size() >= 2) &&                        \
      (x_dims.size() != 2 || y_dims.size() != 2)) {                        \
    int x_inner = x_dims[x_dims.size() - 2] * x_dims[x_dims.size() - 1];   \
    int y_inner = y_dims[y_dims.size() - 2] * y_dims[y_dims.size() - 1];   \
    int out_inner = o_dims[o_dims.size() - 2] * o_dims[o_dims.size() - 1]; \
    if (x_dims.size() > 2 && y_dims.size() > 2) {                          \
      for (size_t i = 0; i < x_dims.count(0, x_dims.size() - 2); ++i) {    \
        lite::arm::math::gemm_##func(x_transpose,                          \
                                     y_transpose,                          \
                                     false,                                \
                                     m_,                                   \
                                     n_,                                   \
                                     k_,                                   \
                                     x_data + i * x_inner,                 \
                                     y_data + i * y_inner,                 \
                                     o_data + i * out_inner,               \
                                     nullptr,                              \
                                     false,                                \
                                     lite::arm::math::GemmNoBias,          \
                                     scale_one.data(),                     \
                                     act_param,                            \
                                     &ctx);                                \
        matmulv2_add_n_scale_bias(                                         \
            o_data + i * out_inner, scale_.data(), m_, n_);                \
      }                                                                    \
    } else if (x_dims.size() > 2 && y_dims.size() == 2) {                  \
      for (size_t i = 0; i < x_dims.count(0, x_dims.size() - 2); ++i) {    \
        lite::arm::math::gemm_##func(x_transpose,                          \
                                     y_transpose,                          \
                                     false,                                \
                                     m_,                                   \
                                     n_,                                   \
                                     k_,                                   \
                                     x_data + i * x_inner,                 \
                                     y_data,                               \
                                     o_data + i * out_inner,               \
                                     nullptr,                              \
                                     false,                                \
                                     lite::arm::math::GemmNoBias,          \
                                     scale_one.data(),                     \
                                     act_param,                            \
                                     &ctx);                                \
        matmulv2_add_n_scale_bias(                                         \
            o_data + i * out_inner, scale_.data(), m_, n_);                \
      }                                                                    \
    } else if (x_dims.size() == 2 && y_dims.size() > 2) {                  \
      for (size_t i = 0; i < y_dims.count(0, y_dims.size() - 2); ++i) {    \
        lite::arm::math::gemm_##func(x_transpose,                          \
                                     y_transpose,                          \
                                     false,                                \
                                     m_,                                   \
                                     n_,                                   \
                                     k_,                                   \
                                     x_data,                               \
                                     y_data + i * y_inner,                 \
                                     o_data + i * out_inner,               \
                                     nullptr,                              \
                                     false,                                \
                                     lite::arm::math::GemmNoBias,          \
                                     scale_one.data(),                     \
                                     act_param,                            \
                                     &ctx);                                \
        matmulv2_add_n_scale_bias(                                         \
            o_data + i * out_inner, scale_.data(), m_, n_);                \
      }                                                                    \
    }                                                                      \
  } else if ((x_dims.size() == 2 && y_dims.size() == 2)) {                 \
    lite::arm::math::gemm_##func(x_transpose,                              \
                                 y_transpose,                              \
                                 false,                                    \
                                 m_,                                       \
                                 n_,                                       \
                                 k_,                                       \
                                 x_data,                                   \
                                 y_data,                                   \
                                 o_data,                                   \
                                 nullptr,                                  \
                                 false,                                    \
                                 lite::arm::math::GemmNoBias,              \
                                 scale_one.data(),                         \
                                 act_param,                                \
                                 &ctx);                                    \
    matmulv2_add_n_scale_bias(o_data, scale_.data(), m_, n_);              \
  } else if (x_dims.size() >= 2 && y_dims.size() == 1) {                   \
    lite::arm::math::gemm_##func(x_transpose,                              \
                                 false,                                    \
                                 false,                                    \
                                 m_,                                       \
                                 n_,                                       \
                                 k_,                                       \
                                 x_data,                                   \
                                 y_data,                                   \
                                 o_data,                                   \
                                 nullptr,                                  \
                                 false,                                    \
                                 lite::arm::math::GemmNoBias,              \
                                 scale_one.data(),                         \
                                 act_param,                                \
                                 &ctx);                                    \
    matmulv2_add_n_scale_bias(o_data, scale_.data(), m_, n_);              \
  } else if (y_dims.size() >= 2 && x_dims.size() == 1) {                   \
    lite::arm::math::gemm_##func(false,                                    \
                                 y_transpose,                              \
                                 false,                                    \
                                 m_,                                       \
                                 n_,                                       \
                                 k_,                                       \
                                 x_data,                                   \
                                 y_data,                                   \
                                 o_data,                                   \
                                 nullptr,                                  \
                                 false,                                    \
                                 lite::arm::math::GemmNoBias,              \
                                 scale_one.data(),                         \
                                 act_param,                                \
                                 &ctx);                                    \
    matmulv2_add_n_scale_bias(o_data, scale_.data(), m_, n_);              \
  } else if (x_dims.size() == 1 && y_dims.size() == 1) {                   \
    if (x_transpose == false && y_transpose == false) {                    \
      o_data[0] = 0.;                                                      \
      for (size_t i = 0; i < x_dims[0]; ++i) {                             \
        o_data[0] += x_data[i] * y_data[i];                                \
      }                                                                    \
    } else if (x_transpose == true && y_transpose == true) {               \
      lite::arm::math::gemm_##func(false,                                  \
                                   false,                                  \
                                   false,                                  \
                                   m_,                                     \
                                   n_,                                     \
                                   k_,                                     \
                                   x_data,                                 \
                                   y_data,                                 \
                                   o_data,                                 \
                                   nullptr,                                \
                                   false,                                  \
                                   lite::arm::math::GemmNoBias,            \
                                   scale_one.data(),                       \
                                   act_param,                              \
                                   &ctx);                                  \
    } else {                                                               \
      LOG(FATAL) << "not supported x_dims.(" << x_dims << ") and y_dims("  \
                 << y_dims << ")"                                          \
                 << ", and x_transpose: " << x_transpose                   \
                 << ", y_transpose: " << y_transpose;                      \
    }                                                                      \
    matmulv2_add_n_scale_bias(o_data, scale_.data(), m_, n_);              \
  } else {                                                                 \
    LOG(FATAL) << "not supported x_dims(" << x_dims << ") and y_dims("     \
               << y_dims << ")";                                           \
  }

template <>
void MatMulV2Compute<PRECISION(kInt8), PRECISION(kFloat)>::Run() {
  auto& param = Param<param_t>();
  const auto* x_data = param.X->data<int8_t>();
  const auto* y_data = param.Y->data<int8_t>();
  auto* o_data = param.Out->mutable_data<float>();
  const float alpha = param.alpha;
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  auto o_dims = param.Out->dims();
  bool x_transpose = param.transpose_X;
  bool y_transpose = param.transpose_Y;
  auto& ctx = this->ctx_->template As<ARMContext>();
  operators::ActivationParam act_param;
  act_param.has_active = false;

#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
  if (ctx.has_sve2_i8mm()) {
    MatMulV2ComputeKernel(sve);
    return;
  }
#endif
  MatMulV2ComputeKernel(s8);
}

#ifdef ENABLE_ARM_FP16
template <>
void MatMulV2Compute<PRECISION(kFP16), PRECISION(kFP16)>::ReInitWhenNeeded() {
  INIT_PARAM
  last_x_shape_ = x_dims;
  last_y_shape_ = y_dims;
}

template <>
void MatMulV2Compute<PRECISION(kFP16), PRECISION(kFP16)>::Run() {
  auto& param = Param<param_t>();

  const auto* x_data = param.X->data<float16_t>();
  const auto* y_data = param.Y->data<float16_t>();
  auto* o_data = param.Out->mutable_data<float16_t>();

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
        lite::arm::math::fp16::sgemm_fp16(x_transpose,
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
        lite::arm::math::fp16::sgemm_fp16(x_transpose,
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
        lite::arm::math::fp16::sgemm_fp16(x_transpose,
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
    lite::arm::math::fp16::sgemm_fp16(x_transpose,
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
    lite::arm::math::fp16::sgemm_fp16(x_transpose,
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
    lite::arm::math::fp16::sgemm_fp16(false,
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
    if (x_transpose == false && y_transpose == false) {
      o_data[0] = 0.;
      for (size_t i = 0; i < x_dims[0]; ++i) {
        o_data[0] += x_data[i] * y_data[i] * alpha;
      }
    } else if (x_transpose == true && y_transpose == true) {
      lite::arm::math::fp16::sgemm_fp16(false,
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
    } else {
      LOG(FATAL) << "not supported x_dims.(" << x_dims << ") and y_dims("
                 << y_dims << ")"
                 << ", and x_transpose: " << x_transpose
                 << ", y_transpose: " << y_transpose;
    }
  } else {
    LOG(FATAL) << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
               << ")";
  }
}
#endif

#undef INIT_PARAM
#undef DELETE_DIM_ONE
}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
typedef paddle::lite::kernels::arm::MatMulV2Compute<PRECISION(kFloat),
                                                    PRECISION(kFloat)>
    Matmulv2_f32_f32;

typedef paddle::lite::kernels::arm::MatMulV2Compute<PRECISION(kInt8),
                                                    PRECISION(kFloat)>
    Matmulv2_int8_f32;

#ifdef ENABLE_ARM_FP16
typedef paddle::lite::kernels::arm::MatMulV2Compute<PRECISION(kFP16),
                                                    PRECISION(kFP16)>
    Matmulv2_f16_f16;

REGISTER_LITE_KERNEL(matmul_v2, kARM, kFP16, kNCHW, Matmulv2_f16_f16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();
#endif  // ENABLE_ARM_FP16

REGISTER_LITE_KERNEL(matmul_v2, kARM, kFloat, kNCHW, Matmulv2_f32_f32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(matmul_v2, kARM, kInt8, kNCHW, Matmulv2_int8_f32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();
