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

#include "lite/kernels/arm/matmul_compute.h"
#include <vector>
#include "lite/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void MatMulCompute::PrepareForRun() {
  auto& ctx = this->ctx_->template As<ARMContext>();
}

void MatMulCompute::Run() {
  auto& param = Param<param_t>();

  const auto* x_data = param.X->data<float>();
  const auto* y_data = param.Y->data<float>();
  auto* o_data = param.Out->mutable_data<float>();

  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  bool x_transpose = param.transpose_X;
  bool y_transpose = param.transpose_Y;
  float alpha = param.alpha;
  auto& ctx = this->ctx_->template As<ARMContext>();

  if (x_dims.size() > 2 && y_dims.size() >= 2) {
    // x: [B, ..., M, K], y: [B, ..., K, N], out: [B, ..., M, N]
    // x: [B, M, K], y: [K, N], out: [B, M, N]
    if (x_transpose || y_transpose) {
      LOG(FATAL) << "not supported transpose for x or y.";
    }
    CHECK_EQ(x_dims[x_dims.size() - 1], y_dims[y_dims.size() - 2])
        << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
        << ")";

    if (y_dims.size() > 2) {
      m_ = x_dims[x_dims.size() - 2];
      k_ = y_dims[y_dims.size() - 2];
      n_ = y_dims[y_dims.size() - 1];
      int hblock = lite::arm::math::get_hblock(ctx.arch());
      int m_round = 0;
      m_round = hblock * ((m_ + hblock - 1) / hblock);
      ctx.ExtendWorkspace(m_round * k_ * sizeof(float));
      int x_inner = x_dims[x_dims.size() - 2] * x_dims[x_dims.size() - 1];
      int y_inner = y_dims[y_dims.size() - 2] * y_dims[y_dims.size() - 1];
      int out_inner = x_dims[x_dims.size() - 2] * y_dims[y_dims.size() - 1];
      if (n_ == 1) {
        for (size_t i = 0; i < x_dims.count(0, x_dims.size() - 2); ++i) {
          lite::arm::math::sgemv(x_data + i * x_inner,
                                 y_data + i * y_inner,
                                 o_data + i * out_inner,
                                 false,
                                 m_,
                                 k_,
                                 false,
                                 nullptr,
                                 false);
        }
        if (fabsf(alpha - 1.f) > 1e-8f) {
          for (size_t i = 0; i < param.Out->dims().production(); ++i) {
            o_data[i] *= alpha;
          }
        }
      } else {
        for (size_t i = 0; i < x_dims.count(0, x_dims.size() - 2); ++i) {
          float* packed_x = static_cast<float*>(ctx.workspace_data<float>()) +
                            ctx.llc_size() / sizeof(float);
          lite::arm::math::prepackA(packed_x,
                                    x_data + i * x_inner,
                                    alpha,
                                    k_,
                                    0,
                                    m_,
                                    0,
                                    k_,
                                    false,
                                    &ctx);
          int ldb = n_;
          if (y_transpose) {
            ldb = k_;
          }
          lite::arm::math::sgemm_prepack(y_transpose,
                                         m_,
                                         n_,
                                         k_,
                                         packed_x,
                                         y_data + i * y_inner,
                                         ldb,
                                         0.f,
                                         o_data + i * out_inner,
                                         n_,
                                         nullptr,
                                         false,
                                         false,
                                         &ctx);
        }
      }
    } else {
      m_ = x_dims[x_dims.size() - 2];
      k_ = y_dims[0];
      n_ = y_dims[1];
      int hblock = lite::arm::math::get_hblock(ctx.arch());
      int m_round = 0;
      m_round = hblock * ((m_ + hblock - 1) / hblock);
      ctx.ExtendWorkspace(m_round * k_ * sizeof(float));
      int x_inner = x_dims[x_dims.size() - 2] * x_dims[x_dims.size() - 1];
      int out_inner = x_dims[x_dims.size() - 2] * y_dims[1];
      if (n_ == 1) {
        for (size_t i = 0; i < x_dims.count(0, x_dims.size() - 2); ++i) {
          lite::arm::math::sgemv(x_data + i * x_inner,
                                 y_data,
                                 o_data + i * out_inner,
                                 false,
                                 m_,
                                 k_,
                                 false,
                                 nullptr,
                                 false);
        }
        if (fabsf(param.alpha - 1.f) > 1e-8f) {
          for (size_t i = 0; i < param.Out->dims().production(); ++i) {
            o_data[i] *= param.alpha;
          }
        }
      } else {
        for (size_t i = 0; i < x_dims.count(0, x_dims.size() - 2); ++i) {
          float* packed_x = static_cast<float*>(ctx.workspace_data<float>()) +
                            ctx.llc_size() / sizeof(float);
          lite::arm::math::prepackA(packed_x,
                                    x_data + i * x_inner,
                                    alpha,
                                    k_,
                                    0,
                                    m_,
                                    0,
                                    k_,
                                    false,
                                    &ctx);
          int ldb = n_;
          if (y_transpose) {
            ldb = k_;
          }
          lite::arm::math::sgemm_prepack(y_transpose,
                                         m_,
                                         n_,
                                         k_,
                                         packed_x,
                                         y_data,
                                         ldb,
                                         0.f,
                                         o_data + i * out_inner,
                                         n_,
                                         nullptr,
                                         false,
                                         false,
                                         &ctx);
        }
      }
    }
  } else if (x_dims.size() == 2 && y_dims.size() == 2) {
    // x: [M, K], y: [K, N], out: [M, N]
    if (!x_transpose && !y_transpose) {
      CHECK_EQ(x_dims[1], y_dims[0])
          << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
          << "), x_transpose is " << x_transpose << ", y_transpose is "
          << y_transpose;
    } else if (!x_transpose && y_transpose) {
      CHECK_EQ(x_dims[1], y_dims[1])
          << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
          << "), x_transpose is " << x_transpose << ", y_transpose is "
          << y_transpose;
    } else if (x_transpose && !y_transpose) {
      CHECK_EQ(x_dims[0], y_dims[0])
          << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
          << "), x_transpose is " << x_transpose << ", y_transpose is "
          << y_transpose;
    } else {
      CHECK_EQ(x_dims[0], y_dims[1])
          << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
          << "), x_transpose is " << x_transpose << ", y_transpose is "
          << y_transpose;
    }
    // not supported transpose
    if (x_transpose || y_transpose) {
      LOG(FATAL) << "not supported transpose for x and y.";
    }
    m_ = x_dims[0];
    k_ = x_dims[1];
    n_ = y_dims[1];
    int hblock = lite::arm::math::get_hblock(ctx.arch());
    int m_round = 0;
    m_round = hblock * ((m_ + hblock - 1) / hblock);
    ctx.ExtendWorkspace(m_round * k_ * sizeof(float));

    if (n_ == 1) {
      lite::arm::math::sgemv(
          x_data, y_data, o_data, x_transpose, m_, k_, false, nullptr, false);
      if (fabsf(param.alpha - 1.f) > 1e-8f) {
        for (size_t i = 0; i < param.Out->dims().production(); ++i) {
          o_data[i] *= param.alpha;
        }
      }
    } else {
      float* packed_x = static_cast<float*>(ctx.workspace_data<float>()) +
                        ctx.llc_size() / sizeof(float);
      lite::arm::math::prepackA(
          packed_x, x_data, alpha, k_, 0, m_, 0, k_, x_transpose, &ctx);
      int ldb = n_;
      if (y_transpose) {
        ldb = k_;
      }
      lite::arm::math::sgemm_prepack(y_transpose,
                                     m_,
                                     n_,
                                     k_,
                                     packed_x,
                                     y_data,
                                     ldb,
                                     0.f,
                                     o_data,
                                     n_,
                                     nullptr,
                                     false,
                                     false,
                                     &ctx);
    }
  } else if (x_dims.size() > 2 && y_dims.size() == 1) {
    // x: [B, M, K], y: [K], out: [B, M]
    CHECK_EQ(x_dims[x_dims.size() - 1], y_dims[0])
        << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
        << ")";
    for (size_t i = 0; i < x_dims.count(0, x_dims.size() - 1); ++i) {
      o_data[i] = 0;
      for (size_t j = 0; j < y_dims[0]; ++j) {
        o_data[i] += x_data[i * y_dims[0] + j] * y_data[j] * alpha;
      }
    }
  } else if (x_dims.size() == 1 && y_dims.size() == 1) {
    // x: [K], y: [K], out: [1]
    if (x_dims[0] == y_dims[0] && x_transpose == false &&
        y_transpose == false) {
      o_data[0] = 0.;
      for (size_t i = 0; i < x_dims[0]; ++i) {
        o_data[0] += x_data[i] * y_data[i] * alpha;
      }
    }
    // x: [M], y: [N], x_transpose: true, y_transpose: true, out: [M, N]
    if (x_transpose == true && y_transpose == true) {
      m_ = x_dims[0];
      k_ = 1;
      n_ = y_dims[0];
      if (n_ == 1) {
        lite::arm::math::sgemv(
            x_data, y_data, o_data, false, m_, k_, false, nullptr, false);
        if (fabsf(alpha - 1.f) > 1e-8f) {
          for (size_t i = 0; i < param.Out->dims().production(); ++i) {
            o_data[i] *= alpha;
          }
        }
      } else {
        float* packed_x = static_cast<float*>(ctx.workspace_data<float>()) +
                          ctx.llc_size() / sizeof(float);
        lite::arm::math::prepackA(
            packed_x, x_data, alpha, k_, 0, m_, 0, k_, false, &ctx);
        int ldb = n_;
        lite::arm::math::sgemm_prepack(false,
                                       m_,
                                       n_,
                                       k_,
                                       packed_x,
                                       y_data,
                                       ldb,
                                       0.f,
                                       o_data,
                                       n_,
                                       nullptr,
                                       false,
                                       false,
                                       &ctx);
      }
    }
  } else {
    LOG(FATAL) << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
               << ")";
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    matmul, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::MatMulCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
