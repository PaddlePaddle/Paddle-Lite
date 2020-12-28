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

#include "lite/backends/arm/math/gemm_s8.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename Dtype>
void gemm_s8(bool is_transA,
             bool is_transB,
             int M,
             int N,
             int K,
             const int8_t* A,
             const int8_t* B,
             Dtype* C,
             const float* bias,
             bool is_bias,
             const float* scale,
             const operators::ActivationParam act_param,
             ARMContext* ctx) {
  if (N == 1) {
    gemv_int8(A,
              B,
              C,
              false,
              M,
              K,
              scale,
              is_bias,
              bias,
              act_param.has_active,
              act_param.active_type,
              ctx,
              act_param.Relu_clipped_coef,
              act_param.Leaky_relu_alpha);

    return;
  }
  // TODO(chenjiao): fix the error of gemv_int8
  /*
  if (M == 1) {
    float bias_ptr[N];   // NOLINT
    float scale_ptr[N];  // NOLINT
    if (is_bias) {
      for (int i = 0; i < N; i++) {
        bias_ptr[i] = bias[0];
      }
    }
    memset(scale_ptr, scale[0], sizeof(float) * N);
    gemv_int8(B,
              A,
              C,
              true,
              M,
              K,
              scale,
              is_bias,
              bias,
              act_param.has_active,
              act_param.active_type,
              ctx,
              act_param.Relu_clipped_coef,
              act_param.Leaky_relu_alpha);
    return;
  }
  */

  int hblock = get_hblock_int8(ctx);
  int m_roundup = hblock * ((M + hblock - 1) / hblock);
  ctx->ExtendWorkspace(m_roundup * K * sizeof(int8_t));
  auto packed_A = static_cast<int8_t*>(ctx->workspace_data<int8_t>()) +
                  ctx->llc_size() / sizeof(int8_t);
  int lda = is_transA ? M : K;
  prepackA_int8(packed_A, A, lda, 0, M, 0, K, is_transA, ctx);

  gemm_prepack_int8<Dtype>(
      packed_A, B, bias, C, M, N, K, is_bias, is_transB, scale, act_param, ctx);
}

template void gemm_s8<float>(bool is_transA,
                             bool is_transB,
                             int M,
                             int N,
                             int K,
                             const int8_t* A,
                             const int8_t* B,
                             float* C,
                             const float* bias,
                             bool is_bias,
                             const float* scale,
                             const operators::ActivationParam act_param,
                             ARMContext* ctx);

template void gemm_s8<int8_t>(bool is_transA,
                              bool is_transB,
                              int M,
                              int N,
                              int K,
                              const int8_t* A,
                              const int8_t* B,
                              int8_t* C,
                              const float* bias,
                              bool is_bias,
                              const float* scale,
                              const operators::ActivationParam act_param,
                              ARMContext* ctx);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
