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
#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
#include "lite/backends/arm/math/sve/gemm_sve_i8mm.h"
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename Dtype>
void gemm_s8(bool is_transA,
             bool is_transB,
             bool packed_b,
             int M,
             int N,
             int K,
             const int8_t* A,
             const int8_t* B,
             Dtype* C,
             const float* bias,
             bool is_bias,
             GemmBiasDirection bias_direction,
             const float* scale,
             const operators::ActivationParam act_param,
             ARMContext* ctx) {
  if (N == 1) {
    gemv_int8(A, B, C, is_transA, M, K, scale, is_bias, bias, act_param, ctx);

    return;
  }
  if (M == 1) {
#ifdef TARGET_IOS
    float* bias_ptr = new float[N];
    float* scale_ptr = new float[N];
#else
    float bias_ptr[N];   // NOLINT
    float scale_ptr[N];  // NOLINT
#endif
    if (is_bias) {
      for (int i = 0; i < N; i++) {
        bias_ptr[i] = bias[0];
      }
    }
    for (int i = 0; i < N; i++) {
      scale_ptr[i] = scale[0];
    }
    gemv_int8(B,
              A,
              C,
              !is_transB,
              N,
              K,
              scale_ptr,
              is_bias,
              bias_ptr,
              act_param,
              ctx);
#ifdef TARGET_IOS
    delete[] bias_ptr;
    delete[] scale_ptr;
#endif
    return;
  }

  int hblock = get_hblock_int8(ctx);
  int m_roundup = hblock * ((M + hblock - 1) / hblock);
  ctx->ExtendWorkspace(m_roundup * K * sizeof(int8_t));
  auto packed_A = static_cast<int8_t*>(ctx->workspace_data<int8_t>()) +
                  ctx->llc_size() / sizeof(int8_t);
  int lda = is_transA ? M : K;
  prepackA_int8(packed_A, A, lda, 0, M, 0, K, is_transA, ctx);

  gemm_prepack_int8<Dtype>(packed_A,
                           B,
                           bias,
                           C,
                           M,
                           N,
                           K,
                           is_bias,
                           bias_direction,
                           is_transB,
                           packed_b,
                           scale,
                           act_param,
                           ctx);
}

template void gemm_s8<float>(bool is_transA,
                             bool is_transB,
                             bool packed_b,
                             int M,
                             int N,
                             int K,
                             const int8_t* A,
                             const int8_t* B,
                             float* C,
                             const float* bias,
                             bool is_bias,
                             GemmBiasDirection bias_direction,
                             const float* scale,
                             const operators::ActivationParam act_param,
                             ARMContext* ctx);

template void gemm_s8<int8_t>(bool is_transA,
                              bool is_transB,
                              bool packed_b,
                              int M,
                              int N,
                              int K,
                              const int8_t* A,
                              const int8_t* B,
                              int8_t* C,
                              const float* bias,
                              bool is_bias,
                              GemmBiasDirection bias_direction,
                              const float* scale,
                              const operators::ActivationParam act_param,
                              ARMContext* ctx);

#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
template <typename Dtype>
void gemm_sve(bool is_transA,
              bool is_transB,
              int M,
              int N,
              int K,
              const int8_t* A,
              const int8_t* B,
              Dtype* C,
              const float* bias,
              bool is_bias,
              GemmBiasDirection bias_direction,
              const float* scale,
              const operators::ActivationParam act_param,
              ARMContext* ctx) {
  if (N == 1) {
    gemv_int8(A, B, C, is_transA, M, K, scale, is_bias, bias, act_param, ctx);
    return;
  }
  if (M == 1) {
#ifdef TARGET_IOS
    float* bias_ptr = new float[N];
    float* scale_ptr = new float[N];
#else
    float bias_ptr[N];   // NOLINT
    float scale_ptr[N];  // NOLINT
#endif
    if (is_bias) {
      for (int i = 0; i < N; i++) {
        bias_ptr[i] = bias[0];
      }
    }
    for (int i = 0; i < N; i++) {
      scale_ptr[i] = scale[0];
    }
    gemv_int8(B,
              A,
              C,
              !is_transB,
              N,
              K,
              scale_ptr,
              is_bias,
              bias_ptr,
              act_param,
              ctx);
#ifdef TARGET_IOS
    delete[] bias_ptr;
    delete[] scale_ptr;
#endif
    return;
  }

  //! prepack
  Tensor tpackedA_sve;
  int hblock_sve = paddle::lite::arm::math::sve::get_hblock_int8_sve(ctx);
  int round_up_a_sve = ((hblock_sve + M - 1) / hblock_sve) * hblock_sve;
  int round_up_k_sve = 8 * ((K + 7) / 8);
  tpackedA_sve.Resize({round_up_a_sve * round_up_k_sve});
  int lda = is_transA ? M : K;
  paddle::lite::arm::math::sve::prepackA_int8_sve(
      tpackedA_sve.mutable_data<int8_t>(), A, lda, 0, M, 0, K, is_transA, ctx);
  // sve
  lite::arm::math::sve::gemm_prepack_int8_sve<Dtype>(
      tpackedA_sve.data<int8_t>(),
      B,
      bias,
      C,
      M,
      N,
      K,
      is_bias,
      is_transB,
      scale,
      act_param,
      ctx);
}

template void gemm_sve<float>(bool is_transA,
                              bool is_transB,
                              int M,
                              int N,
                              int K,
                              const int8_t* A,
                              const int8_t* B,
                              float* C,
                              const float* bias,
                              bool is_bias,
                              GemmBiasDirection bias_direction,
                              const float* scale,
                              const operators::ActivationParam act_param,
                              ARMContext* ctx);

template void gemm_sve<int8_t>(bool is_transA,
                               bool is_transB,
                               int M,
                               int N,
                               int K,
                               const int8_t* A,
                               const int8_t* B,
                               int8_t* C,
                               const float* bias,
                               bool is_bias,
                               GemmBiasDirection bias_direction,
                               const float* scale,
                               const operators::ActivationParam act_param,
                               ARMContext* ctx);
#endif

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
