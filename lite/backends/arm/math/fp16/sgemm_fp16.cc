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

#include "lite/backends/arm/math/fp16/sgemm_fp16.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

void sgemm_fp16(bool is_transA,
                bool is_transB,
                int M,
                int N,
                int K,
                float16_t alpha,
                const float16_t* A,
                int lda,
                const float16_t* B,
                int ldb,
                float16_t beta,
                float16_t* C,
                int ldc,
                const float16_t* bias,
                bool is_bias,
                const operators::ActivationParam act_param,
                ARMContext* ctx) {
  // alpha default is 1;
  bool has_alpha = fabsf(alpha - 1.f) > 1e-8f ? 1 : 0;
  if (N == 1 && !has_alpha && ldc == N) {
    gemv_fp16(A,
              B,
              C,
              is_transA,
              M,
              K,
              beta,
              is_bias,
              bias,
              act_param.has_active,
              act_param,
              ctx);
    return;
  }
  if (M == 1 && !has_alpha && ldc == N) {
#ifdef TARGET_IOS
    float16_t* bias_ptr = new float16_t[N];
#else
    float16_t bias_ptr[N];  // NOLINT
#endif
    if (is_bias) {
      for (int i = 0; i < N; i++) {
        bias_ptr[i] = bias[0];
      }
    }
    gemv_fp16(B,
              A,
              C,
              !is_transB,
              N,
              K,
              beta,
              is_bias,
              bias_ptr,
              act_param.has_active,
              act_param,
              ctx);
#ifdef TARGET_IOS
    delete[] bias_ptr;
#endif
    return;
  }
  int hblock = get_hblock_fp16(ctx);
  int m_roundup = hblock * ((M + hblock - 1) / hblock);
  ctx->ExtendWorkspace(m_roundup * K * sizeof(float16_t));

  auto packed_A = static_cast<float16_t*>(ctx->workspace_data<float16_t>()) +
                  ctx->llc_size() / sizeof(float16_t);

  prepackA_fp16(packed_A, A, alpha, lda, 0, M, 0, K, is_transA, ctx);

  gemm_prepack_fp16(is_transB,
                    M,
                    N,
                    K,
                    packed_A,
                    B,
                    ldb,
                    beta,
                    C,
                    ldc,
                    bias,
                    is_bias,
                    act_param,
                    ctx);
}

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
