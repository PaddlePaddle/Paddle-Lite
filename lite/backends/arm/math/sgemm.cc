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

#include "lite/backends/arm/math/sgemm.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void sgemm(bool is_transA,
           bool is_transB,
           int M,
           int N,
           int K,
           float alpha,
           const float* A,
           int lda,
           const float* B,
           int ldb,
           float beta,
           float* C,
           int ldc,
           const float* bias,
           bool is_bias,
           bool is_relu,
           ARMContext* ctx) {
  auto arch = ctx->arch();
  int hblock = get_hblock(arch);
  int m_roundup = hblock * ((M + hblock - 1) / hblock);

  auto packed_A = static_cast<float*>(
      TargetMalloc(TargetType::kARM, m_roundup * K * sizeof(float)));

  prepackA(packed_A, A, alpha, lda, 0, M, 0, K, is_transA, ctx);

  sgemm_prepack(is_transB,
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
                is_relu,
                ctx);
  TargetFree(TargetType::kARM, packed_A);
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
