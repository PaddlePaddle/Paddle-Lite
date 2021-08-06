// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <cmath>
#include "lite/core/context.h"
#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

constexpr int KBLOCK = 4;
#ifdef __AVX__
constexpr int MBLOCK = 6;
constexpr int NBLOCK = 16;
#else
constexpr int MBLOCK = 4;
constexpr int NBLOCK = 4;
#endif  // __aarch64__
inline int get_hblock(X86Context* ctx) { return MBLOCK; }

void prepackA(float* out,
              const float* in,
              float alpha,
              int ldin,
              int m0,
              int mmax,
              int k0,
              int kmax,
              bool is_trans,
              X86Context* ctx);

/**
 * \brief input data is not transpose
 * for SSE, transform data to block x k x 4 layout
 * for AVX, transform data to block x k x 6 layout
 */
void prepackA(TensorLite* tout,
              const TensorLite& tin,
              float alpha,
              int m,
              int k,
              int group,
              bool is_trans,
              X86Context* ctx);

/**
 * \brief GEMM compute A=M*K B=K*N C=M*N
 * for SSE, compute unit is 4 x 4 + 4 x 1
 * for AVX, compute unit is 6 x 16 + 6 x 8 + 6 x 4 + 6 x 1
 */
void gemm_prepack(bool is_transB,
                  int M,
                  int N,
                  int K,
                  const float* A_packed,
                  const float* B,
                  int ldb,
                  float beta,
                  float* C,
                  int ldc,
                  const float* bias,
                  bool has_bias,
                  const operators::ActivationParam act_param,
                  X86Context* ctx);

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
