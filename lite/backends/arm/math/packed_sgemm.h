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

#include <cmath>
#include "lite/core/context.h"
#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

#ifdef __aarch64__
constexpr int MBLOCK = 8;
constexpr int NBLOCK = 12;
constexpr int KBLOCK = 4;
inline int get_hblock(ARMContext* ctx) { return MBLOCK; }
#else
constexpr int MBLOCK_A73 = 4;
constexpr int MBLOCK_OTH = 6;
constexpr int NBLOCK = 8;
constexpr int KBLOCK = 4;
inline int get_hblock(ARMContext* ctx) {
  if (ctx->arch() == kA73) {
    return MBLOCK_A73;
  } else {
    return MBLOCK_OTH;
  }
}
#endif  // __aarch64__

void prepackA(float* out,
              const float* in,
              float alpha,
              int ldin,
              int m0,
              int mmax,
              int k0,
              int kmax,
              bool is_trans,
              ARMContext* ctx);

void prepackA(TensorLite* tout,
              const TensorLite& tin,
              float alpha,
              int m,
              int k,
              int group,
              bool is_trans,
              ARMContext* ctx);

void sgemm_prepack(bool is_transB,
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
                   ARMContext* ctx);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
