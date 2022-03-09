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
#include "lite/backends/arm/math/fp16/common_preprocess.h"
#include "lite/core/context.h"
#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
typedef __fp16 float16_t;
const int KBLOCK_FP16 = 2;
const int MBLOCK_FP16 = 8;
#ifdef __aarch64__
const int NBLOCK_FP16 = 16;
#else
const int NBLOCK_FP16 = 8;
#endif  // __aarch64__

inline int get_hblock_fp16(ARMContext* ctx) { return MBLOCK_FP16; }

void prepackA_fp16(void* out,
                   const void* in,
                   float16_t alpha,
                   int ldin,
                   int m0,
                   int mmax,
                   int k0,
                   int kmax,
                   bool is_trans,
                   ARMContext* ctx);

void prepackA_fp16(TensorLite* tout,
                   const TensorLite& tin,
                   float16_t alpha,
                   int m,
                   int k,
                   int group,
                   bool is_trans,
                   ARMContext* ctx);

void gemm_prepack_fp16(bool is_transB,
                       int M,
                       int N,
                       int K,
                       const float16_t* A_packed,
                       const float16_t* B,
                       int ldb,
                       float16_t beta,
                       float16_t* C,
                       int ldc,
                       const float16_t* bias,
                       bool has_bias,
                       const operators::ActivationParam act_param,
                       ARMContext* ctx);
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
