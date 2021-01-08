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

const int KBLOCK_FP16 = 4;
#ifdef __aarch64__
// for int7/int8 gemm
const int MBLOCK_INT8_FP16 = 8;
const int NBLOCK_INT8_FP16 = 12;
#else
const int MBLOCK_INT8_FP16 = 6;
const int NBLOCK_INT8_FP16 = 8;
#endif  // __aarch64__

inline int get_hblock_int8(ARMContext* ctx) { return MBLOCK_INT8_FP16 }

void prepackA_fp16(void* out,
                   const void* in,
                   int ldin,
                   int m0,
                   int mmax,
                   int k0,
                   int kmax,
                   bool is_trans,
                   ARMContext* ctx);

void prepackA_fp16(TensorLite* tout,
                   const TensorLite& tin,
                   int m,
                   int k,
                   int group,
                   bool is_trans,
                   ARMContext* ctx);

template <typename dtype>
void gemm_prepack_fp16(const __fp16* A_packed,
                       const __fp16* B,
                       const __fp16* bias,
                       dtype* C,
                       int M,
                       int N,
                       int K,
                       bool is_bias,
                       bool is_transB,
                       const float* scale,
                       const operators::ActivationParam act_param,
                       ARMContext* ctx);

#define ROUNDUP(a, b) ((((a) + (b)-1) / (b)) * (b))
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
