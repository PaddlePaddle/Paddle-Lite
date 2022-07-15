// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
namespace sve {
const int KBLOCK_INT8_SVE = 2;
const int MBLOCK_INT8_SVE = 8;
const int NBLOCK_INT8_SVE = 12;

#define ROUNDUP_SVE(a, b) ((((a) + (b)-1) / (b)) * (b))
inline int get_hblock_int8_sve(ARMContext* ctx) { return MBLOCK_INT8_SVE; }

void prepackA_int8_sve(int8_t* out,
                       const int8_t* in,
                       int ldin,
                       int m0,
                       int mmax,
                       int k0,
                       int kmax,
                       bool is_trans,
                       ARMContext* ctx);

void prepackA_int8_sve(TensorLite* tout,
                       const TensorLite& tin,
                       int m,
                       int k,
                       int group,
                       bool is_trans,
                       ARMContext* ctx);

template <typename dtype>
void gemm_prepack_int8_sve(const int8_t* A_packed,
                           const int8_t* B,
                           const float* bias,
                           dtype* C,
                           int M,
                           int N,
                           int K,
                           bool is_bias,
                           bool is_transB,
                           const float* scale,
                           const operators::ActivationParam act_param,
                           ARMContext* ctx);

}  // namespace sve
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
