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

const int KBLOCK_SVE = 2;
const int MBLOCK_SVE = 8;
#ifdef ENABLE_ARM_FP16
const int NBLOCK_SVE = 16;
#else
const int NBLOCK_SVE = 12;
#endif

inline int get_hblock_sve(ARMContext* ctx, int m) {
  if (m <= 4) {
    return 4;
  } else {
    return MBLOCK_SVE;
  }
}

void prepackA_sve_fp32(void* out,
                       const void* in,
                       float alpha,
                       int ldin,
                       int m0,
                       int mmax,
                       int k0,
                       int kmax,
                       bool is_trans,
                       ARMContext* ctx);

void prepackA_sve_float(TensorLite* tout,
                        const TensorLite& tin,
                        float alpha,
                        int m,
                        int k,
                        int group,
                        bool is_trans,
                        ARMContext* ctx);

void sgemm_prepack_sve_float(bool is_transB,
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
template <typename Dtype>
void sgemm_prepack_sve(bool is_transB,
                       int M,
                       int N,
                       int K,
                       const Dtype* A_packed,
                       int lda,
                       const Dtype* B,
                       int ldb,
                       Dtype beta,
                       Dtype* C,
                       int ldc,
                       const Dtype* bias,
                       bool has_bias,
                       const operators::ActivationParam act_param,
                       ARMContext* ctx);
}  // namespace sve
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
