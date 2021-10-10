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

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

typedef __fp16 float16_t;
constexpr int MBLOCK_C8 = 8;
constexpr int NBLOCK_C8 = 8;
constexpr int KBLOCK_C8 = 8;

void gemm_prepack_c8_fp16(int M,
                          int N,
                          int K,
                          const float16_t* A_packed,
                          const float16_t* B,
                          float16_t* C,
                          const float16_t* bias,
                          ARMContext* ctx);

void gemm_prepack_c8_fp16_small(int M,
                                int N,
                                int K,
                                const float16_t* A_packed,
                                const float16_t* B,
                                float16_t* C,
                                ARMContext* ctx);

void sgemm_prepack_c8_fp16_common(int M,
                                  int N,
                                  int K,
                                  const float16_t* A_packed,
                                  const float16_t* B,
                                  float16_t* C,
                                  ARMContext* ctx);
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
