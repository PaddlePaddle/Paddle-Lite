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

constexpr int MBLOCK_C4 = 4;
constexpr int NBLOCK_C4 = 8;
constexpr int KBLOCK_C4 = 4;

void sgemm_prepack_c4(int M,
                      int N,
                      int K,
                      const float* A_packed,
                      const float* B,
                      float* C,
                      const float* bias,
                      bool has_bias,
                      bool has_relu,
                      ARMContext* ctx);
void sgemm_prepack_c4_a35(int M,
                          int N,
                          int K,
                          const float* A_packed,
                          const float* B,
                          float* C,
                          const float* bias,
                          bool has_bias,
                          bool has_relu,
                          ARMContext* ctx);
void sgemm_prepack_c4_small(int M,
                            int N,
                            int K,
                            const float* A_packed,
                            const float* B,
                            float* C,
                            const float* bias,
                            bool has_bias,
                            bool has_relu,
                            ARMContext* ctx);
void sgemm_prepack_c4_small_a35(int M,
                                int N,
                                int K,
                                const float* A_packed,
                                const float* B,
                                float* C,
                                const float* bias,
                                bool has_bias,
                                bool has_relu,
                                ARMContext* ctx);
void sgemm_prepack_c4_small(int M,
                            int N,
                            int K,
                            const float* A_packed,
                            const float* B,
                            float* C,
                            ARMContext* ctx);
void sgemm_prepack_c4_small_a35(int M,
                                int N,
                                int K,
                                const float* A_packed,
                                const float* B,
                                float* C,
                                ARMContext* ctx);
void sgemm_prepack_c8_int16_small(int M,
                                  int N,
                                  int K,
                                  const int16_t* A_packed,
                                  const int16_t* B,
                                  int32_t* C,
                                  ARMContext* ctx,
                                  int beta = 1);
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
