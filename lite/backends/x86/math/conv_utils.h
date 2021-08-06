// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <immintrin.h>
#include <vector>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

// for input and filter pack
void pack8_m256(lite::Tensor* input,
                lite::Tensor* output,
                const int channel_num,
                const bool is_filter);
void pack4_m128(lite::Tensor* input,
                lite::Tensor* output,
                const int channel_num,
                const bool is_filter);

// for output unpack
void unpack8_m256(lite::Tensor* input, lite::Tensor* output);
void unpack4_m128(lite::Tensor* input, lite::Tensor* output);

// for input padding
void padding8_m256(lite::Tensor* input,
                   lite::Tensor* output,
                   const std::vector<int>& paddings);
void padding4_m128(lite::Tensor* input,
                   lite::Tensor* output,
                   const std::vector<int>& paddings);
void padding1_float(lite::Tensor* input,
                    lite::Tensor* output,
                    const std::vector<int>& paddings);

void pack_padding8_m256(lite::Tensor* input,
                        lite::Tensor* output,
                        const int channel_num,
                        const std::vector<int>& paddings);

#define X_BLOCK_COMPUTE(llc_size, MBLOCK, NBLOCK, KBLOCK, beta)       \
  /* MBLOCK * x (result) + MBLOCK * k (A) + x * k (B) = l2*/          \
  int x_block =                                                       \
      (llc_size - (MBLOCK * K)) / (sizeof(float16_t) * (K + MBLOCK)); \
  x_block /= NBLOCK;                                                  \
  x_block = (x_block == 0) ? 1 : x_block;                             \
  x_block *= NBLOCK;                                                  \
  int x_num = (N + (x_block - 1)) / x_block;                          \
  x_block = (N + x_num - 1) / x_num;                                  \
  x_block = (x_block + NBLOCK - 1) / NBLOCK;                          \
  x_block *= NBLOCK;                                                  \
  x_block = x_block < NBLOCK ? NBLOCK : x_block;                      \
  bool flag_p_remain = false;                                         \
  int remain = 0;                                                     \
  int has_beta = fabsf(beta) > 1e-8f ? 1 : 0;

inline void act_acquire(lite_api::ActivationType act,
                        int& flag_act,       // NOLINT
                        float& local_alpha,  // NOLINT
                        float six,
                        float alpha);
template <PrecisionType Ptype>
inline void trans_gemm_weights(const Tensor& tin,
                               Tensor& tout,  // NOLINT
                               int group,
                               X86Context* ctx);
#ifdef __AVX__
static inline void transpose8_ps(__m256& row0,   // NOLINT
                                 __m256& row1,   // NOLINT
                                 __m256& row2,   // NOLINT
                                 __m256& row3,   // NOLINT
                                 __m256& row4,   // NOLINT
                                 __m256& row5,   // NOLINT
                                 __m256& row6,   // NOLINT
                                 __m256& row7);  // NOLINT

static inline void transpose4x8_ps(__m256& row0,   // NOLINT
                                   __m256& row1,   // NOLINT
                                   __m256& row2,   // NOLINT
                                   __m256& row3);  // NOLINT
#endif
template <>
inline void trans_gemm_weights<PRECISION(kFloat)>(const Tensor& tin,
                                                  Tensor& tout,  // NOLINT
                                                  int group,
                                                  X86Context* ctx) {
  CHECK_EQ(tin.dims().size(), 4) << "conv weights dims size must = 4";
  int m = tin.dims()[0] / group;
  int k = tin.dims().count(1, 4);
  int hblock = lite::x86::math::get_hblock(ctx);
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int group_size_round_up = ((m_roundup * k + 15) / 16) * 16;
  float* w_trans_ptr = nullptr;
  tout.Resize({group_size_round_up * group});
  w_trans_ptr = tout.mutable_data<float>();
  const auto* w_data = tin.data<float>();
  for (int g = 0; g < group; ++g) {
    const float* weights_group = w_data + g * m * k;
    float* weights_trans_ptr = w_trans_ptr + g * group_size_round_up;
    lite::x86::math::prepackA(
        weights_trans_ptr, weights_group, 1.f, k, 0, m, 0, k, false, ctx);
  }
}

// for activation - only support relu, relu6
__m256 activation8_m256(__m256 input, const lite_api::ActivationType act_type);
__m128 activation4_m128(__m128 input, const lite_api::ActivationType act_type);
float activation1_float(float input, const lite_api::ActivationType act_type);

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
