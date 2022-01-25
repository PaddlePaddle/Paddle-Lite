/* Copyright (c) 2021 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef __AVX2__

#include "lite/backends/x86/math/gemm_s8u8_compute.h"
#include <cmath>

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

template <>
void generate_gemm_s8u8_x86_kern<int8_t>::repack_bias(bool is_trans,
                                                      int M,
                                                      int K,
                                                      const float *bias,
                                                      float *out,
                                                      float *Sa,
                                                      float Sb,
                                                      float Sc,
                                                      const int8_t *A) {
  const int8_t *a_ptr = A;
  for (int i = 0; i < M; i++) {
    float bias_val = bias ? bias[i] : 0.f;
    float sum = 0.f;
    float scale = (Sa[i] * Sb) * TRANS_INT8_UINT8_OFFT;
    a_ptr = A + i * K;
    if (is_trans) {
      for (int j = 0; j < K; j++) {
        sum += A[i + j * M] * scale;
      }
    } else {
      for (int j = 0; j < K; j++) {
        sum += a_ptr[j] * scale;
      }
    }
    out[i] = bias_val - sum;
    out[i] = out[i] / Sc;
  }
}

template <>
void generate_gemm_s8u8_x86_kern<float>::repack_bias(bool is_trans,
                                                     int M,
                                                     int K,
                                                     const float *bias,
                                                     float *out,
                                                     float *Sa,
                                                     float Sb,
                                                     float Sc,
                                                     const int8_t *A) {
  const int8_t *a_ptr = A;
  for (int i = 0; i < M; i++) {
    float bias_val = bias ? bias[i] : 0.f;
    float sum = 0.f;
    float scale = (Sa[i] * Sb) * TRANS_INT8_UINT8_OFFT;
    a_ptr = A + i * K;
    if (is_trans) {
      for (int j = 0; j < K; j++) {
        sum += A[i + j * M] * scale;
      }
    } else {
      for (int j = 0; j < K; j++) {
        sum += a_ptr[j] * scale;
      }
    }
    out[i] = bias_val - sum;
  }
}

template <>
void generate_gemm_s8u8_x86_kern<int8_t>::calc_scale(
    int M, float *Sa, float Sb, float Sc, float *out) {
  for (int i = 0; i < M; i++) {
    out[i] = (Sa[i] * Sb) / Sc;
  }
}

template <>
void generate_gemm_s8u8_x86_kern<float>::calc_scale(
    int M, float *Sa, float Sb, float Sc, float *out) {
  for (int i = 0; i < M; i++) {
    out[i] = (Sa[i] * Sb);
  }
}

template <>
void generate_gemm_s8u8_x86_kern<int8_t>::calc_block(
    int M, int N, int K, int *blk_m, int *blk_n) {
  int block_size, scale_tmp;
  int block_m, block_n;

  block_m = M;
  block_n = 32 * _unroll_n;
  // C(int8) + A(int8) + B(int8) + runtime packB(uint8)
  block_size = block_m * block_n + _k_align4 * (block_m + 2 * block_n);
  scale_tmp = static_cast<int>(ceil(block_size * 1.f / _l2_size));
  scale_tmp = (scale_tmp + 1) / 2;
  scale_tmp = scale_tmp * 2;
  block_n = block_n / scale_tmp;
  block_n = block_n / _unroll_n;
  block_n = block_n * _unroll_n;
  block_n = std::max(block_n, _unroll_n);

  *blk_m = block_m;
  *blk_n = block_n;
}

template <>
void generate_gemm_s8u8_x86_kern<float>::calc_block(
    int M, int N, int K, int *blk_m, int *blk_n) {
  int block_size, scale_tmp;
  int block_m, block_n;

  block_m = M;
  block_n = 32 * _unroll_n;
  // C(int8) + A(int8) + B(int8) + runtime packB(uint8)
  block_size =
      block_m * block_n * sizeof(float) + _k_align4 * (block_m + 2 * block_n);
  scale_tmp = static_cast<int>(ceil(block_size * 1.f / _l2_size));
  scale_tmp = (scale_tmp + 1) / 2;
  scale_tmp = scale_tmp * 2;
  block_n = block_n / scale_tmp;
  block_n = block_n / _unroll_n;
  block_n = block_n * _unroll_n;
  block_n = std::max(block_n, _unroll_n);

  *blk_m = block_m;
  *blk_n = block_n;
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle

#endif  // __AVX2__
