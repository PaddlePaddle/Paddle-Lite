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

#include "lite/backends/x86/math/gemm_s8u8_pack.h"
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

#ifndef _MSC_VER
typedef long long int __int64;  // NOLINT
#endif

// PrePack A
#define TRANSPOSEA_4x16                                            \
  vec_12 = _mm_unpacklo_epi8(vec_line[0], vec_line[1]);            \
  vec_23 = _mm_unpacklo_epi8(vec_line[2], vec_line[3]);            \
  vec_out = _mm_unpacklo_epi16(vec_12, vec_23);                    \
  _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr),                \
                _mm_castsi128_ps(vec_out));                        \
  _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr + K_align * 2),  \
                _mm_castsi128_ps(_mm_srli_si128(vec_out, 8)));     \
  vec_out = _mm_unpackhi_epi16(vec_12, vec_23);                    \
  _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr + K_align * 4),  \
                _mm_castsi128_ps(vec_out));                        \
  _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr + K_align * 6),  \
                _mm_castsi128_ps(_mm_srli_si128(vec_out, 8)));     \
  vec_12 = _mm_unpackhi_epi8(vec_line[0], vec_line[1]);            \
  vec_23 = _mm_unpackhi_epi8(vec_line[2], vec_line[3]);            \
  vec_out = _mm_unpacklo_epi16(vec_12, vec_23);                    \
  _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr + K_align * 8),  \
                _mm_castsi128_ps(vec_out));                        \
  _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr + K_align * 10), \
                _mm_castsi128_ps(_mm_srli_si128(vec_out, 8)));     \
  vec_out = _mm_unpackhi_epi16(vec_12, vec_23);                    \
  _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr + K_align * 12), \
                _mm_castsi128_ps(vec_out));                        \
  _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr + K_align * 14), \
                _mm_castsi128_ps(_mm_srli_si128(vec_out, 8)));

#define TRANSPOSEA_4x8                                            \
  vec_12 = _mm_unpacklo_epi8(vec_line[0], vec_line[1]);           \
  vec_23 = _mm_unpacklo_epi8(vec_line[2], vec_line[3]);           \
  vec_out = _mm_unpacklo_epi16(vec_12, vec_23);                   \
  _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr),               \
                _mm_castsi128_ps(vec_out));                       \
  _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr + K_align * 2), \
                _mm_castsi128_ps(_mm_srli_si128(vec_out, 8)));    \
  vec_out = _mm_unpackhi_epi16(vec_12, vec_23);                   \
  _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr + K_align * 4), \
                _mm_castsi128_ps(vec_out));                       \
  _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr + K_align * 6), \
                _mm_castsi128_ps(_mm_srli_si128(vec_out, 8)));

#define TRANSPOSEA_4x4                                            \
  vec_12 = _mm_unpacklo_epi8(vec_line[0], vec_line[1]);           \
  vec_23 = _mm_unpacklo_epi8(vec_line[2], vec_line[3]);           \
  vec_out = _mm_unpacklo_epi16(vec_12, vec_23);                   \
  _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr),               \
                _mm_castsi128_ps(vec_out));                       \
  _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr + K_align * 2), \
                _mm_castsi128_ps(_mm_srli_si128(vec_out, 8)));

#define TRANSPOSEA_4x2                                  \
  vec_12 = _mm_unpacklo_epi8(vec_line[0], vec_line[1]); \
  vec_23 = _mm_unpacklo_epi8(vec_line[2], vec_line[3]); \
  vec_out = _mm_unpacklo_epi16(vec_12, vec_23);         \
  _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr), _mm_castsi128_ps(vec_out));

// if K is not 4-aligned, need to pad zero
void packA_i8_notrans(int M, int K, const int8_t *AA, int8_t *pack_A) {
  int8_t *out_ptr = pack_A;
  int loop_m = 0;
  int loop_k = 0;
  int remain_k = 0;
  int8_t *A = const_cast<int8_t *>(AA);

  __m256i vec_line0, vec_line1, vec_lo, vec_hi;
  __m128i vec_line0_h, vec_line1_h, vec_lo_h, vec_hi_h;

  for (loop_m = 0; loop_m + 1 < M; loop_m += 2) {
    for (loop_k = 0; loop_k + 31 < K; loop_k += 32) {
      vec_line0 = _mm256_loadu_si256(
          reinterpret_cast<__m256i const *>(A + loop_m * K + loop_k));
      vec_line1 = _mm256_loadu_si256(
          reinterpret_cast<__m256i const *>(A + (loop_m + 1) * K + loop_k));
      vec_lo = _mm256_unpacklo_epi32(vec_line0, vec_line1);
      vec_hi = _mm256_unpackhi_epi32(vec_line0, vec_line1);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_ptr),
                          _mm256_permute2x128_si256(vec_lo, vec_hi, 0x20));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_ptr + 32),
                          _mm256_permute2x128_si256(vec_lo, vec_hi, 0x31));
      out_ptr += 2 * 32;
    }
    for (; loop_k + 15 < K; loop_k += 16) {
      vec_line0_h = _mm_loadu_si128(
          reinterpret_cast<__m128i const *>(A + loop_m * K + loop_k));
      vec_line1_h = _mm_loadu_si128(
          reinterpret_cast<__m128i const *>(A + (loop_m + 1) * K + loop_k));
      vec_lo_h = _mm_unpacklo_epi32(vec_line0_h, vec_line1_h);
      vec_hi_h = _mm_unpackhi_epi32(vec_line0_h, vec_line1_h);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(out_ptr), vec_lo_h);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(out_ptr + 16), vec_hi_h);
      out_ptr += 2 * 16;
    }
    for (; loop_k + 7 < K; loop_k += 8) {
      vec_line0_h = _mm_loadl_epi64(
          reinterpret_cast<__m128i const *>(A + loop_m * K + loop_k));
      vec_line1_h = _mm_loadl_epi64(
          reinterpret_cast<__m128i const *>(A + (loop_m + 1) * K + loop_k));
      vec_lo_h = _mm_unpacklo_epi32(vec_line0_h, vec_line1_h);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(out_ptr), vec_lo_h);
      out_ptr += 2 * 8;
    }
    for (; loop_k + 3 < K; loop_k += 4) {
      vec_line0_h =
          _mm_set1_epi32(*(reinterpret_cast<int *>(A + loop_m * K + loop_k)));
      vec_line1_h = _mm_set1_epi32(
          *(reinterpret_cast<int *>(A + (loop_m + 1) * K + loop_k)));
      vec_lo_h = _mm_unpacklo_epi32(vec_line0_h, vec_line1_h);
      _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr),
                    _mm_castsi128_ps(vec_lo_h));
      out_ptr += 2 * 4;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      vec_line0_h = _mm_setzero_si128();
      vec_line1_h = _mm_setzero_si128();
      for (int i = 0; i < remain_k; i++) {
        int8_t *tmp = reinterpret_cast<int8_t *>(&vec_line0_h);
        tmp[i] = *(A + loop_m * K + loop_k + i);
        tmp = reinterpret_cast<int8_t *>(&vec_line1_h);
        tmp[i] = *(A + (loop_m + 1) * K + loop_k + i);
      }
      vec_lo_h = _mm_unpacklo_epi32(vec_line0_h, vec_line1_h);
      _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr),
                    _mm_castsi128_ps(vec_lo_h));
      out_ptr += 2 * 4;
    }
  }
  for (; loop_m < M; loop_m++) {
    for (loop_k = 0; loop_k + 31 < K; loop_k += 32) {
      vec_line0 = _mm256_loadu_si256(
          reinterpret_cast<__m256i const *>(A + loop_m * K + loop_k));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_ptr), vec_line0);
      out_ptr += 32;
    }
    for (; loop_k + 15 < K; loop_k += 16) {
      vec_line0_h = _mm_loadu_si128(
          reinterpret_cast<__m128i const *>(A + loop_m * K + loop_k));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(out_ptr), vec_line0_h);
      out_ptr += 16;
    }
    for (; loop_k + 7 < K; loop_k += 8) {
      vec_line0_h = _mm_loadl_epi64(
          reinterpret_cast<__m128i const *>(A + loop_m * K + loop_k));
      _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr),
                    _mm_castsi128_ps(vec_line0_h));
      out_ptr += 8;
    }
    for (; loop_k + 3 < K; loop_k += 4) {
      vec_line0_h =
          _mm_set1_epi32(*(reinterpret_cast<int *>(A + loop_m * K + loop_k)));
      _mm_store_ss(reinterpret_cast<float *>(out_ptr),
                   _mm_castsi128_ps(vec_line0_h));
      out_ptr += 4;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      vec_line0_h = _mm_setzero_si128();
      for (int i = 0; i < remain_k; i++) {
        int8_t *tmp = reinterpret_cast<int8_t *>(&vec_line0_h);
        tmp[i] = *(A + loop_m * K + loop_k + i);
      }
      _mm_store_ss(reinterpret_cast<float *>(out_ptr),
                   _mm_castsi128_ps(vec_line0_h));
      out_ptr += 4;
    }
  }
}

#define ZERO_ALL                     \
  vec_line[0] = _mm_setzero_si128(); \
  vec_line[1] = _mm_setzero_si128(); \
  vec_line[2] = _mm_setzero_si128(); \
  vec_line[3] = _mm_setzero_si128();

void packA_i8_trans(int M, int K, const int8_t *AA, int8_t *pack_A) {
  int8_t *out_ptr = pack_A;
  int loop_m = 0;
  int loop_k = 0;
  int remain_k = 0;
  int K_align = 0;
  int8_t *A = const_cast<int8_t *>(AA);

  __m128i vec_12, vec_23, vec_out;
  __m128i vec_line[4];

  K_align = (K + 3) / 4;
  K_align = K_align * 4;

  for (loop_m = 0; loop_m + 15 < M; loop_m += 16) {
    for (loop_k = 0; loop_k + 3 < K; loop_k += 4) {
      vec_line[0] = _mm_loadu_si128(
          reinterpret_cast<__m128i const *>(A + loop_k * M + loop_m));
      vec_line[1] = _mm_loadu_si128(
          reinterpret_cast<__m128i const *>(A + (loop_k + 1) * M + loop_m));
      vec_line[2] = _mm_loadu_si128(
          reinterpret_cast<__m128i const *>(A + (loop_k + 2) * M + loop_m));
      vec_line[3] = _mm_loadu_si128(
          reinterpret_cast<__m128i const *>(A + (loop_k + 3) * M + loop_m));
      TRANSPOSEA_4x16 out_ptr += 2 * 4;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      ZERO_ALL
      for (int i = 0; i < remain_k; i++) {
        vec_line[i] = _mm_loadu_si128(
            reinterpret_cast<__m128i const *>(A + (loop_k + i) * M + loop_m));
      }
      TRANSPOSEA_4x16 out_ptr += 2 * 4;
    }
    out_ptr += 14 * K_align;  // total 16 * K_align
  }
  for (; loop_m + 7 < M; loop_m += 8) {
    for (loop_k = 0; loop_k + 3 < K; loop_k += 4) {
      vec_line[0] = _mm_loadl_epi64(
          reinterpret_cast<__m128i const *>(A + loop_k * M + loop_m));
      vec_line[1] = _mm_loadl_epi64(
          reinterpret_cast<__m128i const *>(A + (loop_k + 1) * M + loop_m));
      vec_line[2] = _mm_loadl_epi64(
          reinterpret_cast<__m128i const *>(A + (loop_k + 2) * M + loop_m));
      vec_line[3] = _mm_loadl_epi64(
          reinterpret_cast<__m128i const *>(A + (loop_k + 3) * M + loop_m));
      TRANSPOSEA_4x8 out_ptr += 2 * 4;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      ZERO_ALL
      for (int i = 0; i < remain_k; i++) {
        vec_line[i] = _mm_loadl_epi64(
            reinterpret_cast<__m128i const *>(A + (loop_k + i) * M + loop_m));
      }
      TRANSPOSEA_4x8 out_ptr += 2 * 4;
    }
    out_ptr += 6 * K_align;  // total 8 * K_align
  }
  for (; loop_m + 3 < M; loop_m += 4) {
    for (loop_k = 0; loop_k + 3 < K; loop_k += 4) {
      vec_line[0] =
          _mm_set1_epi32(*(reinterpret_cast<int *>(A + loop_k * M + loop_m)));
      vec_line[1] = _mm_set1_epi32(
          *(reinterpret_cast<int *>(A + (loop_k + 1) * M + loop_m)));
      vec_line[2] = _mm_set1_epi32(
          *(reinterpret_cast<int *>(A + (loop_k + 2) * M + loop_m)));
      vec_line[3] = _mm_set1_epi32(
          *(reinterpret_cast<int *>(A + (loop_k + 3) * M + loop_m)));
      TRANSPOSEA_4x4 out_ptr += 2 * 4;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      ZERO_ALL
      for (int i = 0; i < remain_k; i++) {
        vec_line[i] = _mm_set1_epi32(
            *(reinterpret_cast<int *>(A + (loop_k + i) * M + loop_m)));
      }
      TRANSPOSEA_4x4 out_ptr += 2 * 4;
    }
    out_ptr += 2 * K_align;  // total 4 * K_align
  }
  for (; loop_m + 1 < M; loop_m += 2) {
    for (loop_k = 0; loop_k + 3 < K; loop_k += 4) {
      vec_line[0] = _mm_set1_epi16(
          *(reinterpret_cast<int16_t *>(A + loop_k * M + loop_m)));
      vec_line[1] = _mm_set1_epi16(
          *(reinterpret_cast<int16_t *>(A + (loop_k + 1) * M + loop_m)));
      vec_line[2] = _mm_set1_epi16(
          *(reinterpret_cast<int16_t *>(A + (loop_k + 2) * M + loop_m)));
      vec_line[3] = _mm_set1_epi16(
          *(reinterpret_cast<int16_t *>(A + (loop_k + 3) * M + loop_m)));
      TRANSPOSEA_4x2 out_ptr += 2 * 4;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      ZERO_ALL
      for (int i = 0; i < remain_k; i++) {
        vec_line[i] = _mm_set1_epi16(
            *(reinterpret_cast<int16_t *>(A + (loop_k + i) * M + loop_m)));
      }
      TRANSPOSEA_4x2 out_ptr += 2 * 4;
    }
  }
  for (; loop_m < M; loop_m++) {
    for (loop_k = 0; loop_k + 3 < K; loop_k += 4) {
      out_ptr[0] = *(A + loop_k * M + loop_m);
      out_ptr[1] = *(A + (loop_k + 1) * M + loop_m);
      out_ptr[2] = *(A + (loop_k + 2) * M + loop_m);
      out_ptr[3] = *(A + (loop_k + 3) * M + loop_m);
      out_ptr += 4;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      ZERO_ALL
      for (int i = 0; i < remain_k; i++) {
        out_ptr[i] = *(A + (loop_k + i) * M + loop_m);
      }
      out_ptr += 4;
    }
  }
}

// runtime Pack B
/*
Attention:
1. B need to add 128 during packing, transfering from int8 to uint8.
2. B has transpose mode as well.
3. K is 4-aligned after packing.
4. don't forget to minus 128 by bias_data.
*/
// No Trans
#define INT8_ADD_128(in, vec_128_s16)                                        \
  {                                                                          \
    __m256i in_lo = _mm256_adds_epi16(                                       \
        vec_128_s16, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(in)));      \
    __m256i in_hi = _mm256_adds_epi16(                                       \
        vec_128_s16, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(in, 1))); \
    in_lo = _mm256_packus_epi16(in_lo, in_hi);                               \
    in = _mm256_permute4x64_epi64(in_lo, 216);                               \
  }

#define TRANSPOSE_4x32                                              \
  vec_l01 = _mm256_unpacklo_epi8(vec_line0, vec_line1);             \
  vec_l23 = _mm256_unpacklo_epi8(vec_line2, vec_line3);             \
  vec_h01 = _mm256_unpackhi_epi8(vec_line0, vec_line1);             \
  vec_h23 = _mm256_unpackhi_epi8(vec_line2, vec_line3);             \
  vec_l03 = _mm256_unpacklo_epi16(vec_l01, vec_l23);                \
  vec_h03 = _mm256_unpackhi_epi16(vec_l01, vec_l23);                \
  vec_l03_1 = _mm256_unpacklo_epi16(vec_h01, vec_h23);              \
  vec_h03_1 = _mm256_unpackhi_epi16(vec_h01, vec_h23);              \
  vec_out0 = _mm256_permute2x128_si256(vec_l03, vec_h03, 0x20);     \
  INT8_ADD_128(vec_out0, vec_128_s16)                               \
  vec_out1 = _mm256_permute2x128_si256(vec_l03_1, vec_h03_1, 0x20); \
  INT8_ADD_128(vec_out1, vec_128_s16)                               \
  vec_out2 = _mm256_permute2x128_si256(vec_l03, vec_h03, 0x31);     \
  INT8_ADD_128(vec_out2, vec_128_s16)                               \
  vec_out3 = _mm256_permute2x128_si256(vec_l03_1, vec_h03_1, 0x31); \
  INT8_ADD_128(vec_out3, vec_128_s16)

#define STORE_4x32                                                          \
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_ptr), vec_out0);      \
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_ptr + 32), vec_out1); \
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_ptr + 64), vec_out2); \
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_ptr + 96), vec_out3); \
  out_ptr += 32 * 4;

#define STORE_4x24                                                          \
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_ptr), vec_out0);      \
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_ptr + 32), vec_out1); \
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_ptr + 64), vec_out2); \
  out_ptr += 24 * 4;

#define STORE_4x16                                                          \
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_ptr), vec_out0);      \
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_ptr + 32), vec_out1); \
  out_ptr += 16 * 4;

#define STORE_4x8                                                      \
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_ptr), vec_out0); \
  out_ptr += 8 * 4;

#define STORE_4x4                                        \
  _mm_storeu_si128(reinterpret_cast<__m128i *>(out_ptr), \
                   _mm256_castsi256_si128(vec_out0));    \
  out_ptr += 4 * 4;

#define STORE_4x2                                                      \
  {                                                                    \
    _mm_storel_pi(reinterpret_cast<__m64 *>(out_ptr),                  \
                  _mm_castsi128_ps(_mm256_castsi256_si128(vec_out0))); \
    out_ptr += 2 * 4;                                                  \
  }

#define STORE_4x1                                                     \
  {                                                                   \
    _mm_store_ss(reinterpret_cast<float *>(out_ptr),                  \
                 _mm_castsi128_ps(_mm256_castsi256_si128(vec_out0))); \
    out_ptr += 4;                                                     \
  }

#define LOAD_32                                                             \
  vec_line0 = _mm256_loadu_si256(                                           \
      reinterpret_cast<const __m256i *>(b_ptr + loop_k * stride + loop_n)); \
  vec_line1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(         \
      b_ptr + (loop_k + 1) * stride + loop_n));                             \
  vec_line2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(         \
      b_ptr + (loop_k + 2) * stride + loop_n));                             \
  vec_line3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(         \
      b_ptr + (loop_k + 3) * stride + loop_n));

#define LOAD_EPI32(num)                                                      \
  vec_line0 = _mm256_maskload_epi32(                                         \
      reinterpret_cast<const int *>(b_ptr + loop_k * stride + loop_n),       \
      vec_mask_##num);                                                       \
  vec_line1 = _mm256_maskload_epi32(                                         \
      reinterpret_cast<const int *>(b_ptr + (loop_k + 1) * stride + loop_n), \
      vec_mask_##num);                                                       \
  vec_line2 = _mm256_maskload_epi32(                                         \
      reinterpret_cast<const int *>(b_ptr + (loop_k + 2) * stride + loop_n), \
      vec_mask_##num);                                                       \
  vec_line3 = _mm256_maskload_epi32(                                         \
      reinterpret_cast<const int *>(b_ptr + (loop_k + 3) * stride + loop_n), \
      vec_mask_##num);

#define LOAD_EPI64(num)                                                    \
  vec_line0 = _mm256_maskload_epi64(                                       \
      reinterpret_cast<const __int64 *>(b_ptr + loop_k * stride + loop_n), \
      vec_mask_##num);                                                     \
  vec_line1 =                                                              \
      _mm256_maskload_epi64(reinterpret_cast<const __int64 *>(             \
                                b_ptr + (loop_k + 1) * stride + loop_n),   \
                            vec_mask_##num);                               \
  vec_line2 =                                                              \
      _mm256_maskload_epi64(reinterpret_cast<const __int64 *>(             \
                                b_ptr + (loop_k + 2) * stride + loop_n),   \
                            vec_mask_##num);                               \
  vec_line3 =                                                              \
      _mm256_maskload_epi64(reinterpret_cast<const __int64 *>(             \
                                b_ptr + (loop_k + 3) * stride + loop_n),   \
                            vec_mask_##num);

#define LOAD_REMAIN(remain)                                             \
  switch (remain) {                                                     \
    case 1:                                                             \
      vec_line0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>( \
          b_ptr + loop_k * stride + loop_n));                           \
      vec_line1 = _mm256_setzero_si256();                               \
      vec_line2 = _mm256_setzero_si256();                               \
      vec_line3 = _mm256_setzero_si256();                               \
      break;                                                            \
    case 2:                                                             \
      vec_line0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>( \
          b_ptr + loop_k * stride + loop_n));                           \
      vec_line1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>( \
          b_ptr + (loop_k + 1) * stride + loop_n));                     \
      vec_line2 = _mm256_setzero_si256();                               \
      vec_line3 = _mm256_setzero_si256();                               \
      break;                                                            \
    case 3:                                                             \
      vec_line0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>( \
          b_ptr + loop_k * stride + loop_n));                           \
      vec_line1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>( \
          b_ptr + (loop_k + 1) * stride + loop_n));                     \
      vec_line2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>( \
          b_ptr + (loop_k + 2) * stride + loop_n));                     \
      vec_line3 = _mm256_setzero_si256();                               \
      break;                                                            \
    case 0:                                                             \
      vec_line0 = _mm256_setzero_si256();                               \
      vec_line1 = _mm256_setzero_si256();                               \
      vec_line2 = _mm256_setzero_si256();                               \
      vec_line3 = _mm256_setzero_si256();                               \
      break;                                                            \
    default:                                                            \
      break;                                                            \
  }

#define LOAD_REMAIN_EPI64(remain, num)                                         \
  switch (remain) {                                                            \
    case 1:                                                                    \
      vec_line0 = _mm256_maskload_epi64(                                       \
          reinterpret_cast<const __int64 *>(b_ptr + loop_k * stride + loop_n), \
          vec_mask_##num);                                                     \
      vec_line1 = _mm256_setzero_si256();                                      \
      vec_line2 = _mm256_setzero_si256();                                      \
      vec_line3 = _mm256_setzero_si256();                                      \
      break;                                                                   \
    case 2:                                                                    \
      vec_line0 = _mm256_maskload_epi64(                                       \
          reinterpret_cast<const __int64 *>(b_ptr + loop_k * stride + loop_n), \
          vec_mask_##num);                                                     \
      vec_line1 =                                                              \
          _mm256_maskload_epi64(reinterpret_cast<const __int64 *>(             \
                                    b_ptr + (loop_k + 1) * stride + loop_n),   \
                                vec_mask_##num);                               \
      vec_line2 = _mm256_setzero_si256();                                      \
      vec_line3 = _mm256_setzero_si256();                                      \
      break;                                                                   \
    case 3:                                                                    \
      vec_line0 = _mm256_maskload_epi64(                                       \
          reinterpret_cast<const __int64 *>(b_ptr + loop_k * stride + loop_n), \
          vec_mask_##num);                                                     \
      vec_line1 =                                                              \
          _mm256_maskload_epi64(reinterpret_cast<const __int64 *>(             \
                                    b_ptr + (loop_k + 1) * stride + loop_n),   \
                                vec_mask_##num);                               \
      vec_line2 =                                                              \
          _mm256_maskload_epi64(reinterpret_cast<const __int64 *>(             \
                                    b_ptr + (loop_k + 2) * stride + loop_n),   \
                                vec_mask_##num);                               \
      vec_line3 = _mm256_setzero_si256();                                      \
      break;                                                                   \
    default:                                                                   \
      break;                                                                   \
  }

#define LOAD_REMAIN_EPI32(remain, num)                                       \
  switch (remain) {                                                          \
    case 1:                                                                  \
      vec_line0 = _mm256_maskload_epi32(                                     \
          reinterpret_cast<const int *>(b_ptr + loop_k * stride + loop_n),   \
          vec_mask_##num);                                                   \
      vec_line1 = _mm256_setzero_si256();                                    \
      vec_line2 = _mm256_setzero_si256();                                    \
      vec_line3 = _mm256_setzero_si256();                                    \
      break;                                                                 \
    case 2:                                                                  \
      vec_line0 = _mm256_maskload_epi32(                                     \
          reinterpret_cast<const int *>(b_ptr + loop_k * stride + loop_n),   \
          vec_mask_##num);                                                   \
      vec_line1 =                                                            \
          _mm256_maskload_epi32(reinterpret_cast<const int *>(               \
                                    b_ptr + (loop_k + 1) * stride + loop_n), \
                                vec_mask_##num);                             \
      vec_line2 = _mm256_setzero_si256();                                    \
      vec_line3 = _mm256_setzero_si256();                                    \
      break;                                                                 \
    case 3:                                                                  \
      vec_line0 = _mm256_maskload_epi32(                                     \
          reinterpret_cast<const int *>(b_ptr + loop_k * stride + loop_n),   \
          vec_mask_##num);                                                   \
      vec_line1 =                                                            \
          _mm256_maskload_epi32(reinterpret_cast<const int *>(               \
                                    b_ptr + (loop_k + 1) * stride + loop_n), \
                                vec_mask_##num);                             \
      vec_line2 =                                                            \
          _mm256_maskload_epi32(reinterpret_cast<const int *>(               \
                                    b_ptr + (loop_k + 2) * stride + loop_n), \
                                vec_mask_##num);                             \
      vec_line3 = _mm256_setzero_si256();                                    \
      break;                                                                 \
    default:                                                                 \
      break;                                                                 \
  }

void packB_i82u8_notrans(
    int N, int K, int stride, const int8_t *B, uint8_t *pack_B) {
  int loop_n = 0;
  int loop_k = 0;
  int remain_k = 0;
  int k_align4 = 0;
  int8_t *b_ptr = const_cast<int8_t *>(B);
  uint8_t *out_ptr = pack_B;

  __m256i vec_line0, vec_line1, vec_line2, vec_line3;
  __m256i vec_l01, vec_l23, vec_h01, vec_h23;
  __m256i vec_l03, vec_h03, vec_l03_1, vec_h03_1;
  __m256i vec_out0, vec_out1, vec_out2, vec_out3;
  __m256i vec_128_s16 =
      _mm256_set1_epi16(static_cast<int16_t>(TRANS_INT8_UINT8_OFFT));

  // mask load, store
  __m256i vec_mask_24, vec_mask_16, vec_mask_8, vec_mask_4;
  int64_t mask0[4] = {-1, -1, -1, 0};
  int mask1[8] = {-1, 0, 0, 0, 0, 0, 0, 0};
  vec_mask_24 = _mm256_loadu_si256(reinterpret_cast<__m256i const *>(mask0));
  mask0[2] = static_cast<int64_t>(0);
  vec_mask_16 = _mm256_loadu_si256(reinterpret_cast<__m256i const *>(mask0));
  mask0[1] = static_cast<int64_t>(0);
  vec_mask_8 = _mm256_loadu_si256(reinterpret_cast<__m256i const *>(mask0));
  vec_mask_4 = _mm256_loadu_si256(reinterpret_cast<__m256i const *>(mask1));

  int8_t *vec_ptr[4];
  vec_ptr[0] = reinterpret_cast<int8_t *>(&vec_line0);
  vec_ptr[1] = reinterpret_cast<int8_t *>(&vec_line1);
  vec_ptr[2] = reinterpret_cast<int8_t *>(&vec_line2);
  vec_ptr[3] = reinterpret_cast<int8_t *>(&vec_line3);

  k_align4 = ((K + 3) / 4);
  k_align4 = k_align4 * 4;

  for (loop_n = 0; loop_n + 31 < N; loop_n += 32) {
    for (loop_k = 0; loop_k + 3 < K; loop_k += 4) {
      LOAD_32;
      TRANSPOSE_4x32;
      STORE_4x32;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      LOAD_REMAIN(remain_k);
      TRANSPOSE_4x32;
      STORE_4x32;
    }
  }
  for (; loop_n + 23 < N; loop_n += 24) {
    for (loop_k = 0; loop_k + 3 < K; loop_k += 4) {
      LOAD_EPI64(24);
      TRANSPOSE_4x32;
      STORE_4x24;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      LOAD_REMAIN_EPI64(remain_k, 24);
      TRANSPOSE_4x32;
      STORE_4x24;
    }
  }
  for (; loop_n + 15 < N; loop_n += 16) {
    for (loop_k = 0; loop_k + 3 < K; loop_k += 4) {
      LOAD_EPI64(16);
      TRANSPOSE_4x32;
      STORE_4x16;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      LOAD_REMAIN_EPI64(remain_k, 16);
      TRANSPOSE_4x32;
      STORE_4x16;
    }
  }
  for (; loop_n + 7 < N; loop_n += 8) {
    for (loop_k = 0; loop_k + 3 < K; loop_k += 4) {
      LOAD_EPI64(8);
      TRANSPOSE_4x32;
      STORE_4x8;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      LOAD_REMAIN_EPI64(remain_k, 8);
      TRANSPOSE_4x32;
      STORE_4x8;
    }
  }
  for (; loop_n + 3 < N; loop_n += 4) {
    for (loop_k = 0; loop_k + 3 < K; loop_k += 4) {
      LOAD_EPI32(4);
      TRANSPOSE_4x32;
      STORE_4x4;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      LOAD_REMAIN_EPI32(remain_k, 4);
      TRANSPOSE_4x32;
      STORE_4x4;
    }
  }
  for (; loop_n + 1 < N; loop_n += 2) {
    for (loop_k = 0; loop_k + 3 < K; loop_k += 4) {
      LOAD_REMAIN(0);
      vec_ptr[0][0] = *(b_ptr + loop_k * stride + loop_n);
      vec_ptr[0][1] = *(b_ptr + loop_k * stride + loop_n + 1);
      vec_ptr[1][0] = *(b_ptr + (loop_k + 1) * stride + loop_n);
      vec_ptr[1][1] = *(b_ptr + (loop_k + 1) * stride + loop_n + 1);
      vec_ptr[2][0] = *(b_ptr + (loop_k + 2) * stride + loop_n);
      vec_ptr[2][1] = *(b_ptr + (loop_k + 2) * stride + loop_n + 1);
      vec_ptr[3][0] = *(b_ptr + (loop_k + 3) * stride + loop_n);
      vec_ptr[3][1] = *(b_ptr + (loop_k + 3) * stride + loop_n + 1);
      TRANSPOSE_4x32;
      STORE_4x2;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      LOAD_REMAIN(0);
      for (int i = 0; i < remain_k; i++) {
        vec_ptr[i][0] = *(b_ptr + (loop_k + i) * stride + loop_n);
        vec_ptr[i][1] = *(b_ptr + (loop_k + i) * stride + loop_n + 1);
      }
      TRANSPOSE_4x32;
      STORE_4x2;
    }
  }
  for (; loop_n < N; loop_n++) {
    for (loop_k = 0; loop_k + 3 < K; loop_k += 4) {
      LOAD_REMAIN(0);
      vec_ptr[0][0] = *(b_ptr + loop_k * stride + loop_n);
      vec_ptr[1][0] = *(b_ptr + (loop_k + 1) * stride + loop_n);
      vec_ptr[2][0] = *(b_ptr + (loop_k + 2) * stride + loop_n);
      vec_ptr[3][0] = *(b_ptr + (loop_k + 3) * stride + loop_n);
      TRANSPOSE_4x32;
      STORE_4x1;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      LOAD_REMAIN(0);
      for (int i = 0; i < remain_k; i++) {
        vec_ptr[i][0] = *(b_ptr + (loop_k + i) * stride + loop_n);
      }
      TRANSPOSE_4x32;
      STORE_4x1;
    }
  }
}

// TRANS
// in0: __m128i  in1: __m256i
#define INT8_ADD_128_HALF(in, vec_128_s16)                                     \
  {                                                                            \
    __m256i in_256 = _mm256_adds_epi16(vec_128_s16, _mm256_cvtepi8_epi16(in)); \
    __m128i in_lo = _mm256_castsi256_si128(in_256);                            \
    __m128i in_hi = _mm256_extractf128_si256(in_256, 1);                       \
    in = _mm_packus_epi16(in_lo, in_hi);                                       \
  }

#define TRANSPOSE_STORE_4x16(out_offt, stride)                           \
  _mm_storeu_si128(reinterpret_cast<__m128i *>(out_ptr + (out_offt)*16), \
                   veci_line[0]);                                        \
  _mm_storeu_si128(                                                      \
      reinterpret_cast<__m128i *>(out_ptr + stride + (out_offt)*16),     \
      veci_line[1]);                                                     \
  _mm_storeu_si128(                                                      \
      reinterpret_cast<__m128i *>(out_ptr + stride * 2 + (out_offt)*16), \
      veci_line[2]);                                                     \
  _mm_storeu_si128(                                                      \
      reinterpret_cast<__m128i *>(out_ptr + stride * 3 + (out_offt)*16), \
      veci_line[3]);

#define TRANSPOSE_STORE_4x8(out_offt, stride)                            \
  _mm_storeu_si128(reinterpret_cast<__m128i *>(out_ptr + (out_offt)*16), \
                   veci_line[0]);                                        \
  _mm_storeu_si128(                                                      \
      reinterpret_cast<__m128i *>(out_ptr + stride + (out_offt)*16),     \
      veci_line[1]);

#define TRANSPOSE_STORE_2x16(out_offt)                                       \
  _mm_storel_epi64(reinterpret_cast<__m128i *>(out_ptr + (out_offt)*8),      \
                   veci_line[0]);                                            \
  _mm_storel_epi64(reinterpret_cast<__m128i *>(out_ptr + (out_offt)*8 + 8),  \
                   veci_line[1]);                                            \
  _mm_storel_epi64(reinterpret_cast<__m128i *>(out_ptr + (out_offt)*8 + 16), \
                   veci_line[2]);                                            \
  _mm_storel_epi64(reinterpret_cast<__m128i *>(out_ptr + (out_offt)*8 + 24), \
                   veci_line[3]);

#define TRANSPOSE_STORE_2x8(out_offt)                                       \
  _mm_storel_epi64(reinterpret_cast<__m128i *>(out_ptr + (out_offt)*8),     \
                   veci_line[0]);                                           \
  _mm_storel_epi64(reinterpret_cast<__m128i *>(out_ptr + (out_offt)*8 + 8), \
                   veci_line[1]);

#define TRANSPOSE_4x16(in_offt, out_offt, stride)                        \
  vec_line[0] = _mm_loadu_ps(reinterpret_cast<float const *>(            \
      b_ptr + step * ((in_offt) + 0) + loop_k));                         \
  vec_line[1] = _mm_loadu_ps(reinterpret_cast<float const *>(            \
      b_ptr + step * ((in_offt) + 1) + loop_k));                         \
  vec_line[2] = _mm_loadu_ps(reinterpret_cast<float const *>(            \
      b_ptr + step * ((in_offt) + 2) + loop_k));                         \
  vec_line[3] = _mm_loadu_ps(reinterpret_cast<float const *>(            \
      b_ptr + step * ((in_offt) + 3) + loop_k));                         \
  _MM_TRANSPOSE4_PS(vec_line[0], vec_line[1], vec_line[2], vec_line[3]); \
  veci_line[0] = _mm_castps_si128(vec_line[0]);                          \
  veci_line[1] = _mm_castps_si128(vec_line[1]);                          \
  veci_line[2] = _mm_castps_si128(vec_line[2]);                          \
  veci_line[3] = _mm_castps_si128(vec_line[3]);                          \
  INT8_ADD_128_HALF(veci_line[0], vec_128_s16)                           \
  INT8_ADD_128_HALF(veci_line[1], vec_128_s16)                           \
  INT8_ADD_128_HALF(veci_line[2], vec_128_s16)                           \
  INT8_ADD_128_HALF(veci_line[3], vec_128_s16)                           \
  TRANSPOSE_STORE_4x16(out_offt, stride)

#define TRANSPOSE_4x8(in_offt, out_offt, stride)                            \
  vec_line[0] = _mm_loadl_pi(vecf_0,                                        \
                             reinterpret_cast<__m64 const *>(               \
                                 b_ptr + step * ((in_offt) + 0) + loop_k)); \
  vec_line[1] = _mm_loadl_pi(vecf_0,                                        \
                             reinterpret_cast<__m64 const *>(               \
                                 b_ptr + step * ((in_offt) + 1) + loop_k)); \
  vec_line[2] = _mm_loadl_pi(vecf_0,                                        \
                             reinterpret_cast<__m64 const *>(               \
                                 b_ptr + step * ((in_offt) + 2) + loop_k)); \
  vec_line[3] = _mm_loadl_pi(vecf_0,                                        \
                             reinterpret_cast<__m64 const *>(               \
                                 b_ptr + step * ((in_offt) + 3) + loop_k)); \
  _MM_TRANSPOSE4_PS(vec_line[0], vec_line[1], vec_line[2], vec_line[3]);    \
  veci_line[0] = _mm_castps_si128(vec_line[0]);                             \
  veci_line[1] = _mm_castps_si128(vec_line[1]);                             \
  INT8_ADD_128_HALF(veci_line[0], vec_128_s16)                              \
  INT8_ADD_128_HALF(veci_line[1], vec_128_s16)                              \
  TRANSPOSE_STORE_4x8(out_offt, stride)

#define TRANSPOSE_4x4(in_offt, out_offt)                                     \
  vec_line[0] = _mm_castsi128_ps(_mm_set1_epi32(                             \
      *(reinterpret_cast<int *>(b_ptr + step * ((in_offt) + 0) + loop_k)))); \
  vec_line[1] = _mm_castsi128_ps(_mm_set1_epi32(                             \
      *(reinterpret_cast<int *>(b_ptr + step * ((in_offt) + 1) + loop_k)))); \
  vec_line[2] = _mm_castsi128_ps(_mm_set1_epi32(                             \
      *(reinterpret_cast<int *>(b_ptr + step * ((in_offt) + 2) + loop_k)))); \
  vec_line[3] = _mm_castsi128_ps(_mm_set1_epi32(                             \
      *(reinterpret_cast<int *>(b_ptr + step * ((in_offt) + 3) + loop_k)))); \
  _MM_TRANSPOSE4_PS(vec_line[0], vec_line[1], vec_line[2], vec_line[3]);     \
  veci_line[0] = _mm_castps_si128(vec_line[0]);                              \
  INT8_ADD_128_HALF(veci_line[0], vec_128_s16)                               \
  _mm_storeu_si128(reinterpret_cast<__m128i *>(out_ptr + (out_offt)*16),     \
                   veci_line[0]);

#define TRANSPOSE_4xX(num, in_offt, out_offt)                              \
  {                                                                        \
    vec_line[0] = _mm_set1_ps(0.f);                                        \
    vec_line[1] = _mm_set1_ps(0.f);                                        \
    vec_line[2] = _mm_set1_ps(0.f);                                        \
    vec_line[3] = _mm_set1_ps(0.f);                                        \
    int8_t *tmp0 = reinterpret_cast<int8_t *>(&vec_line[0]);               \
    int8_t *tmp1 = reinterpret_cast<int8_t *>(&vec_line[1]);               \
    int8_t *tmp2 = reinterpret_cast<int8_t *>(&vec_line[2]);               \
    int8_t *tmp3 = reinterpret_cast<int8_t *>(&vec_line[3]);               \
    for (int i = 0; i < num; i++) {                                        \
      tmp0[i] = *(b_ptr + step * ((in_offt) + 0) + loop_k + i);            \
      tmp1[i] = *(b_ptr + step * ((in_offt) + 1) + loop_k + i);            \
      tmp2[i] = *(b_ptr + step * ((in_offt) + 2) + loop_k + i);            \
      tmp3[i] = *(b_ptr + step * ((in_offt) + 3) + loop_k + i);            \
    }                                                                      \
    _MM_TRANSPOSE4_PS(vec_line[0], vec_line[1], vec_line[2], vec_line[3]); \
    veci_line[0] = _mm_castps_si128(vec_line[0]);                          \
    INT8_ADD_128_HALF(veci_line[0], vec_128_s16)                           \
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out_ptr + (out_offt)*16), \
                     veci_line[0]);                                        \
  }

#define TRANSPOSE_2x16(in_offt, out_offt)                                \
  vec_line[0] = _mm_loadu_ps(reinterpret_cast<float const *>(            \
      b_ptr + step * ((in_offt) + 0) + loop_k));                         \
  vec_line[1] = _mm_loadu_ps(reinterpret_cast<float const *>(            \
      b_ptr + step * ((in_offt) + 1) + loop_k));                         \
  _MM_TRANSPOSE4_PS(vec_line[0], vec_line[1], vec_line[2], vec_line[3]); \
  veci_line[0] = _mm_castps_si128(vec_line[0]);                          \
  veci_line[1] = _mm_castps_si128(vec_line[1]);                          \
  veci_line[2] = _mm_castps_si128(vec_line[2]);                          \
  veci_line[3] = _mm_castps_si128(vec_line[3]);                          \
  INT8_ADD_128_HALF(veci_line[0], vec_128_s16)                           \
  INT8_ADD_128_HALF(veci_line[1], vec_128_s16)                           \
  INT8_ADD_128_HALF(veci_line[2], vec_128_s16)                           \
  INT8_ADD_128_HALF(veci_line[3], vec_128_s16)                           \
  TRANSPOSE_STORE_2x16(out_offt)

#define TRANSPOSE_2x8(in_offt, out_offt)                                    \
  vec_line[0] = _mm_loadl_pi(vecf_0,                                        \
                             reinterpret_cast<__m64 const *>(               \
                                 b_ptr + step * ((in_offt) + 0) + loop_k)); \
  vec_line[1] = _mm_loadl_pi(vecf_0,                                        \
                             reinterpret_cast<__m64 const *>(               \
                                 b_ptr + step * ((in_offt) + 1) + loop_k)); \
  _MM_TRANSPOSE4_PS(vec_line[0], vec_line[1], vec_line[2], vec_line[3]);    \
  veci_line[0] = _mm_castps_si128(vec_line[0]);                             \
  veci_line[1] = _mm_castps_si128(vec_line[1]);                             \
  INT8_ADD_128_HALF(veci_line[0], vec_128_s16)                              \
  INT8_ADD_128_HALF(veci_line[1], vec_128_s16)                              \
  TRANSPOSE_STORE_2x8(out_offt)

#define TRANSPOSE_2x4(in_offt, out_offt)                                     \
  vec_line[0] = _mm_castsi128_ps(_mm_set1_epi32(                             \
      *(reinterpret_cast<int *>(b_ptr + step * ((in_offt) + 0) + loop_k)))); \
  vec_line[1] = _mm_castsi128_ps(_mm_set1_epi32(                             \
      *(reinterpret_cast<int *>(b_ptr + step * ((in_offt) + 1) + loop_k)))); \
  _MM_TRANSPOSE4_PS(vec_line[0], vec_line[1], vec_line[2], vec_line[3]);     \
  veci_line[0] = _mm_castps_si128(vec_line[0]);                              \
  INT8_ADD_128_HALF(veci_line[0], vec_128_s16)                               \
  _mm_storel_epi64(reinterpret_cast<__m128i *>(out_ptr + (out_offt)*8),      \
                   veci_line[0]);

#define TRANSPOSE_2xX(num, in_offt, out_offt)                              \
  {                                                                        \
    vec_line[0] = _mm_set1_ps(0.f);                                        \
    vec_line[1] = _mm_set1_ps(0.f);                                        \
    int8_t *tmp0 = reinterpret_cast<int8_t *>(&vec_line[0]);               \
    int8_t *tmp1 = reinterpret_cast<int8_t *>(&vec_line[1]);               \
    for (int i = 0; i < num; i++) {                                        \
      tmp0[i] = *(b_ptr + step * ((in_offt) + 0) + loop_k + i);            \
      tmp1[i] = *(b_ptr + step * ((in_offt) + 1) + loop_k + i);            \
    }                                                                      \
    _MM_TRANSPOSE4_PS(vec_line[0], vec_line[1], vec_line[2], vec_line[3]); \
    veci_line[0] = _mm_castps_si128(vec_line[0]);                          \
    INT8_ADD_128_HALF(veci_line[0], vec_128_s16)                           \
    _mm_storel_epi64(reinterpret_cast<__m128i *>(out_ptr + (out_offt)*8),  \
                     veci_line[0]);                                        \
  }

void packB_i82u8_trans(
    int N, int K, int step, const int8_t *B, uint8_t *pack_B) {
  int loop_n = 0, loop_k = 0;
  int remain_k = 0;
  int8_t *b_ptr = const_cast<int8_t *>(B);
  uint8_t *out_ptr = pack_B;
  int k_align4 = ((K + 3) / 4);
  k_align4 = k_align4 * 4;

  __m128 vec_line[4] = {0};
  __m128i veci_line[4] = {0};
  __m128 vecf_0 = _mm_set1_ps(0.f);
  __m256i vec_128_s16 =
      _mm256_set1_epi16(static_cast<int16_t>(TRANS_INT8_UINT8_OFFT));

  for (loop_n = 0; loop_n + 31 < N; loop_n += 32) {
    for (loop_k = 0; loop_k + 15 < K; loop_k += 16) {
      TRANSPOSE_4x16(loop_n, 0, 128);
      TRANSPOSE_4x16((loop_n + 4), 1, 128);
      TRANSPOSE_4x16((loop_n + 8), 2, 128);
      TRANSPOSE_4x16((loop_n + 12), 3, 128);
      TRANSPOSE_4x16((loop_n + 16), 4, 128);
      TRANSPOSE_4x16((loop_n + 20), 5, 128);
      TRANSPOSE_4x16((loop_n + 24), 6, 128);
      TRANSPOSE_4x16((loop_n + 28), 7, 128);
      out_ptr += 32 * 16;
    }
    for (; loop_k + 7 < K; loop_k += 8) {
      TRANSPOSE_4x8(loop_n, 0, 128);
      TRANSPOSE_4x8((loop_n + 4), 1, 128);
      TRANSPOSE_4x8((loop_n + 8), 2, 128);
      TRANSPOSE_4x8((loop_n + 12), 3, 128);
      TRANSPOSE_4x8((loop_n + 16), 4, 128);
      TRANSPOSE_4x8((loop_n + 20), 5, 128);
      TRANSPOSE_4x8((loop_n + 24), 6, 128);
      TRANSPOSE_4x8((loop_n + 28), 7, 128);
      out_ptr += 32 * 8;
    }
    for (; loop_k + 3 < K; loop_k += 4) {
      TRANSPOSE_4x4(loop_n, 0);
      TRANSPOSE_4x4((loop_n + 4), 1);
      TRANSPOSE_4x4((loop_n + 8), 2);
      TRANSPOSE_4x4((loop_n + 12), 3);
      TRANSPOSE_4x4((loop_n + 16), 4);
      TRANSPOSE_4x4((loop_n + 20), 5);
      TRANSPOSE_4x4((loop_n + 24), 6);
      TRANSPOSE_4x4((loop_n + 28), 7);
      out_ptr += 32 * 4;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      TRANSPOSE_4xX(remain_k, loop_n, 0);
      TRANSPOSE_4xX(remain_k, (loop_n + 4), 1);
      TRANSPOSE_4xX(remain_k, (loop_n + 8), 2);
      TRANSPOSE_4xX(remain_k, (loop_n + 12), 3);
      TRANSPOSE_4xX(remain_k, (loop_n + 16), 4);
      TRANSPOSE_4xX(remain_k, (loop_n + 20), 5);
      TRANSPOSE_4xX(remain_k, (loop_n + 24), 6);
      TRANSPOSE_4xX(remain_k, (loop_n + 28), 7);
      out_ptr += 32 * 4;
    }
  }
  for (; loop_n + 23 < N; loop_n += 24) {
    for (loop_k = 0; loop_k + 15 < K; loop_k += 16) {
      TRANSPOSE_4x16(loop_n, 0, 96);
      TRANSPOSE_4x16((loop_n + 4), 1, 96);
      TRANSPOSE_4x16((loop_n + 8), 2, 96);
      TRANSPOSE_4x16((loop_n + 12), 3, 96);
      TRANSPOSE_4x16((loop_n + 16), 4, 96);
      TRANSPOSE_4x16((loop_n + 20), 5, 96);
      out_ptr += 24 * 16;
    }
    for (; loop_k + 7 < K; loop_k += 8) {
      TRANSPOSE_4x8(loop_n, 0, 96);
      TRANSPOSE_4x8((loop_n + 4), 1, 96);
      TRANSPOSE_4x8((loop_n + 8), 2, 96);
      TRANSPOSE_4x8((loop_n + 12), 3, 96);
      TRANSPOSE_4x8((loop_n + 16), 4, 96);
      TRANSPOSE_4x8((loop_n + 20), 5, 96);
      out_ptr += 24 * 8;
    }
    for (; loop_k + 3 < K; loop_k += 4) {
      TRANSPOSE_4x4(loop_n, 0);
      TRANSPOSE_4x4((loop_n + 4), 1);
      TRANSPOSE_4x4((loop_n + 8), 2);
      TRANSPOSE_4x4((loop_n + 12), 3);
      TRANSPOSE_4x4((loop_n + 16), 4);
      TRANSPOSE_4x4((loop_n + 20), 5);
      out_ptr += 24 * 4;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      TRANSPOSE_4xX(remain_k, loop_n, 0);
      TRANSPOSE_4xX(remain_k, (loop_n + 4), 1);
      TRANSPOSE_4xX(remain_k, (loop_n + 8), 2);
      TRANSPOSE_4xX(remain_k, (loop_n + 12), 3);
      TRANSPOSE_4xX(remain_k, (loop_n + 16), 4);
      TRANSPOSE_4xX(remain_k, (loop_n + 20), 5);
      out_ptr += 24 * 4;
    }
  }
  for (; loop_n + 15 < N; loop_n += 16) {
    for (loop_k = 0; loop_k + 15 < K; loop_k += 16) {
      TRANSPOSE_4x16(loop_n, 0, 64);
      TRANSPOSE_4x16((loop_n + 4), 1, 64);
      TRANSPOSE_4x16((loop_n + 8), 2, 64);
      TRANSPOSE_4x16((loop_n + 12), 3, 64);
      out_ptr += 16 * 16;
    }
    for (; loop_k + 7 < K; loop_k += 8) {
      TRANSPOSE_4x8(loop_n, 0, 64);
      TRANSPOSE_4x8((loop_n + 4), 1, 64);
      TRANSPOSE_4x8((loop_n + 8), 2, 64);
      TRANSPOSE_4x8((loop_n + 12), 3, 64);
      out_ptr += 16 * 8;
    }
    for (; loop_k + 3 < K; loop_k += 4) {
      TRANSPOSE_4x4(loop_n, 0);
      TRANSPOSE_4x4((loop_n + 4), 1);
      TRANSPOSE_4x4((loop_n + 8), 2);
      TRANSPOSE_4x4((loop_n + 12), 3);
      out_ptr += 16 * 4;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      TRANSPOSE_4xX(remain_k, loop_n, 0);
      TRANSPOSE_4xX(remain_k, (loop_n + 4), 1);
      TRANSPOSE_4xX(remain_k, (loop_n + 8), 2);
      TRANSPOSE_4xX(remain_k, (loop_n + 12), 3);
      out_ptr += 16 * 4;
    }
  }
  for (; loop_n + 7 < N; loop_n += 8) {
    for (loop_k = 0; loop_k + 15 < K; loop_k += 16) {
      TRANSPOSE_4x16(loop_n, 0, 32);
      TRANSPOSE_4x16((loop_n + 4), 1, 32);
      out_ptr += 8 * 16;
    }
    for (; loop_k + 7 < K; loop_k += 8) {
      TRANSPOSE_4x8(loop_n, 0, 32);
      TRANSPOSE_4x8((loop_n + 4), 1, 32);
      out_ptr += 8 * 8;
    }
    for (; loop_k + 3 < K; loop_k += 4) {
      TRANSPOSE_4x4(loop_n, 0);
      TRANSPOSE_4x4((loop_n + 4), 1);
      out_ptr += 8 * 4;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      TRANSPOSE_4xX(remain_k, loop_n, 0);
      TRANSPOSE_4xX(remain_k, (loop_n + 4), 1);
      out_ptr += 8 * 4;
    }
  }
  for (; loop_n + 3 < N; loop_n += 4) {
    for (loop_k = 0; loop_k + 15 < K; loop_k += 16) {
      TRANSPOSE_4x16(loop_n, 0, 16);
      out_ptr += 4 * 16;
    }
    for (; loop_k + 7 < K; loop_k += 8) {
      TRANSPOSE_4x8(loop_n, 0, 16);
      out_ptr += 4 * 8;
    }
    for (; loop_k + 3 < K; loop_k += 4) {
      TRANSPOSE_4x4(loop_n, 0);
      out_ptr += 4 * 4;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      TRANSPOSE_4xX(remain_k, loop_n, 0);
      out_ptr += 4 * 4;
    }
  }
  for (; loop_n + 1 < N; loop_n += 2) {
    for (loop_k = 0; loop_k + 15 < K; loop_k += 16) {
      TRANSPOSE_2x16(loop_n, 0);
      out_ptr += 2 * 16;
    }
    for (; loop_k + 7 < K; loop_k += 8) {
      TRANSPOSE_2x8(loop_n, 0);
      out_ptr += 2 * 8;
    }
    for (; loop_k + 3 < K; loop_k += 4) {
      TRANSPOSE_2x4(loop_n, 0);
      out_ptr += 2 * 4;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      TRANSPOSE_2xX(remain_k, loop_n, 0);
      out_ptr += 2 * 4;
    }
  }
  for (; loop_n < N; loop_n++) {
    for (loop_k = 0; loop_k + 15 < K; loop_k += 16) {
      veci_line[0] = _mm_loadu_si128(
          reinterpret_cast<__m128i const *>(b_ptr + step * loop_n + loop_k));
      INT8_ADD_128_HALF(veci_line[0], vec_128_s16)
      _mm_storeu_si128(reinterpret_cast<__m128i *>(out_ptr), veci_line[0]);
      out_ptr += 1 * 16;
    }
    for (; loop_k + 7 < K; loop_k += 8) {
      veci_line[0] = _mm_set1_epi64x(
          *(reinterpret_cast<__int64 *>(b_ptr + step * loop_n + loop_k)));
      INT8_ADD_128_HALF(veci_line[0], vec_128_s16)
      _mm_storel_epi64(reinterpret_cast<__m128i *>(out_ptr), veci_line[0]);
      out_ptr += 1 * 8;
    }
    for (; loop_k + 3 < K; loop_k += 4) {
      veci_line[0] = _mm_set1_epi32(
          *(reinterpret_cast<int *>(b_ptr + step * loop_n + loop_k)));
      INT8_ADD_128_HALF(veci_line[0], vec_128_s16)
      _mm_store_ss(reinterpret_cast<float *>(out_ptr),
                   _mm_castsi128_ps(veci_line[0]));
      out_ptr += 1 * 4;
    }
    remain_k = K - loop_k;
    if (remain_k > 0) {
      veci_line[0] = _mm_set1_epi32(0);
      int8_t *vec_tmp = reinterpret_cast<int8_t *>(&veci_line[0]);
      for (int i = 0; i < remain_k; i++) {
        vec_tmp[i] = *(b_ptr + step * loop_n + loop_k + i);
      }
      INT8_ADD_128_HALF(veci_line[0], vec_128_s16)
      _mm_store_ss(reinterpret_cast<float *>(out_ptr),
                   _mm_castsi128_ps(veci_line[0]));
      out_ptr += 1 * 4;
    }
  }
}

// PackA 's K dim need 4-aligned,
// so it needs M * K_4aligned Bytes.
void gemm_s8u8s8_prepackA(
    int M, int K, const int8_t *A, int8_t *pack_A, bool is_trans) {
  if (is_trans) {
    packA_i8_trans(M, K, A, pack_A);
  } else {
    packA_i8_notrans(M, K, A, pack_A);
  }
}

void gemm_s8u8s8_runpackB(
    int N, int K, int stride, const int8_t *B, uint8_t *pack_B, bool is_trans) {
  if (is_trans) {
    packB_i82u8_trans(N, K, stride, B, pack_B);
  } else {
    packB_i82u8_notrans(N, K, stride, B, pack_B);
  }
}

#undef TRANSPOSE_4x32
#undef TRANSPOSEA_4x16
#undef TRANSPOSEA_4x8
#undef TRANSPOSEA_4x4
#undef TRANSPOSEA_4x2
#undef ZERO_ALL
#undef INT8_ADD_128
#undef STORE_4x32
#undef STORE_4x24
#undef STORE_4x16
#undef STORE_4x8
#undef STORE_4x4
#undef STORE_4x2
#undef STORE_4x1
#undef LOAD_32
#undef LOAD_EPI32
#undef LOAD_EPI64
#undef LOAD_REMAIN
#undef LOAD_REMAIN_EPI64
#undef LOAD_REMAIN_EPI32
#undef INT8_ADD_128_HALF
#undef TRANSPOSE_STORE_4x16
#undef TRANSPOSE_STORE_4x8
#undef TRANSPOSE_STORE_2x16
#undef TRANSPOSE_STORE_2x8
#undef TRANSPOSE_4x16
#undef TRANSPOSE_4x8
#undef TRANSPOSE_4x4
#undef TRANSPOSE_4xX
#undef TRANSPOSE_2x16
#undef TRANSPOSE_2x8
#undef TRANSPOSE_2x4
#undef TRANSPOSE_2xX

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle

#endif  // __AVX2__
