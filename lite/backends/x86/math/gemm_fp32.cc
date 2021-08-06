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

#include "lite/backends/x86/math/gemm_fp32.h"
#include <arm_neon.h>
#include "lite/backends/x86/math/conv_utils.h"
namespace paddle {
namespace lite {
namespace x86 {
namespace math {
#ifdef __AVX__
void gemm_prepack_6x16(bool is_transB,
                       int M,
                       int N,
                       int K,
                       const float *A_packed,
                       const float *B,
                       int ldb,
                       float beta,
                       float *C,
                       int ldc,
                       const float *bias,
                       bool has_bias,
                       const operators::ActivationParam act_param,
                       ARMContext *ctx);

void prepackA_6x4(float *out,
                  const float *in,
                  float alpha,
                  int ldin,
                  int m0,
                  int mmax,
                  int k0,
                  int kmax);

void prepackA_trans_6x4(float *out,
                        const float *in,
                        float alpha,
                        int ldin,
                        int m0,
                        int mmax,
                        int k0,
                        int kmax);
#else
void gemm_prepack_4x4(bool is_transB,
                      int M,
                      int N,
                      int K,
                      const float *A_packed,
                      const float *B,
                      int ldb,
                      float beta,
                      float *C,
                      int ldc,
                      const float *bias,
                      bool has_bias,
                      const operators::ActivationParam act_param,
                      ARMContext *ctx);

void prepackA_4x4(float *out,
                  const float *in,
                  float alpha,
                  int ldin,
                  int m0,
                  int mmax,
                  int k0,
                  int kmax);

void prepackA_trans_4x4(float *out,
                        const float *in,
                        float alpha,
                        int ldin,
                        int m0,
                        int mmax,
                        int k0,
                        int kmax);
#endif
/**
 * \brief input data is not transpose
 * for SSE, transform data to block x k x 4 layout
 * for AVX, transform data to block x k x 6 layout
 */
void prepackA(void *out,
              const void *in,
              float alpha,
              const int ldin,
              const int m0,
              const int mmax,
              const int k0,
              const int kmax,
              bool is_trans,
              ARMContext *ctx) {
#define PREPACKA_PARAMS                                                       \
  static_cast<float *>(out), static_cast<const float *>(in), alpha, ldin, m0, \
      mmax, k0, kmax
#ifdef __AVX__
  if (is_trans) {
    prepackA_trans_6x4(PREPACKA_PARAMS);
  } else {
    prepackA_6x4(PREPACKA_PARAMS);
  }
#else
  if (is_trans) {
    prepackA_trans_4x4(PREPACKA_PARAMS);
  } else {
    prepackA_4x4(PREPACKA_PARAMS);
  }
#endif
}

void prepackA(TensorLite *tout,
              const TensorLite &tin,
              float alpha,
              int m,
              int k,
              int group,
              bool is_trans,
              ARMContext *ctx) {
  int hblock = get_hblock(ctx);
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int group_size_round_up = ((m_roundup * k + 15) / 16) * 16;
  if (tout->numel() < group_size_round_up * group) {
    tout->Resize({group_size_round_up * group});
  }
  int lda = k;
  if (is_trans) {
    lda = m;
  }
  for (int g = 0; g < group; ++g) {
    const float *weights_group = tin.data<float>() + g * m * k;
    float *weights_trans_ptr =
        tout->mutable_data<float>() + g * group_size_round_up;
    prepackA(weights_trans_ptr,
             weights_group,
             alpha,
             lda,
             0,
             m,
             0,
             k,
             is_trans,
             ctx);
  }
}

/**
 * \brief GEMM compute A=M*K B=K*N C=M*N
 * for SSE, compute unit is 4 x 4 + 4 x 1
 * for AVX, compute unit is 6 x 16 + 6 x 8 + 6 x 4 + 6 x 1
 */
void gemm_prepack(bool is_transB,
                  int M,
                  int N,
                  int K,
                  const float *A_packed,
                  const float *B,
                  int ldb,
                  float beta,
                  float *C,
                  int ldc,
                  const float *bias,
                  bool has_bias,
                  const operators::ActivationParam act_param,
                  ARMContext *ctx) {
#ifdef __AVX__
  gemm_prepack_6x16(is_transB,
                    M,
                    N,
                    K,
                    A_packed,
                    B,
                    ldb,
                    beta,
                    C,
                    ldc,
                    bias,
                    has_bias,
                    act_param,
                    ctx);
#else   // SSE
  gemm_prepack_4x4(is_transB,
                   M,
                   N,
                   K,
                   A_packed,
                   B,
                   ldb,
                   beta,
                   C,
                   ldc,
                   bias,
                   has_bias,
                   act_param,
                   ctx);
#endif  // AVX
}
#ifdef __AVX__
/**
 * \brief do C6 Trans
 * in: m * k
 * out: m / 6 * k * 6
 * if has_alpha then in = in * alpha
*/
void prepackA_6x4(float *out,
                  const float *in,
                  float alpha,
                  const int ldin,
                  const int m0,
                  const int mmax,
                  const int k0,
                  const int kmax) {
  int x_len = kmax - k0;
  float zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(float) * x_len);

  float *dout = out;
  const float *inptr = in;
  bool has_alpha = fabsf(alpha - 1.f) > 1e-8f;

  int cnt = x_len / 8;
  int remain = x_len & 7;
  int cnt_4 = remain >> 2;
  remain = remain & 3;
  __m256 valpha = _mm256_set1_ps(alpha);
  __m128 valpha_128 = _mm_set1_ps(alpha);
#pragma omp parallel for
  for (int y = m0; y < mmax; y += 6) {
    float *outptr = dout + (y - m0) * x_len;
    const float *inptr0 = inptr + y * ldin + k0;
    const float *inptr1 = inptr0 + ldin;
    const float *inptr2 = inptr1 + ldin;
    const float *inptr3 = inptr2 + ldin;
    const float *inptr4 = inptr3 + ldin;
    const float *inptr5 = inptr4 + ldin;
    if ((y + 5) >= mmax) {
      switch (y + 5 - mmax) {
        case 4:
          inptr1 = zerobuff;
        case 3:
          inptr2 = zerobuff;
        case 2:
          inptr3 = zerobuff;
        case 1:
          inptr4 = zerobuff;
        case 3:
          inptr5 = zerobuff;
        default:
          break;
      }
    }
    for (int x = 0; x < cnt; x++) {
      __m256 vin0 = _mm256_loadu_ps(inptr0);
      __m256 vin1 = _mm256_loadu_ps(inptr1);
      __m256 vin2 = _mm256_loadu_ps(inptr2);
      __m256 vin3 = _mm256_loadu_ps(inptr3);
      __m256 vin4 = _mm256_loadu_ps(inptr4);
      __m256 vin5 = _mm256_loadu_ps(inptr5);
      if (has_alpha) {
        vin0 = _mm256_mul_ps(vin0, valpha);
        vin1 = _mm256_mul_ps(vin1, valpha);
        vin2 = _mm256_mul_ps(vin2, valpha);
        vin3 = _mm256_mul_ps(vin3, valpha);
        vin4 = _mm256_mul_ps(vin4, valpha);
        vin5 = _mm256_mul_ps(vin5, valpha);
      }
      // vtmp0=a0b0a2b2a4b4a6b6
      __m256 vtmp0 = _mm256_unpacklo_ps(vin0, vin1);
      // vtmp2=c0d0c2d2c4d4c6d6
      __m256 vtmp2 = _mm256_unpacklo_ps(vin2, vin3);
      // vtmp3=e0f0e2f2
      __m256 vtmp4 = _mm256_unpacklo_ps(vin4, vin5);
      // vtmp1=a1b1a3b3a5b5a7b7
      __m256 vtmp1 = _mm256_unpackhi_ps(vin0, vin1);
      __m256 vtmp3 = _mm256_unpackhi_ps(vin2, vin3);
      __m256 vtmp5 = _mm256_unpackhi_ps(vin4, vin5);
      // vtran0 = a0b0c0d0a4b4c4d4
      __m256 vtran0 =
          _mm256_shuffle_ps(vtmp0, vtmp2, 0xaa);  // 0xaa=[01,00,01,00]
      // vtran1 = a2b2c2d2a6b6c6d6
      __m256 vtran1 =
          _mm256_shuffle_ps(vtmp0, vtmp2, 0xdd);  // 0xaa=[11,10,11,10]
      // vtran2 = e0f0a1b1e2f2a3b3
      __m256 vtran2 =
          _mm256_shuffle_ps(vtmp4, vtmp1, 0xaa);  // 0xaa=[01,00,01,00]
      // vtran3 = e4f4a5b5e6f6a7b7
      __m256 vtran3 =
          _mm256_shuffle_ps(vtmp4, vtmp1, 0xdd);  // 0xaa=[11,10,11,10]
      // vtran4 = c1d1e1d1c3d3e3f3
      __m256 vtran4 =
          _mm256_shuffle_ps(vtmp3, vtmp5, 0xaa);  // 0xaa=[01,00,01,00]
      // vtran5 = c5d5e5d5c7d7e7f7
      __m256 vtran5 =
          _mm256_shuffle_ps(vtmp3, vtmp5, 0xdd);  // 0xaa=[11,10,11,10]
      // swap 128
      // vin0 =  e0f0a1b1c1d1e1d1
      vin0 = _mm256_permute2f128_ps(vtran2, vtran4, 0x20);
      // vin1 = e2f2a3b3c3d3e3f3
      vin1 = _mm256_permute2f128_ps(vtran2, vtran4, 0x31);
      // vin2 =  e4f4a5b5c5d5e5d5
      vin2 = _mm256_permute2f128_ps(vtran3, vtran5, 0x20);
      // vin3 = e6f6a7b7c7d7e7f7
      vin3 = _mm256_permute2f128_ps(vtran3, vtran5, 0x31);
      // vin4 = a0b0c0d0e0f0a1b1
      vin4 = _mm256_permute2f128_ps(vtran0, vin0, 0x20);
      // vin5 = c1d1e1d1a2b2c2d2
      vin5 = _mm256_permute2f128_ps(vin0, vtran1, 0x21);
      // vtmp0 = a4b4c4d4e4f4a5b5
      vtmp0 = _mm256_permute2f128_ps(vtran0, vtran3, 0x21);
      // vtmp1 = c5d5e5f5a6b6c6d6
      vtmp1 = _mm256_permute2f128_ps(vin2, vtran1, 0x31);

      _mm256_store_ps(outptr, vin4);
      inptr0 += 8;
      _mm256_store_ps(outptr + 8, vin5);
      inptr1 += 8;
      _mm256_store_ps(outptr + 16, vin1);
      inptr2 += 8;
      _mm256_store_ps(outptr + 24, vtmp0);
      inptr3 += 8;
      _mm256_store_ps(outptr + 32, vtmp1);
      inptr4 += 8;
      _mm256_store_ps(outptr + 40, vin3);
      inptr5 += 8;
      outptr += 48;
    }
    for (int x = 0; x < cnt_4; x++) {
      __m128 vin0 = _mm_loadu_ps(inptr0);
      __m128 vin1 = _mm_loadu_ps(inptr1);
      __m128 vin2 = _mm_loadu_ps(inptr2);
      __m128 vin3 = _mm_loadu_ps(inptr3);
      __m128 vin4 = _mm_loadu_ps(inptr4);
      __m128 vin5 = _mm_loadu_ps(inptr5);
      if (has_alpha) {
        vin0 = _mm_mul_ps(vin0, valpha_128);
        vin1 = _mm_mul_ps(vin1, valpha_128);
        vin2 = _mm_mul_ps(vin2, valpha_128);
        vin3 = _mm_mul_ps(vin3, valpha_128);
        vin4 = _mm_mul_ps(vin4, valpha_128);
        vin5 = _mm_mul_ps(vin5, valpha_128);
      }
      // vtmp0=a0b0a2b2
      __m128 vtmp0 = _mm_unpacklo_ps(vin0, vin1);
      // vtmp2=c0d0c2d2
      __m128 vtmp2 = _mm_unpacklo_ps(vin2, vin3);
      // vtmp4=e0f0e2f2
      __m128 vtmp4 = _mm_unpacklo_ps(vin4, vin5);
      // vtmp1=a1b1a3b3
      __m128 vtmp1 = _mm_unpackhi_ps(vin0, vin1);
      __m128 vtmp3 = _mm_unpackhi_ps(vin2, vin3);
      // vtmp5=e1f1e3f3
      __m128 vtmp5 = _mm_unpackhi_ps(vin4, vin5);
      // vtran0 = a0b0c0d0
      __m128 vtran0 = _mm_shuffle_ps(vtmp0, vtmp2, 0xaa);  // 0xaa=[01,00,01,00]
      // vtran1 = a2b2c2d2
      __m128 vtran1 = _mm_shuffle_ps(vtmp0, vtmp2, 0xdd);  // 0xaa=[11,10,11,10]
      // vtran2 = e0f0a1b1
      __m128 vtran2 = _mm_shuffle_ps(vtmp4, vtmp1, 0xaa);  // 0xaa=[01,00,01,00]
      // vtran1 = e2f2a3b3
      __m128 vtran3 = _mm_shuffle_ps(vtmp4, vtmp1, 0xdd);  // 0xaa=[11,10,11,10]
      // vtran0 = c1d1e1f1
      __m128 vtran4 = _mm_shuffle_ps(vtmp3, vtmp5, 0xaa);  // 0xaa=[01,00,01,00]
      // vtran1 = c3d3e3f3
      __m128 vtran5 = _mm_shuffle_ps(vtmp3, vtmp5, 0xdd);  // 0xaa=[11,10,11,10]
      _mm_store_ps(outptr, vtran0);
      inptr0 += 4;
      _mm_store_ps(outptr + 4, vtran2);
      inptr1 += 4;
      _mm_store_ps(outptr + 8, vtran4);
      inptr2 += 4;
      _mm_store_ps(outptr + 12, vtran1);
      inptr3 += 4;
      _mm_store_ps(outptr + 16, vtran3);
      inptr4 += 4;
      _mm_store_ps(outptr + 20, vtran5);
      inptr5 += 4;
      outptr += 24;
    }
    for (int x = 0; x < remain; x++) {
      if (has_alpha) {
        *outptr++ = *inptr0++ * alpha;
        *outptr++ = *inptr1++ * alpha;
        *outptr++ = *inptr2++ * alpha;
        *outptr++ = *inptr3++ * alpha;
        *outptr++ = *inptr4++ * alpha;
        *outptr++ = *inptr5++ * alpha;
      } else {
        *outptr++ = *inptr0++;
        *outptr++ = *inptr1++;
        *outptr++ = *inptr2++;
        *outptr++ = *inptr3++;
        *outptr++ = *inptr4++;
        *outptr++ = *inptr5++;
      }
    }
  }
}

/**
 * \brief data prepack in 6-uint
 * in: k * m
 * out: m / 6 * k * 6
 * if has_alpha then in = in * alpha
*/
void prepackA_trans_6x4(float *out,
                        const float *in,
                        float alpha,
                        const int ldin,
                        const int m0,
                        const int mmax,
                        const int k0,
                        const int kmax) {
  float *outptr = out;
  const float *inptr = in + k0 * ldin + m0;

  int x_len = mmax - m0;
  int y_len = kmax - k0;
  int cnt = x_len / 6;
  uint16_t right_remain = x_len & 5;
  int stride_out = 6 * y_len;
  bool has_alpha = fabsf(alpha - 1.f) > 1e-8f;

  __m128 valpha = _mm_set1_ps(alpha);

#pragma omp parallel for
  for (int y = 0; y < y_len - 3; y += 4) {
    const float *ptr0 = inptr + y * ldin;
    const float *ptr1 = ptr0 + ldin;
    const float *ptr2 = ptr1 + ldin;
    const float *ptr3 = ptr2 + ldin;
    float *outptr_row_col = outptr + y * 6;
    for (int i = 0; i < cnt; i++) {
      __m128 vin0 = _mm_loadu_ps(ptr0);
      __m128 vin1 = _mm_loadu_ps(ptr1);
      __m128 vin2 = _mm_loadu_ps(ptr2);
      __m128 vin3 = _mm_loadu_ps(ptr3);
      if (has_alpha) {
        vin0 = _mm_mul_ps(vin0, valpha);
        vin1 = _mm_mul_ps(vin1, valpha);
        vin2 = _mm_mul_ps(vin2, valpha);
        vin3 = _mm_mul_ps(vin3, valpha);
        outptr_row_col[4] = ptr0[4] * alpha;
        outptr_row_col[5] = ptr0[5] * alpha;
        outptr_row_col[10] = ptr1[4] * alpha;
        outptr_row_col[11] = ptr1[5] * alpha;
        outptr_row_col[16] = ptr2[4] * alpha;
        outptr_row_col[17] = ptr2[5] * alpha;
        outptr_row_col[22] = ptr3[4] * alpha;
        outptr_row_col[23] = ptr3[5] * alpha;
      } else {
        outptr_row_col[4] = ptr0[4];
        outptr_row_col[5] = ptr0[5];
        outptr_row_col[10] = ptr1[4];
        outptr_row_col[11] = ptr1[5];
        outptr_row_col[16] = ptr2[4];
        outptr_row_col[17] = ptr2[5];
        outptr_row_col[22] = ptr3[4];
        outptr_row_col[23] = ptr3[5];
      }
      __mm_store_ps(outptr_row_col, vin0);
      __mm_store_ps(outptr_row_col + 6, vin1);
      __mm_store_ps(outptr_row_col + 12, vin2);
      __mm_store_ps(outptr_row_col + 18, vin3);
      ptr0 += 6;
      ptr1 += 6;
      ptr2 += 6;
      ptr3 += 6;
      outptr_row_col += stride_out;
    }
    for (int i = 0; i < right_remain; i++) {
      if (has_alpha) {
        outptr_row_col[i] = (*ptr0++ * alpha);
        outptr_row_col[i + 6] = (*ptr1++ * alpha);
        outptr_row_col[i + 12] = (*ptr2++ * alpha);
        outptr_row_col[i + 18] = (*ptr3++ * alpha);
      } else {
        outptr_row_col[i] = *ptr0++;
        outptr_row_col[i + 6] = *ptr1++;
        outptr_row_col[i + 12] = *ptr2++;
        outptr_row_col[i + 18] = *ptr3++;
      }
    }
    for (int i = right_remain; i < 6; i++) {
      outptr_row_col[i] = 0.f;
      outptr_row_col[i + 6] = 0.f;
      outptr_row_col[i + 12] = 0.f;
      outptr_row_col[i + 18] = 0.f;
    }
  }
#pragma omp parallel for
  for (int y = 4 * (y_len / 4); y < y_len; ++y) {
    const float *ptr0 = inptr + y * ldin;
    float *outptr_row_col = outptr + y * 6;
    for (int i = 0; i < cnt; i++) {
      __m128 vin0 = _mm_loadu_ps(ptr0);
      if (has_alpha) {
        vin0 = _mm_mul_ps(vin0, valpha);
        outptr_row_col[4] = ptr0[4] * alpha;
        outptr_row_col[5] = ptr0[5] * alpha;
      } else {
        outptr_row_col[4] = ptr0[4];
        outptr_row_col[5] = ptr0[5];
      }
      __mm_store_ps(outptr_row_col, vin0);
      ptr0 += 6;
      outptr_row_col += stride_out;
    }
    for (int i = 0; i < right_remain; i++) {
      if (has_alpha) {
        outptr_row_col[i] = (*ptr0++ * alpha);
      } else {
        outptr_row_col[i] = *ptr0++;
      }
    }
    for (int i = right_remain; i < 6; i++) {
      outptr_row_col[i] = 0.f;
    }
  }
}

/**
 * @brief data prepack in 16-unit
 * in: k * n
 * out: n / 16 * k * 16 + n / 8 * k * 8 + n / 4 * k * 4 + remain(n%4) * k
 */
void loadb(float *out,
           const float *in,
           const int ldin,
           const int k0,
           const int kmax,
           const int n0,
           const int nmax) {
  uint16_t *outptr = reinterpret_cast<uint16_t *>(out);
  const uint16_t *inptr =
      reinterpret_cast<const uint16_t *>(in) + k0 * ldin + n0;
  uint16_t mask_buffer[4] = {0, 1, 2, 3};
  int x_len = nmax - n0;
  int y_len = kmax - k0;
  int cnt = x_len >> 4;
  int right_remain = x_len & 15;

  uint16_t *outptr_row = outptr;
  int rem_8 = right_remain >> 3;
  int rem_rem_8 = right_remain & 7;
  int rem_cnt = rem_8 >> 2;
  int rem_rem = rem_8 & 3;

  int cnt_y = 4 * (y_len / 4);
  // int stride_k = y_len << 1;
  int cnt_16 = (cnt > 0) ? 16 : 0;
  int cnt_8 = (rem_8 > 0) ? 8 : 0;
  int cnt_4 = (rem_cnt > 0) ? 4 : 0;
  int cnt_1 = (rem_rem > 0) ? 1 : 0;
  int stride_16 = cnt_16 * y_len;
  int stride_8 = cnt_8 * y_len;
  int stride_4 = cnt_4 * y_len;
  int stride_1 = cnt_1 * y_len;
  int stride_w_8 = stride_16 * cnt;
  int stride_w_4 = stride_w_8 + stride_8 * rem_8;
  int stride_w_1 = stride_w_4 + stride_4 * rem_cnt;

#pragma omp parallel for
  for (int y = 0; y < y_len - 3; y += 4) {
    const uint16_t *ptr0 = inptr + y * ldin;
    const uint16_t *ptr1 = ptr0 + ldin;
    const uint16_t *ptr2 = ptr1 + ldin;
    const uint16_t *ptr3 = ptr2 + ldin;

    uint16_t *outptr_row_col = outptr_row + y * cnt_16;
    uint16_t *outptr_row_8 = outptr_row + stride_w_8 + y * cnt_8;
    uint16_t *outptr_row_4 = outptr_row + stride_w_4 + y * cnt_4;
    uint16_t *outptr_row_1 = outptr_row + stride_w_1 + y * cnt_1;
    if (cnt > 0) {
      for (int i = 0; i < cnt; i++) {
        __m256i v0 = _m256_loadu_ps(ptr0);
        __m256i v01 = _m256_loadu_ps(ptr0 + 8);
        __m256i v1 = _m256_loadu_ps(ptr1);
        __m256i v11 = _m256_loadu_ps(ptr1 + 8);
        __m256i v2 = _m256_loadu_ps(ptr2);
        __m256i v21 = _m256_loadu_ps(ptr2 + 8);
        _m256_store_ps(outptr_row_col, v0);
        __m256i v3 = _m256_loadu_ps(ptr3);
        _m256_store_ps(outptr_row_col + 8, v01);
        __m256i v31 = _m256_loadu_ps(ptr3 + 8);
        _m256_store_ps(outptr_row_col + 16, v1);
        ptr0 += 16;
        _m256_store_ps(outptr_row_col + 24, v11);
        ptr1 += 16;
        _m256_store_ps(outptr_row_col + 32, v2);
        ptr2 += 16;
        _m256_store_ps(outptr_row_col + 40, v21);
        ptr3 += 16;
        _m256_store_ps(outptr_row_col + 48, v3);
        _m256_store_ps(outptr_row_col + 56, v31);
        outptr_row_col += stride_16;
      }
    }
    if (rem_8 > 0) {
      __m256i v0 = _m256_loadu_ps(ptr0);
      __m256i v1 = _m256_loadu_ps(ptr1);
      __m256i v2 = _m256_loadu_ps(ptr2);
      __m256i v3 = _m256_loadu_ps(ptr3);
      ptr0 += 8;
      _m256_store_ps(outptr_row_8, v0);
      ptr1 += 8;
      _m256_store_ps(outptr_row_8 + 8, v1);
      ptr2 += 8;
      _m256_store_ps(outptr_row_8 + 16, v2);
      ptr3 += 8;
      _m256_store_ps(outptr_row_8 + 24, v3);
    }
    if (rem_cnt > 0) {
      __m128i v0 = _mm_loadu_ps(ptr0);
      __m128i v1 = _mm_loadu_ps(ptr1);
      __m128i v2 = _mm_loadu_ps(ptr2);
      __m128i v3 = _mm_loadu_ps(ptr3);
      ptr0 += 4;
      _mm_store_ps(outptr_row_4, v0);
      ptr1 += 4;
      _mm_store_ps(outptr_row_4 + 4, v1);
      ptr2 += 4;
      _mm_store_ps(outptr_row_4 + 8, v2);
      ptr3 += 4;
      _mm_store_ps(outptr_row_4 + 12, v3);
    }
    if (rem_rem > 0) {
      for (int i = 0; i < rem_rem; i++) {
        outptr_row_1[0] = *ptr0++;
        outptr_row_1[1] = *ptr1++;
        outptr_row_1[2] = *ptr2++;
        outptr_row_1[3] = *ptr3++;
        outptr_row_1 += stride_1;
      }
    }
  }

#pragma omp parallel for
  for (int y = cnt_y; y < y_len; ++y) {
    const uint16_t *ptr0 = inptr + y * ldin;
    uint16_t *outptr_row_col = outptr_row + y * cnt_16;
    uint16_t *outptr_row_8 = outptr_row + stride_w_8 + y * cnt_8;
    uint16_t *outptr_row_4 = outptr_row + stride_w_4 + y * cnt_4;
    uint16_t *outptr_row_1 = outptr_row + stride_w_1 + y * cnt_1;
    if (cnt > 0) {
      for (int i = 0; i < cnt; i++) {
        __m256i v0 = _m256_loadu_ps(ptr0);
        __m256i v01 = _m256_loadu_ps(ptr0 + 8);
        _m256_store_ps(outptr_row_col, v0);
        _m256_store_ps(outptr_row_col + 8, v01);
        ptr0 += 16;
        outptr_row_col += stride_16;
      }
    }
    if (rem_8 > 0) {
      __m256i v0 = _m256_loadu_ps(ptr0);
      ptr0 += 8;
      _m256_store_ps(outptr_row_8, v0);
    }
    if (rem_cnt > 0) {
      __m128i v0 = _mm_loadu_ps(ptr0);
      ptr0 += 4;
      _mm_store_ps(outptr_row_4, v0);
    }
    if (rem_rem > 0) {
      for (int i = 0; i < rem_rem; i++) {
        *outptr_row_1 = *ptr0++;
        outptr_row_1 += stride_1;
      }
    }
  }
}

/**
 * @brief do C16 Trans
 * in: n * k
 * out: n / 16 * k * 16 + n / 8 * k * 8 + n / 4 * k * 4 + remain(n%4) * k
 */
void loadb_trans(float *out,
                 const float *in,
                 const int ldin,
                 const int k0,
                 const int kmax,
                 const int n0,
                 const int nmax) {
  int x_len = kmax - k0;
  int size = ((x_len + 7) / 8) * 8;
  uint16_t *outptr = reinterpret_cast<uint16_t *>(out);
  const uint16_t *inptr = reinterpret_cast<const uint16_t *>(in);
  uint16_t zerobuff[size];  // NOLINT
  memset(zerobuff, 0, sizeof(uint16_t) * size);
  int cnt = x_len >> 3;
  int remain = x_len & 7;
  int y = n0;
  int y_remain = (nmax - n0) & 3;

  //! data B is not transposed, transpose B to k * 16
  for (; y < nmax - 15; y += 16) {
    const uint16_t *inptr0 = inptr + y * ldin + k0;
    const uint16_t *inptr1 = inptr0 + ldin;
    const uint16_t *inptr2 = inptr1 + ldin;
    const uint16_t *inptr3 = inptr2 + ldin;
    const uint16_t *inptr4 = inptr3 + ldin;
    const uint16_t *inptr5 = inptr4 + ldin;
    const uint16_t *inptr6 = inptr5 + ldin;
    const uint16_t *inptr7 = inptr6 + ldin;
    const uint16_t *inptr8 = inptr7 + ldin;
    const uint16_t *inptr9 = inptr8 + ldin;
    const uint16_t *inptr10 = inptr9 + ldin;
    const uint16_t *inptr11 = inptr10 + ldin;
    const uint16_t *inptr12 = inptr11 + ldin;
    const uint16_t *inptr13 = inptr12 + ldin;
    const uint16_t *inptr14 = inptr13 + ldin;
    const uint16_t *inptr15 = inptr14 + ldin;

    //! cope with row index exceed real size, set to zero buffer
    for (int x = 0; x < cnt; x++) {
      __m256 vin0 = _mm256_loadu_ps(inptr0);
      __m256 vin1 = _mm256_loadu_ps(inptr1);
      __m256 vin2 = _mm256_loadu_ps(inptr2);
      __m256 vin3 = _mm256_loadu_ps(inptr3);
      __m256 vin4 = _mm256_loadu_ps(inptr4);
      __m256 vin5 = _mm256_loadu_ps(inptr5);
      __m256 vin6 = _mm256_loadu_ps(inptr6);
      __m256 vin7 = _mm256_loadu_ps(inptr7);
      transpose8_ps(vin0, vin1, vin2, vin3, vin4, vin5, vin6, vin7);
      __m256 vin8 = _mm256_loadu_ps(inptr8);
      __m256 vin9 = _mm256_loadu_ps(inptr9);
      __m256 vin10 = _mm256_loadu_ps(inptr10);
      __m256 vin11 = _mm256_loadu_ps(inptr11);
      _mm256_store_ps(outptr, vin0);
      __m256 vin12 = _mm256_loadu_ps(inptr12);
      _mm256_store_ps(outptr + 16, vin1);
      __m256 vin13 = _mm256_loadu_ps(inptr13);
      _mm256_store_ps(outptr + 32, vin2);
      __m256 vin14 = _mm256_loadu_ps(inptr14);
      _mm256_store_ps(outptr + 48, vin3);
      __m256 vin15 = _mm256_loadu_ps(inptr15);
      transpose8_ps(vin8, vin9, vin10, vin11, vin12, vin13, vin14, vin15);
      _mm256_store_ps(outptr + 8, vin8);
      inptr0 += 8;
      _mm256_store_ps(outptr + 24, vin9);
      inptr1 += 8;
      _mm256_store_ps(outptr + 40, vin10);
      inptr2 += 8;
      _mm256_store_ps(outptr + 56, vin11);
      inptr3 += 8;
      _mm256_store_ps(outptr + 64, vin4);
      inptr4 += 8;
      _mm256_store_ps(outptr + 72, vin12);
      inptr5 += 8;
      _mm256_store_ps(outptr + 80, vin5);
      inptr6 += 8;
      _mm256_store_ps(outptr + 88, vin13);
      inptr7 += 8;
      _mm256_store_ps(outptr + 96, vin6);
      inptr8 += 8;
      _mm256_store_ps(outptr + 104, vin14);
      inptr9 += 8;
      _mm256_store_ps(outptr + 112, vin7);
      inptr10 += 8;
      _mm256_store_ps(outptr + 120, vin15);
      inptr11 += 8;
      inptr12 += 8;
      inptr13 += 8;
      inptr14 += 8;
      inptr15 += 8;
      outptr += 128;
    }
    for (int x = 0; x < remain; x++) {
      *outptr++ = *inptr0++;
      *outptr++ = *inptr1++;
      *outptr++ = *inptr2++;
      *outptr++ = *inptr3++;
      *outptr++ = *inptr4++;
      *outptr++ = *inptr5++;
      *outptr++ = *inptr6++;
      *outptr++ = *inptr7++;
      *outptr++ = *inptr8++;
      *outptr++ = *inptr9++;
      *outptr++ = *inptr10++;
      *outptr++ = *inptr11++;
      *outptr++ = *inptr12++;
      *outptr++ = *inptr13++;
      *outptr++ = *inptr14++;
      *outptr++ = *inptr15++;
    }
  }
  for (; y < nmax - 7; y += 8) {
    const uint16_t *inptr0 = inptr + y * ldin + k0;
    const uint16_t *inptr1 = inptr0 + ldin;
    const uint16_t *inptr2 = inptr1 + ldin;
    const uint16_t *inptr3 = inptr2 + ldin;
    const uint16_t *inptr4 = inptr3 + ldin;
    const uint16_t *inptr5 = inptr4 + ldin;
    const uint16_t *inptr6 = inptr5 + ldin;
    const uint16_t *inptr7 = inptr6 + ldin;

    //! cope with row index exceed real size, set to zero buffer
    for (int x = 0; x < cnt; x++) {
      __m256 vin0 = _mm256_loadu_ps(inptr0);
      __m256 vin1 = _mm256_loadu_ps(inptr1);
      __m256 vin2 = _mm256_loadu_ps(inptr2);
      __m256 vin3 = _mm256_loadu_ps(inptr3);
      __m256 vin4 = _mm256_loadu_ps(inptr4);
      __m256 vin5 = _mm256_loadu_ps(inptr5);
      __m256 vin6 = _mm256_loadu_ps(inptr6);
      __m256 vin7 = _mm256_loadu_ps(inptr7);
      transpose8_ps(vin0, vin1, vin2, vin3, vin4, vin5, vin6, vin7);
      inptr0 += 8;
      _mm256_store_ps(outptr, vin0);
      inptr1 += 8;
      _mm256_store_ps(outptr + 8, vin1);
      inptr2 += 8;
      _mm256_store_ps(outptr + 16, vin2);
      inptr3 += 8;
      _mm256_store_ps(outptr + 24, vin3);
      inptr4 += 8;
      _mm256_store_ps(outptr + 32, vin4);
      inptr5 += 8;
      _mm256_store_ps(outptr + 40, vin5);
      inptr6 += 8;
      _mm256_store_ps(outptr + 48, vin6);
      inptr7 += 8;
      _mm256_store_ps(outptr + 56, vin7);
      outptr += 64;
    }
    for (int x = 0; x < remain; x++) {
      *outptr++ = *inptr0++;
      *outptr++ = *inptr1++;
      *outptr++ = *inptr2++;
      *outptr++ = *inptr3++;
      *outptr++ = *inptr4++;
      *outptr++ = *inptr5++;
      *outptr++ = *inptr6++;
      *outptr++ = *inptr7++;
    }
  }
  for (; y < nmax - 3; y += 4) {
    const uint16_t *inptr0 = inptr + y * ldin + k0;
    const uint16_t *inptr1 = inptr0 + ldin;
    const uint16_t *inptr2 = inptr1 + ldin;
    const uint16_t *inptr3 = inptr2 + ldin;

    //! cope with row index exceed real size, set to zero buffer
    for (int x = 0; x < cnt; x++) {
      __m256 vin0 = _mm256_loadu_ps(inptr0);
      __m256 vin1 = _mm256_loadu_ps(inptr1);
      __m256 vin2 = _mm256_loadu_ps(inptr2);
      __m256 vin3 = _mm256_loadu_ps(inptr3);
      transpose4x8_ps(vin0, vin1, vin2, vin3);
      // 4x8
      inptr0 += 8;
      _mm256_store_ps(outptr, vin0);
      inptr1 += 8;
      _mm256_store_ps(outptr + 8, vin1);
      inptr2 += 8;
      _mm256_store_ps(outptr + 16, vin2);
      inptr3 += 8;
      _mm256_store_ps(outptr + 24, vin3);
      outptr += 32;
    }
    for (int x = 0; x < remain; x++) {
      *outptr++ = *inptr0++;
      *outptr++ = *inptr1++;
      *outptr++ = *inptr2++;
      *outptr++ = *inptr3++;
    }
  }

  if (y_remain == 1) {
    memcpy(outptr, inptr + y * ldin + k0, sizeof(uint16_t) * x_len);
  } else if (y_remain == 2) {
    const uint16_t *inptr0 = inptr + y * ldin + k0;
    const uint16_t *inptr1 = inptr0 + ldin;
    memcpy(outptr, inptr0, sizeof(uint16_t) * x_len);
    memcpy(outptr + x_len, inptr1, sizeof(uint16_t) * x_len);
  } else if (y_remain == 3) {
    const uint16_t *inptr0 = inptr + y * ldin + k0;
    const uint16_t *inptr1 = inptr0 + ldin;
    const uint16_t *inptr2 = inptr1 + ldin;
    memcpy(outptr, inptr0, sizeof(uint16_t) * x_len);
    memcpy(outptr + x_len, inptr1, sizeof(uint16_t) * x_len);
    memcpy(outptr + 2 * x_len, inptr2, sizeof(uint16_t) * x_len);
  }
}
#else
/**
 * @brief do C4 Trans
 * in: m * k
 * out: m / 4 * k * 4
 * if has_alpha then in = in * alpha
 */
void prepackA_4x4(float *out,
                  const float *in,
                  float alpha,
                  const int ldin,
                  const int m0,
                  const int mmax,
                  const int k0,
                  const int kmax) {
  // suport 4x4 trans
  // _MM_TRANSPOSE4_PS(__m128 row0, __m128 row1, __m128 row2, __m128 row3)
  int x_len = kmax - k0;
  float zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(float) * x_len);

  float *dout = out;
  const float *inptr = in;
  bool has_alpha = fabsf(alpha - 1.f) > 1e-8f;

  int cnt = x_len / 4;
  int remain = x_len & 3;
  __m128 valpha = _mm_set1_ps(alpha);
#pragma omp parallel for
  for (int y = m0; y < mmax; y += 4) {
    float *outptr = dout + (y - m0) * x_len;
    const float *inptr0 = inptr + y * ldin + k0;
    const float *inptr1 = inptr0 + ldin;
    const float *inptr2 = inptr1 + ldin;
    const float *inptr3 = inptr2 + ldin;
    if ((y + 3) >= mmax) {
      switch (y + 3 - mmax) {
        case 2:
          inptr1 = zerobuff;
        case 1:
          inptr2 = zerobuff;
        case 0:
          inptr3 = zerobuff;
        default:
          break;
      }
    }
    for (int x = 0; x < cnt; x++) {
      __m128 vin0 = _mm_loadu_ps(inptr0);
      __m128 vin1 = _mm_loadu_ps(inptr1);
      __m128 vin2 = _mm_loadu_ps(inptr2);
      __m128 vin3 = _mm_loadu_ps(inptr3);
      if (has_alpha) {
        vin0 = _mm_mul_ps(vin0, valpha);
        vin1 = _mm_mul_ps(vin1, valpha);
        vin2 = _mm_mul_ps(vin2, valpha);
        vin3 = _mm_mul_ps(vin3, valpha);
      }
      _MM_TRANSPOSE4_PS(vin0, vin1, vin2, vin3)

      _mm_store_ps(outptr, vin0);
      inptr0 += 4;
      _mm_store_ps(outptr + 4, vin1);
      inptr1 += 4;
      _mm_store_ps(outptr + 8, vin2);
      inptr2 += 4;
      _mm_store_ps(outptr + 12, vin3);
      inptr3 += 4;
      outptr += 16;
    }
    for (int x = 0; x < remain; x++) {
      if (has_alpha) {
        *outptr++ = *inptr0++ * alpha;
        *outptr++ = *inptr1++ * alpha;
        *outptr++ = *inptr2++ * alpha;
        *outptr++ = *inptr3++ * alpha;
      } else {
        *outptr++ = *inptr0++;
        *outptr++ = *inptr1++;
        *outptr++ = *inptr2++;
        *outptr++ = *inptr3++;
      }
    }
  }
}

/**
 * @brief data prepack in 4-uint
 * in: k * m
 * out: m / 4 * k * 4
 * if has_alpha then in = in * alpha
 */
void prepackA_trans_4x4(float *out,
                        const float *in,
                        float alpha,
                        const int ldin,
                        const int m0,
                        const int mmax,
                        const int k0,
                        const int kmax) {
  float *outptr = out;
  const float *inptr = in + k0 * ldin + m0;

  int x_len = mmax - m0;
  int y_len = kmax - k0;
  int cnt = x_len >> 2;
  uint16_t right_remain = x_len & 3;
  int stride_out = 4 * y_len;
  bool has_alpha = fabsf(alpha - 1.f) > 1e-8f;
  __m128 valpha = _mm_set1_ps(alpha);

#pragma omp parallel for
  for (int y = 0; y < y_len - 3; y += 4) {
    const float *ptr0 = inptr + y * ldin;
    const float *ptr1 = ptr0 + ldin;
    const float *ptr2 = ptr1 + ldin;
    const float *ptr3 = ptr2 + ldin;
    float *outptr_row_col = outptr + y * 4;
    for (int i = 0; i < cnt; i++) {
      __m128 vin0 = _mm_loadu_ps(ptr0);
      __m128 vin1 = _mm_loadu_ps(ptr1);
      __m128 vin2 = _mm_loadu_ps(ptr2);
      __m128 vin3 = _mm_loadu_ps(ptr3);
      if (has_alpha) {
        vin0 = _mm_mul_ps(vin0, valpha);
        vin1 = _mm_mul_ps(vin1, valpha);
        vin2 = _mm_mul_ps(vin2, valpha);
        vin3 = _mm_mul_ps(vin3, valpha);
      }
      __mm_store_ps(outptr_row_col, vin0);
      ptr0 += 4;
      __mm_store_ps(outptr_row_col + 4, vin1);
      ptr1 += 4;
      __mm_store_ps(outptr_row_col + 8, vin2);
      ptr2 += 4;
      __mm_store_ps(outptr_row_col + 12, vin3);
      ptr3 += 4;
      outptr_row_col += stride_out;
    }
    for (int i = 0; i < right_remain; i++) {
      if (has_alpha) {
        outptr_row_col[i] = (*ptr0++ * alpha);
        outptr_row_col[i + 4] = (*ptr1++ * alpha);
        outptr_row_col[i + 8] = (*ptr2++ * alpha);
        outptr_row_col[i + 12] = (*ptr3++ * alpha);
      } else {
        outptr_row_col[i] = *ptr0++;
        outptr_row_col[i + 4] = *ptr1++;
        outptr_row_col[i + 8] = *ptr2++;
        outptr_row_col[i + 12] = *ptr3++;
      }
    }
    for (int i = right_remain; i < 4; i++) {
      outptr_row_col[i] = 0.f;
      outptr_row_col[i + 4] = 0.f;
      outptr_row_col[i + 8] = 0.f;
      outptr_row_col[i + 12] = 0.f;
    }
  }
#pragma omp parallel for
  for (int y = 4 * (y_len / 4); y < y_len; ++y) {
    const float *ptr0 = inptr + y * ldin;
    float *outptr_row_col = outptr + y * 4;
    for (int i = 0; i < cnt; i++) {
      __m128 vin0 = _mm_loadu_ps(ptr0);
      if (has_alpha) {
        vin0 = _mm_mul_ps(vin0, valpha);
      }
      ptr0 += 4;
      __mm_store_ps(outptr_row_col, vin0);
      outptr_row_col += stride_out;
    }
    for (int i = 0; i < right_remain; i++) {
      if (has_alpha) {
        outptr_row_col[i] = (*ptr0++ * alpha);
      } else {
        outptr_row_col[i] = *ptr0++;
      }
    }
    for (int i = right_remain; i < 4; i++) {
      outptr_row_col[i] = 0.f;
    }
  }
}

/**
 * @brief prepack in 4-unit
 * in: k * n
 * out: n / 4 * k * 4 + remain(n%4) * k
 */
void loadb(float *out,
           const float *in,
           const int ldin,
           const int k0,
           const int kmax,
           const int n0,
           const int nmax) {
  uint16_t *outptr = reinterpret_cast<uint16_t *>(out);
  const uint16_t *inptr =
      reinterpret_cast<const uint16_t *>(in) + k0 * ldin + n0;
  int x_len = nmax - n0;
  int y_len = kmax - k0;
  int cnt = x_len >> 2;
  int right_remain = x_len & 3;
  uint16_t *outptr_row = outptr;

  int cnt_y = 4 * (y_len / 4);
  // int stride_k = y_len << 1;
  int cnt_4 = (cnt > 0) ? 4 : 0;
  int cnt_1 = (right_remain > 0) ? 1 : 0;
  int stride_4 = cnt_4 * y_len;
  int stride_1 = cnt_1 * y_len;
  int stride_w = stride_4 * cnt;

#pragma omp parallel for
  for (int y = 0; y < y_len - 3; y += 4) {
    const uint16_t *ptr0 = inptr + y * ldin;
    const uint16_t *ptr1 = ptr0 + ldin;
    const uint16_t *ptr2 = ptr1 + ldin;
    const uint16_t *ptr3 = ptr2 + ldin;

    uint16_t *outptr_row_col = outptr_row + y * cnt_4;
    uint16_t *outptr_row_1 = outptr_row + stride_w + y * cnt_1;
    if (cnt > 0) {
      for (int x = 0; x < cnt; x++) {
        __m128i v0 = _mm_loadu_ps(ptr0);
        __m128i v1 = _mm_loadu_ps(ptr1);
        __m128i v2 = _mm_loadu_ps(ptr2);
        __m128i v3 = _mm_loadu_ps(ptr3);
        ptr0 += 4;
        _mm_store_ps(outptr_row_col, v0);
        ptr1 += 4;
        _mm_store_ps(outptr_row_col + 4, v1);
        ptr2 += 4;
        _mm_store_ps(outptr_row_col + 8, v2);
        ptr3 += 4;
        _mm_store_ps(outptr_row_col + 12, v3);
        outptr_row_col += stride_4;
      }
    }
    if (right_remain > 0) {
      for (int i = 0; i < right_remain; i++) {
        outptr_row_1[0] = *ptr0++;
        outptr_row_1[1] = *ptr1++;
        outptr_row_1[2] = *ptr2++;
        outptr_row_1[3] = *ptr3++;
        outptr_row_1 += stride_1;
      }
    }
  }

#pragma omp parallel for
  for (int y = cnt_y; y < y_len; ++y) {
    const uint16_t *ptr0 = inptr + y * ldin;
    uint16_t *outptr_row_col = outptr_row + y * cnt_4;
    uint16_t *outptr_row_1 = outptr_row + stride_w + y * cnt_1;
    if (cnt > 0) {
      for (int i = 0; i < cnt; i++) {
        __m128i v0 = _mm_loadu_ps(ptr0);
        ptr0 += 4;
        _mm_store_ps(outptr_row_4, v0);
        outptr_row_col += stride_4;
      }
    }
    if (right_remain > 0) {
      for (int i = 0; i < right_remain; i++) {
        *outptr_row_1 = *ptr0++;
        outptr_row_1 += stride_1;
      }
    }
  }
}

/**
 * @brief do C4 Trans
 * in: n * k
 * out: n / 4 * k * 4 + remain(n%4) * k
 */
void loadb_trans(float *out,
                 const float *in,
                 const int ldin,
                 const int k0,
                 const int kmax,
                 const int n0,
                 const int nmax) {
  int x_len = kmax - k0;
  int size = ((x_len + 7) / 8) * 8;
  uint16_t *outptr = reinterpret_cast<uint16_t *>(out);
  const uint16_t *inptr = reinterpret_cast<const uint16_t *>(in);
  uint16_t zerobuff[size];  // NOLINT
  memset(zerobuff, 0, sizeof(uint16_t) * size);
  int cnt = x_len >> 2;
  int remain = x_len & 3;
  int y = n0;
  int y_remain = (nmax - n0) & 3;

  //! data B is not transposed, transpose B to k * 16
  for (; y < nmax - 3; y += 4) {
    const uint16_t *inptr0 = inptr + y * ldin + k0;
    const uint16_t *inptr1 = inptr0 + ldin;
    const uint16_t *inptr2 = inptr1 + ldin;
    const uint16_t *inptr3 = inptr2 + ldin;

    //! cope with row index exceed real size, set to zero buffer
    for (int x = 0; x < cnt; x++) {
      __mm vin0 = _mm_loadu_ps(inptr0);
      __mm vin1 = _mm_loadu_ps(inptr1);
      __mm vin2 = _mm_loadu_ps(inptr2);
      __mm vin3 = _mm_loadu_ps(inptr3);
      // 4x4
      _MM_TRANSPOSE4_PS(vin0, vin1, vin2, vin3)
      inptr0 += 4;
      _mm256_store_ps(outptr, vin0);
      inptr1 += 4;
      _mm256_store_ps(outptr + 4, vin1);
      inptr2 += 4;
      _mm256_store_ps(outptr + 8, vin2);
      inptr3 += 4;
      _mm256_store_ps(outptr + 12, vin3);
      outptr += 16;
    }
    for (int x = 0; x < remain; x++) {
      *outptr++ = *inptr0++;
      *outptr++ = *inptr1++;
      *outptr++ = *inptr2++;
      *outptr++ = *inptr3++;
    }
  }

  if (y_remain == 1) {
    memcpy(outptr, inptr + y * ldin + k0, sizeof(uint16_t) * x_len);
  } else if (y_remain == 2) {
    const uint16_t *inptr0 = inptr + y * ldin + k0;
    const uint16_t *inptr1 = inptr0 + ldin;
    memcpy(outptr, inptr0, sizeof(uint16_t) * x_len);
    memcpy(outptr + x_len, inptr1, sizeof(uint16_t) * x_len);
  } else if (y_remain == 3) {
    const uint16_t *inptr0 = inptr + y * ldin + k0;
    const uint16_t *inptr1 = inptr0 + ldin;
    const uint16_t *inptr2 = inptr1 + ldin;
    memcpy(outptr, inptr0, sizeof(uint16_t) * x_len);
    memcpy(outptr + x_len, inptr1, sizeof(uint16_t) * x_len);
    memcpy(outptr + 2 * x_len, inptr2, sizeof(uint16_t) * x_len);
  }
}
#endif
#ifdef __AVX__
/**
 * @brief gemm compute of 6x16-unit
 * input A_packed: M * K, format M / 6 * K * 6
 * input B: K * N
 * output C: M * N
 * parameter beta: C = beta * C + A_packed * B
 * parameter bias: if has_bias then C = beta * C + A_packed * B + bias
 * parameter act_param: acitvation proccess
 */
void gemm_prepack_6x16(bool is_transB,
                       int M,
                       int N,
                       int K,
                       const float *A_packed,
                       const float *B,
                       int ldb,
                       float beta,
                       float *C,
                       int ldc,
                       const float *bias,
                       bool has_bias,
                       const operators::ActivationParam act_param,
                       X86Context *ctx) {
  size_t llc_size = ctx->llc_size() > 0 ? ctx->llc_size() : 512 * 1024;
  // auto workspace = ctx->workspace_data<float16_t>();
  int threads = ctx->threads();
  llc_size = llc_size * 9 / 10;

  auto act_type = act_param.active_type;
  float local_alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    act_acquire(act_type,
                flag_act,
                local_alpha,
                act_param.Relu_clipped_coef,
                act_param.Leaky_relu_alpha);
  }
  //! MBLOCK * x (result) + MBLOCK * k (A) + x * k (B) = l2
  X_BLOCK_COMPUTE(llc_size, MBLOCK, NBLOCK, KBLOCK, beta)
  auto b_size = (x_block + 15) / 16 * K;
  // (x_block + 15) / 16 * K
  auto workspace =
      static_cast<float *>(TargetMalloc(TARGET(kx86), b_size * sizeof(float)));
  // memset(workspace, 0, b_size * sizeof(float));

  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    if (xmax > N) {
      xmax = N;
    }
    int bblocks = (xmax - x0 + NBLOCK - 1) / NBLOCK;
    remain = xmax - x0 - (bblocks - 1) * NBLOCK;
    if (remain > 0 && remain != 16) {
      flag_p_remain = true;
    }
    //! load bpanel
    float *b_pannel = workspace;
    if (is_transB) {
      loadb_trans(b_pannel, B, ldb, 0, K, x0, xmax);
    } else {
      loadb(b_pannel, B, ldb, 0, K, x0, xmax);
    }
#pragma omp parallel for num_threads(threads)
    for (unsigned int y = 0; y < M; y += MBLOCK) {
      unsigned int ymax = y + MBLOCK;
      if (ymax > M) {
        ymax = M;
      }

      float bias_local[8] = {0};
      if (has_bias) {
        int k = 0;
        for (int i = y; i < ymax; i++) {
          bias_local[k] = bias[y + k];
          k++;
        }
        for (; k < 8; k++) {
          bias_local[k] = 0.f;
        }
      }
      // prepare out data
      float cout0[NBLOCK];
      float cout1[NBLOCK];
      float cout2[NBLOCK];
      float cout3[NBLOCK];
      float cout4[NBLOCK];
      float cout5[NBLOCK];

      float *c_ptr0 = C + y * ldc + x0;
      float *c_ptr1 = c_ptr0 + ldc;
      float *c_ptr2 = c_ptr1 + ldc;
      float *c_ptr3 = c_ptr2 + ldc;
      float *c_ptr4 = c_ptr3 + ldc;
      float *c_ptr5 = c_ptr4 + ldc;

      const float *a_ptr_l = A_packed + y * K;
      const float *b_ptr = b_pannel;
      for (int xb = 0; xb < bblocks; xb++) {
        if ((y + 5) >= ymax) {
          switch ((y + 5) - ymax) {
            case 4:
              c_ptr1 = cout1;
            case 3:
              c_ptr2 = cout2;
            case 2:
              c_ptr3 = cout3;
            case 1:
              c_ptr4 = cout4;
            case 0:
              c_ptr5 = cout5;
            default:
              break;
          }
        }
        if (flag_p_remain && (xb == bblocks - 1)) {
          int cnt_8 = remain > 7 ? 1 : 0;
          int rem_8 = remain & 7;
          int cnt_4 = rem_8 >> 2;
          int rem_4 = rem_8 & 3;
          // 6 x 8
          if (cnt_8 > 0) {
            const float *a_ptr = a_ptr_l;
            __m256 vout0 = _mm256_set1_ps(bias_local[0]);
            __m256 vout1 = _mm256_set1_ps(bias_local[1]);
            __m256 vout2 = _mm256_set1_ps(bias_local[2]);
            __m256 vout3 = _mm256_set1_ps(bias_local[3]);
            __m256 vout4 = _mm256_set1_ps(bias_local[4]);
            __m256 vout5 = _mm256_set1_ps(bias_local[5]);
            if (has_beta) {
              __m256 vbeta = _mm256_set1_ps(beta);
              __m256 va = _mm256_load_ps(c_ptr0);
              __m256 vb = _mm256_load_ps(c_ptr1);
              __m256 vc = _mm256_load_ps(c_ptr2);
              __m256 vd = _mm256_load_ps(c_ptr3);
              __m256 ve = _mm256_load_ps(c_ptr4);
              vout0 = _mm256_fmadd_ps(va, vbeta, vout0);
              __m256 vf = _mm256_load_ps(c_ptr5);
              vout1 = _mm256_fmadd_ps(vb, vbeta, vout1);
              vout2 = _mm256_fmadd_ps(vc, vbeta, vout2);
              vout3 = _mm256_fmadd_ps(vd, vbeta, vout3);
              vout4 = _mm256_fmadd_ps(ve, vbeta, vout4);
              vout5 = _mm256_fmadd_ps(vf, vbeta, vout5);
            }
            for (int i = 0; i < K - 2; i += 2) {
              __m256 vina0 = _mm256_set1_ps(a_ptr[0]);
              __m256 vinb0 = _mm256_load_ps(b_ptr);
              __m256 vina1 = _mm256_set1_ps(a_ptr[1]);
              __m256 vina2 = _mm256_set1_ps(a_ptr[2]);
              __m256 vina3 = _mm256_set1_ps(a_ptr[3]);
              __m256 vina4 = _mm256_set1_ps(a_ptr[4]);
              vout0 = _mm256_fmadd_ps(vina0, vinb0, vout0);
              __m256 vina5 = _mm256_set1_ps(a_ptr[5]);
              vout1 = _mm256_fmadd_ps(vina1, vinb0, vout1);
              __m256 vinb1 = _mm256_load_ps(b_ptr + 8);
              vout2 = _mm256_fmadd_ps(vina2, vinb0, vout2);
              vina0 = _mm256_set1_ps(a_ptr[6]);
              vout3 = _mm256_fmadd_ps(vina3, vinb0, vout3);
              vina1 = _mm256_set1_ps(a_ptr[7]);
              vout4 = _mm256_fmadd_ps(vina4, vinb0, vout4);
              vina2 = _mm256_set1_ps(a_ptr[8]);
              vout5 = _mm256_fmadd_ps(vina5, vinb0, vout5);
              vina3 = _mm256_set1_ps(a_ptr[9]);
              vout0 = _mm256_fmadd_ps(vina0, vinb1, vout0);
              vina4 = _mm256_set1_ps(a_ptr[10]);
              vout1 = _mm256_fmadd_ps(vina1, vinb1, vout1);
              vina5 = _mm256_set1_ps(a_ptr[11]);
              vout2 = _mm256_fmadd_ps(vina2, vinb1, vout2);
              b_ptr += 16;
              vout3 = _mm256_fmadd_ps(vina3, vinb1, vout3);
              a_ptr += 12;
              vout4 = _mm256_fmadd_ps(vina4, vinb1, vout4);
              vout5 = _mm256_fmadd_ps(vina5, vinb1, vout5);
            }
            if (K % 2) {
              __m256 vina0 = _mm256_set1_ps(a_ptr[0]);
              __m256 vinb0 = _mm256_load_ps(b_ptr);
              __m256 vina1 = _mm256_set1_ps(a_ptr[1]);
              __m256 vina2 = _mm256_set1_ps(a_ptr[2]);
              __m256 vina3 = _mm256_set1_ps(a_ptr[3]);
              __m256 vina4 = _mm256_set1_ps(a_ptr[4]);
              vout0 = _mm256_fmadd_ps(vina0, vinb0, vout0);
              __m256 vina5 = _mm256_set1_ps(a_ptr[5]);
              vout1 = _mm256_fmadd_ps(vina1, vinb0, vout1);
              a_ptr += 6;
              vout2 = _mm256_fmadd_ps(vina2, vinb0, vout2);
              b_ptr += 8;
              vout3 = _mm256_fmadd_ps(vina3, vinb0, vout3);
              vout4 = _mm256_fmadd_ps(vina4, vinb0, vout4);
              vout5 = _mm256_fmadd_ps(vina5, vinb0, vout5);
            }
            if (flag_act == 0) {  // no relu
              __mm256_store_ps(c_ptr0, vout0);
              __mm256_store_ps(c_ptr1, vout1);
            } else if (flag_act == 1) {  // relu
              __m256 vzero = _mm256_set1_ps(0.f);
              vout0 = _mm256_max_ps(vout0, vzero);
              vout1 = _mm256_max_ps(vout1, vzero);
              vout2 = _mm256_max_ps(vout2, vzero);
              vout3 = _mm256_max_ps(vout3, vzero);
              vout4 = _mm256_max_ps(vout4, vzero);
              __mm256_store_ps(c_ptr0, vout0);
              vout5 = _mm256_max_ps(vout5, vzero);
              __mm256_store_ps(c_ptr1, vout1);
            } else if (flag_act == 2) {  // relu6
              __m256 vzero = _mm256_set1_ps(0.f);
              __m256 valpha = _mm256_set1_ps(local_alpha);
              vout0 = _mm256_max_ps(vout0, vzero);
              vout1 = _mm256_max_ps(vout1, vzero);
              vout2 = _mm256_max_ps(vout2, vzero);
              vout3 = _mm256_max_ps(vout3, vzero);
              vout4 = _mm256_max_ps(vout4, vzero);
              vout0 = _mm256_min_ps(vout0, valpha);
              vout5 = _mm256_max_ps(vout5, vzero);
              vout1 = _mm256_min_ps(vout1, valpha);
              vout2 = _mm256_min_ps(vout2, valpha);
              vout3 = _mm256_min_ps(vout3, valpha);
              __mm256_store_ps(c_ptr0, vout0);
              vout4 = _mm256_min_ps(vout4, valpha);
              __mm256_store_ps(c_ptr1, vout1);
              vout5 = _mm256_min_ps(vout5, valpha);
            } else if (flag_act == 3) {  // leakyrelu
              __m256 vzero = _mm256_set1_ps(0.f);
              __m256 valpha = _mm256_set1_ps(local_alpha);
              __m256i vgt0 =
                  _mm256_cmpgt_epi32(_mm256_castps_si256(vout0), vzero);
              __m256 vsum0 = _mm256_mul_ps(vout0, valpha);
              __m256i vgt1 =
                  _mm256_cmpgt_epi32(_mm256_castps_si256(vout1), vzero);
              __m256 vsum1 = _mm256_mul_ps(vout1, valpha);
              __m256i vgt2 =
                  _mm256_cmpgt_epi32(_mm256_castps_si256(vout2), vzero);
              __m256 vsum2 = _mm256_mul_ps(vout2, valpha);
              __m256i vgt3 =
                  _mm256_cmpgt_epi32(_mm256_castps_si256(vout3), vzero);
              __m256 vsum3 = _mm256_mul_ps(vout3, valpha);
              vout0 = _mm256_blend_pd(vsum0, vout0, vgt0);
              vgt0 = _mm256_cmpgt_epi32(_mm256_castps_si256(vout4), vzero);
              vsum0 = _mm256_mul_ps(vout4, valpha);
              vout1 = _mm256_blend_pd(vsum1, vout1, vgt1);
              vgt1 = _mm256_cmpgt_epi32(_mm256_castps_si256(vout5), vzero);
              vsum1 = _mm256_mul_ps(vout5, valpha);
              vout2 = _mm256_blend_pd(vsum2, vout2, vgt2);
              __mm256_store_ps(c_ptr0, vout0);
              vout3 = _mm256_blend_pd(vsum3, vout3, vgt3);
              vout4 = _mm256_blend_pd(vsum0, vout4, vgt0);
              __mm256_store_ps(c_ptr1, vout1);
              vout5 = _mm256_blend_pd(vsum1, vout5, vgt1);
            }
            c_ptr0 += 8;
            __mm256_store_ps(c_ptr2, vout2);
            c_ptr1 += 8;
            __mm256_store_ps(c_ptr3, vout3);
            c_ptr2 += 8;
            __mm256_store_ps(c_ptr4, vout4);
            c_ptr3 += 8;
            __mm256_store_ps(c_ptr5, vout5);
            c_ptr4 += 8;
            c_ptr5 += 8;
          }
          // 6 x 4
          if (cnt_4 > 0) {
            const float *a_ptr = a_ptr_l;
            __m128 vout0 = _mm_set1_ps(bias_local[0]);
            __m128 vout1 = _mm_set1_ps(bias_local[1]);
            __m128 vout2 = _mm_set1_ps(bias_local[2]);
            __m128 vout3 = _mm_set1_ps(bias_local[3]);
            __m128 vout4 = _mm_set1_ps(bias_local[4]);
            __m128 vout5 = _mm_set1_ps(bias_local[5]);
            if (has_beta) {
              __m128 vbeta = _mm_set1_ps(beta);
              __m128 va = _mm_load_ps(c_ptr0);
              __m128 vb = _mm_load_ps(c_ptr1);
              __m128 vc = _mm_load_ps(c_ptr2);
              __m128 vd = _mm_load_ps(c_ptr3);
              __m128 ve = _mm_load_ps(c_ptr4);
              vout0 = _mm_fmadd_ps(va, vbeta, vout0);
              __m256 vf = _mm_load_ps(c_ptr5);
              vout1 = _mm_fmadd_ps(vb, vbeta, vout1);
              vout2 = _mm_fmadd_ps(vc, vbeta, vout2);
              vout3 = _mm_fmadd_ps(vd, vbeta, vout3);
              vout4 = _mm_fmadd_ps(ve, vbeta, vout4);
              vout5 = _mm_fmadd_ps(vf, vbeta, vout5);
            }
            for (int i = 0; i < K - 2; i += 2) {
              __m128 vina0 = _mm_set1_ps(a_ptr[0]);
              __m128 vinb0 = _mm_load_ps(b_ptr);
              __m128 vina1 = _mm_set1_ps(a_ptr[1]);
              __m128 vina2 = _mm_set1_ps(a_ptr[2]);
              __m128 vina3 = _mm_set1_ps(a_ptr[3]);
              __m128 vina4 = _mm_set1_ps(a_ptr[4]);
              vout0 = _mm_fmadd_ps(vina0, vinb0, vout0);
              __m128 vina5 = _mm_set1_ps(a_ptr[5]);
              vout1 = _mm_fmadd_ps(vina1, vinb0, vout1);
              __m128 vinb1 = _mm_load_ps(b_ptr + 4);
              vout2 = _mm_fmadd_ps(vina2, vinb0, vout2);
              vina0 = _mm_set1_ps(a_ptr[6]);
              vout3 = _mm_fmadd_ps(vina3, vinb0, vout3);
              vina1 = _mm_set1_ps(a_ptr[7]);
              vout4 = _mm_fmadd_ps(vina4, vinb0, vout4);
              vina2 = _mm_set1_ps(a_ptr[8]);
              vout5 = _mm_fmadd_ps(vina5, vinb0, vout5);
              vina3 = _mm_set1_ps(a_ptr[9]);
              vout0 = _mm_fmadd_ps(vina0, vinb1, vout0);
              vina4 = _mm_set1_ps(a_ptr[10]);
              vout1 = _mm_fmadd_ps(vina1, vinb1, vout1);
              vina5 = _mm_set1_ps(a_ptr[11]);
              vout2 = _mm_fmadd_ps(vina2, vinb1, vout2);
              b_ptr += 8;
              vout3 = _mm_fmadd_ps(vina3, vinb1, vout3);
              a_ptr += 12;
              vout4 = _mm_fmadd_ps(vina4, vinb1, vout4);
              vout5 = _mm_fmadd_ps(vina5, vinb1, vout5);
            }
            if (K % 2) {
              __m128 vina0 = _mm_set1_ps(a_ptr[0]);
              __m128 vinb0 = _mm_load_ps(b_ptr);
              __m128 vina1 = _mm_set1_ps(a_ptr[1]);
              __m128 vina2 = _mm_set1_ps(a_ptr[2]);
              __m128 vina3 = _mm_set1_ps(a_ptr[3]);
              __m128 vina4 = _mm_set1_ps(a_ptr[4]);
              vout0 = _mm_fmadd_ps(vina0, vinb0, vout0);
              __m128 vina5 = _mm_set1_ps(a_ptr[5]);
              vout1 = _mm_fmadd_ps(vina1, vinb0, vout1);
              a_ptr += 6;
              vout2 = _mm_fmadd_ps(vina2, vinb0, vout2);
              vout3 = _mm_fmadd_ps(vina3, vinb0, vout3);
              b_ptr += 4;
              vout4 = _mm_fmadd_ps(vina4, vinb0, vout4);
              vout5 = _mm_fmadd_ps(vina5, vinb0, vout5);
            }
            if (flag_act == 0) {  // no relu
              __mm_storeu_ps(c_ptr0, vout0);
              __mm_storeu_ps(c_ptr1, vout1);
            } else if (flag_act == 1) {  // relu
              __m128 vzero = _mm_set1_ps(0.f);
              vout0 = _mm_max_ps(vout0, vzero);
              vout1 = _mm_max_ps(vout1, vzero);
              vout2 = _mm_max_ps(vout2, vzero);
              vout3 = _mm_max_ps(vout3, vzero);
              vout4 = _mm_max_ps(vout4, vzero);
              __mm_storeu_ps(c_ptr0, vout0);
              vout5 = _mm_max_ps(vout5, vzero);
              __mm_storeu_ps(c_ptr1, vout1);
            } else if (flag_act == 2) {  // relu6
              __m128 vzero = _mm_set1_ps(0.f);
              __m128 valpha = _mm_set1_ps(local_alpha);
              vout0 = _mm_max_ps(vout0, vzero);
              vout1 = _mm_max_ps(vout1, vzero);
              vout2 = _mm_max_ps(vout2, vzero);
              vout3 = _mm_max_ps(vout3, vzero);
              vout4 = _mm_max_ps(vout4, vzero);
              vout0 = _mm_min_ps(vout0, valpha);
              vout5 = _mm_max_ps(vout5, vzero);
              vout1 = _mm_min_ps(vout1, valpha);
              vout2 = _mm_min_ps(vout2, valpha);
              vout3 = _mm_min_ps(vout3, valpha);
              __mm_storeu_ps(c_ptr0, vout0);
              vout4 = _mm_min_ps(vout4, valpha);
              __mm_storeu_ps(c_ptr1, vout1);
              vout5 = _mm_min_ps(vout5, valpha);
            } else if (flag_act == 3) {  // leakyrelu
              __m128 vzero = _mm_set1_ps(0.f);
              __m128 valpha = _mm_set1_ps(local_alpha);
              __m128i vgt0 = _mm_cmpgt_epi32(_mm_castps_si128(vout0), vzero);
              __m128 vsum0 = _mm_mul_ps(vout0, valpha);
              __m128i vgt1 = _mm_cmpgt_epi32(_mm_castps_si128(vout1), vzero);
              __m128 vsum1 = _mm_mul_ps(vout1, valpha);
              __m128i vgt2 = _mm_cmpgt_epi32(_mm_castps_si128(vout2), vzero);
              __m128 vsum2 = _mm_mul_ps(vout2, valpha);
              __m128i vgt3 = _mm_cmpgt_epi32(_mm_castps_si128(vout3), vzero);
              __m128 vsum3 = _mm_mul_ps(vout3, valpha);
              vout0 = _mm_blend_pd(vsum0, vout0, vgt0);
              vgt0 = _mm_cmpgt_epi32(_mm_castps_si128(vout4), vzero);
              vsum0 = _mm_mul_ps(vout4, valpha);
              vout1 = _mm_blend_pd(vsum1, vout1, vgt1);
              vgt1 = _mm_cmpgt_epi32(_mm_castps_si128(vout5), vzero);
              vsum1 = _mm_mul_ps(vout5, valpha);
              vout2 = _mm_blend_pd(vsum2, vout2, vgt2);
              __mm_storeu_ps(c_ptr0, vout0);
              vout3 = _mm_blend_pd(vsum3, vout3, vgt3);
              vout4 = _mm_blend_pd(vsum0, vout4, vgt0);
              __mm_storeu_ps(c_ptr1, vout1);
              vout5 = _mm_blend_pd(vsum1, vout5, vgt1);
            }
            c_ptr0 += 4;
            __mm_storeu_ps(c_ptr2, vout2);
            c_ptr1 += 4;
            __mm_storeu_ps(c_ptr3, vout3);
            c_ptr2 += 4;
            __mm_storeu_ps(c_ptr4, vout4);
            c_ptr3 += 4;
            __mm_storeu_ps(c_ptr5, vout5);
            c_ptr4 += 4;
            c_ptr5 += 4;
          }
          // 6 x 1
          for (int i = 0; i < rem_4; i++) {
            const float *a_ptr = a_ptr_l;
            __m128 vout0 = _mm_loadu_ps(bias_local);
            __m128 vout1 = __mm_loadu_ps(bias_local + 4);
            for (int i = 0; i < K - 2; i += 2) {
              __m128 vina0 = _mm_load_ps(a_ptr);
              __m128 vina1 = _mm_load_ps(a_ptr + 4);
              __m128 vinb0 = _mm_set1_ps(b_ptr[0]);
              __m128 vinb1 = _mm_set1_ps(b_ptr[1]);
              __m128 vina2 = _mm_load_ps(a_ptr + 8);
              vout0 = _mm_fmadd_ps(vina0, vinb0, vout0);
              __m128 vina3 = _mm_load_ps(a_ptr + 12);
              vout1 = _mm_fmadd_ps(vina1, vinb0, vout1);
              a_ptr += 12;
              vout0 = _mm_fmadd_ps(vina2, vinb1, vout0);
              b_ptr += 2;
              vout1 = _mm_fmadd_ps(vina3, vinb1, vout1);
            }
            if (K % 2) {
              __m128 vina0 = _mm_load_ps(a_ptr);
              __m128 vina1 = _mm_load_ps(a_ptr + 4);
              __m128 vinb0 = _mm_set1_ps(b_ptr[0]);
              vout0 = _mm_fmadd_ps(vina0, vinb0, vout0);
              vout1 = _mm_fmadd_ps(vina1, vinb0, vout1);
              a_ptr += 6;
              b_ptr++;
            }
            if (flag_act == 1) {  // relu
              __m128 vzero = _mm_set1_ps(0.f);
              vout0 = _mm_max_ps(vout0, vzero);
              vout1 = _mm_max_ps(vout1, vzero);
            } else if (flag_act == 2) {  // relu6
              __m128 vzero = _mm_set1_ps(0.f);
              __m128 valpha = _mm_set1_ps(local_alpha);
              vout0 = _mm_max_ps(vout0, vzero);
              vout1 = _mm_max_ps(vout1, vzero);
              vout0 = _mm_min_ps(vout0, valpha);
              vout1 = _mm_min_ps(vout1, valpha);
            } else if (flag_act == 3) {  // leakyrelu
              __m128 vzero = _mm_set1_ps(0.f);
              __m128 valpha = _mm_set1_ps(local_alpha);
              __m128i vgt0 = _mm_cmpgt_epi32(_mm_castps_si128(vout0), vzero);
              __m128 vsum0 = _mm_mul_ps(vout0, valpha);
              __m128i vgt1 = _mm_cmpgt_epi32(_mm_castps_si128(vout1), vzero);
              __m128 vsum1 = _mm_mul_ps(vout1, valpha);
              vout0 = _mm_blend_pd(vsum0, vout0, vgt0);
              vout1 = _mm_blend_pd(vsum1, vout1, vgt1);
            }
            if (has_beta) {
              *c_ptr0++ =
                  (reinterpret_cast<float *>(&vout0))[0] + c_ptr0[0] * beta;
              *c_ptr1++ =
                  (reinterpret_cast<float *>(&vout0))[1] + c_ptr1[0] * beta;
              *c_ptr2++ =
                  (reinterpret_cast<float *>(&vout0))[2] + c_ptr2[0] * beta;
              *c_ptr3++ =
                  (reinterpret_cast<float *>(&vout0))[3] + c_ptr3[0] * beta;
              *c_ptr4++ =
                  (reinterpret_cast<float *>(&vout1))[0] + c_ptr4[0] * beta;
              *c_ptr5++ =
                  (reinterpret_cast<float *>(&vout1))[1] + c_ptr5[0] * beta;
            } else {
              *c_ptr0++ = (reinterpret_cast<float *>(&vout0))[0];
              *c_ptr1++ = (reinterpret_cast<float *>(&vout0))[1];
              *c_ptr2++ = (reinterpret_cast<float *>(&vout0))[2];
              *c_ptr3++ = (reinterpret_cast<float *>(&vout0))[3];
              *c_ptr4++ = (reinterpret_cast<float *>(&vout1))[0];
              *c_ptr5++ = (reinterpret_cast<float *>(&vout1))[1];
            }
          }
        } else {
          const float *a_ptr = a_ptr_l;
          // 6 x 16
          __m256 vout0 = _mm256_set1_ps(bias_local[0]);
          __m256 vout1 = _mm256_set1_ps(bias_local[1]);
          __m256 vout2 = _mm256_set1_ps(bias_local[2]);
          __m256 vout3 = _mm256_set1_ps(bias_local[3]);
          __m256 vout4 = _mm256_set1_ps(bias_local[4]);
          __m256 vout5 = _mm256_set1_ps(bias_local[5]);
          __m256 vout01 = _mm256_set1_ps(bias_local[0]);
          __m256 vout11 = _mm256_set1_ps(bias_local[1]);
          __m256 vout21 = _mm256_set1_ps(bias_local[2]);
          __m256 vout31 = _mm256_set1_ps(bias_local[3]);
          __m256 vout41 = _mm256_set1_ps(bias_local[4]);
          __m256 vout51 = _mm256_set1_ps(bias_local[5]);
          if (has_beta) {
            __m256 vbeta = _mm256_set1_ps(beta);
            __m256 va = _mm256_load_ps(c_ptr0);
            __m256 vb = _mm256_load_ps(c_ptr1);
            __m256 vc = _mm256_load_ps(c_ptr2);
            __m256 vd = _mm256_load_ps(c_ptr3);
            __m256 ve = _mm256_load_ps(c_ptr4);
            vout0 = _mm256_fmadd_ps(va, vbeta, vout0);
            __m256 vf = _mm256_load_ps(c_ptr5);
            vout1 = _mm256_fmadd_ps(vb, vbeta, vout1);
            va = _mm256_load_ps(c_ptr0 + 8);
            vout2 = _mm256_fmadd_ps(vc, vbeta, vout2);
            vb = _mm256_load_ps(c_ptr1 + 8);
            vout3 = _mm256_fmadd_ps(vd, vbeta, vout3);
            vc = _mm256_load_ps(c_ptr2 + 8);
            vout4 = _mm256_fmadd_ps(ve, vbeta, vout4);
            vd = _mm256_load_ps(c_ptr3 + 8);
            vout5 = _mm256_fmadd_ps(vf, vbeta, vout5);
            ve = _mm256_load_ps(c_ptr4 + 8);
            vout01 = _mm256_fmadd_ps(va, vbeta, vout01);
            vf = _mm256_load_ps(c_ptr5 + 8);
            vout11 = _mm256_fmadd_ps(vb, vbeta, vout11);
            vout21 = _mm256_fmadd_ps(vc, vbeta, vout21);
            vout31 = _mm256_fmadd_ps(vd, vbeta, vout31);
            vout41 = _mm256_fmadd_ps(ve, vbeta, vout41);
            vout51 = _mm256_fmadd_ps(vf, vbeta, vout51);
          }
          for (int i = 0; i < K - 2; i += 2) {
            __m256 vina0 = _mm256_set1_ps(a_ptr[0]);
            __m256 vinb0 = _mm256_load_ps(b_ptr);
            __m256 vinb1 = _mm256_load_ps(b_ptr + 8);
            __m256 vina1 = _mm256_set1_ps(a_ptr[1]);
            b_ptr += 16;
            vout0 = _mm256_fmadd_ps(vina0, vinb0, vout0);
            vout01 = _mm256_fmadd_ps(vina0, vinb1, vout01);
            vina0 = _mm256_set1_ps(a_ptr[2]);
            vout1 = _mm256_fmadd_ps(vina1, vinb0, vout1);
            vout11 = _mm256_fmadd_ps(vina1, vinb1, vout11);
            vina1 = _mm256_set1_ps(a_ptr[3]);
            vout2 = _mm256_fmadd_ps(vina0, vinb0, vout2);
            vout21 = _mm256_fmadd_ps(vina0, vinb1, vout21);
            vina0 = _mm256_set1_ps(a_ptr[4]);
            vout3 = _mm256_fmadd_ps(vina1, vinb0, vout3);
            vout31 = _mm256_fmadd_ps(vina1, vinb1, vout31);
            vina1 = _mm256_set1_ps(a_ptr[5]);
            vout4 = _mm256_fmadd_ps(vina0, vinb0, vout4);
            vout41 = _mm256_fmadd_ps(vina0, vinb1, vout41);
            vina0 = _mm256_set1_ps(a_ptr[6]);
            vout5 = _mm256_fmadd_ps(vina1, vinb0, vout5);
            vinb0 = _mm256_load_ps(b_ptr);
            vout51 = _mm256_fmadd_ps(vina1, vinb1, vout51);
            vinb1 = _mm256_load_ps(b_ptr + 8);
            vina1 = _mm256_set1_ps(a_ptr[7]);
            b_ptr += 16;
            vout0 = _mm256_fmadd_ps(vina0, vinb0, vout0);
            vout01 = _mm256_fmadd_ps(vina0, vinb1, vout01);
            vina0 = _mm256_set1_ps(a_ptr[8]);
            vout1 = _mm256_fmadd_ps(vina1, vinb0, vout1);
            vout11 = _mm256_fmadd_ps(vina1, vinb1, vout11);
            vina1 = _mm256_set1_ps(a_ptr[9]);
            vout2 = _mm256_fmadd_ps(vina0, vinb0, vout2);
            vout21 = _mm256_fmadd_ps(vina0, vinb1, vout21);
            vina0 = _mm256_set1_ps(a_ptr[10]);
            vout3 = _mm256_fmadd_ps(vina1, vinb0, vout3);
            vout31 = _mm256_fmadd_ps(vina1, vinb1, vout31);
            vina1 = _mm256_set1_ps(a_ptr[5]);
            vout4 = _mm256_fmadd_ps(vina0, vinb0, vout4);
            vout41 = _mm256_fmadd_ps(vina0, vinb1, vout41);
            a_ptr += 12;
            vout5 = _mm256_fmadd_ps(vina1, vinb0, vout5);
            vout51 = _mm256_fmadd_ps(vina1, vinb0, vout51);
          }
          if (K % 2) {
            __m256 vina0 = _mm256_set1_ps(a_ptr[0]);
            __m256 vinb0 = _mm256_load_ps(b_ptr);
            __m256 vinb1 = _mm256_load_ps(b_ptr + 8);
            __m256 vina1 = _mm256_set1_ps(a_ptr[1]);
            b_ptr += 16;
            vout0 = _mm256_fmadd_ps(vina0, vinb0, vout0);
            vout01 = _mm256_fmadd_ps(vina0, vinb1, vout01);
            vina0 = _mm256_set1_ps(a_ptr[2]);
            vout1 = _mm256_fmadd_ps(vina1, vinb0, vout1);
            vout11 = _mm256_fmadd_ps(vina1, vinb1, vout11);
            vina1 = _mm256_set1_ps(a_ptr[3]);
            vout2 = _mm256_fmadd_ps(vina0, vinb0, vout2);
            vout21 = _mm256_fmadd_ps(vina0, vinb1, vout21);
            vina0 = _mm256_set1_ps(a_ptr[4]);
            vout3 = _mm256_fmadd_ps(vina1, vinb0, vout3);
            vout31 = _mm256_fmadd_ps(vina1, vinb1, vout31);
            vina1 = _mm256_set1_ps(a_ptr[5]);
            vout4 = _mm256_fmadd_ps(vina0, vinb0, vout4);
            vout41 = _mm256_fmadd_ps(vina0, vinb1, vout41);
            a_ptr += 6;
            vout5 = _mm256_fmadd_ps(vina0, vinb0, vout5);
            vout51 = _mm256_fmadd_ps(vina0, vinb1, vout51);
          }
          if (flag_act == 0) {  // no relu
            __mm256_store_ps(c_ptr0, vout0);
            __mm256_store_ps(c_ptr1, vout1);
            __mm256_store_ps(c_ptr2, vout2);
            __mm256_store_ps(c_ptr3, vout3);
            __mm256_store_ps(c_ptr4, vout4);
            __mm256_store_ps(c_ptr5, vout5);
            __mm256_store_ps(c_ptr0 + 8, vout01);
            __mm256_store_ps(c_ptr1 + 8, vout11);
            __mm256_store_ps(c_ptr2 + 8, vout21);
          } else if (flag_act == 1) {  // relu
            __m256 vzero = _mm256_set1_ps(0.f);
            vout0 = _mm256_max_ps(vout0, vzero);
            vout1 = _mm256_max_ps(vout1, vzero);
            vout2 = _mm256_max_ps(vout2, vzero);
            vout3 = _mm256_max_ps(vout3, vzero);
            vout4 = _mm256_max_ps(vout4, vzero);
            __mm256_store_ps(c_ptr0, vout0);
            vout5 = _mm256_max_ps(vout5, vzero);
            __mm256_store_ps(c_ptr1, vout1);
            vout01 = _mm256_max_ps(vout01, vzero);
            __mm256_store_ps(c_ptr2, vout2);
            vout11 = _mm256_max_ps(vout11, vzero);
            __mm256_store_ps(c_ptr3, vout3);
            vout21 = _mm256_max_ps(vout21, vzero);
            __mm256_store_ps(c_ptr4, vout4);
            vout31 = _mm256_max_ps(vout31, vzero);
            __mm256_store_ps(c_ptr5, vout5);
            vout41 = _mm256_max_ps(vout41, vzero);
            __mm256_store_ps(c_ptr0 + 8, vout01);
            vout51 = _mm256_max_ps(vout51, vzero);
            __mm256_store_ps(c_ptr1 + 8, vout11);
            __mm256_store_ps(c_ptr2 + 8, vout21);
          } else if (flag_act == 2) {  // relu6
            __m256 vzero = _mm256_set1_ps(0.f);
            __m256 valpha = _mm256_set1_ps(local_alpha);
            vout0 = _mm256_max_ps(vout0, vzero);
            vout1 = _mm256_max_ps(vout1, vzero);
            vout2 = _mm256_max_ps(vout2, vzero);
            vout3 = _mm256_max_ps(vout3, vzero);
            vout4 = _mm256_max_ps(vout4, vzero);
            vout5 = _mm256_max_ps(vout5, vzero);
            vout0 = _mm256_min_ps(vout0, valpha);
            vout01 = _mm256_max_ps(vout01, vzero);
            vout1 = _mm256_min_ps(vout1, valpha);
            vout11 = _mm256_max_ps(vout11, vzero);
            vout2 = _mm256_min_ps(vout2, valpha);
            vout21 = _mm256_max_ps(vout21, vzero);
            vout3 = _mm256_min_ps(vout3, valpha);
            vout31 = _mm256_max_ps(vout31, vzero);
            vout4 = _mm256_min_ps(vout4, valpha);
            vout41 = _mm256_max_ps(vout41, vzero);
            vout5 = _mm256_min_ps(vout5, valpha);
            vout51 = _mm256_max_ps(vout51, vzero);
            __mm256_store_ps(c_ptr0, vout0);
            vout01 = _mm256_min_ps(vout01, valpha);
            __mm256_store_ps(c_ptr1, vout1);
            vout11 = _mm256_min_ps(vout11, valpha);
            __mm256_store_ps(c_ptr2, vout2);
            vout21 = _mm256_min_ps(vout21, valpha);
            __mm256_store_ps(c_ptr3, vout3);
            vout31 = _mm256_min_ps(vout31, valpha);
            __mm256_store_ps(c_ptr4, vout4);
            vout41 = _mm256_min_ps(vout41, valpha);
            __mm256_store_ps(c_ptr5, vout5);
            vout51 = _mm256_min_ps(vout51, valpha);
            __mm256_store_ps(c_ptr0 + 8, vout01);
            __mm256_store_ps(c_ptr1 + 8, vout11);
            __mm256_store_ps(c_ptr2 + 8, vout21);
          } else if (flag_act == 3) {  // leakyrelu
            __m256 vzero = _mm256_set1_ps(0.f);
            __m256 valpha = _mm256_set1_ps(local_alpha);
            __m256i vgt0 =
                _mm256_cmpgt_epi32(_mm256_castps_si256(vout0), vzero);
            __m256 vsum0 = _mm256_mul_ps(vout0, valpha);
            vout0 = _mm256_blend_pd(vsum0, vout0, vgt0);
            vgt0 = _mm256_cmpgt_epi32(_mm256_castps_si256(vout1), vzero);
            vsum0 = _mm256_mul_ps(vout1, valpha);
            vout1 = _mm256_blend_pd(vsum0, vout1, vgt0);
            vgt0 = _mm256_cmpgt_epi32(_mm256_castps_si256(vout2), vzero);
            vsum0 = _mm256_mul_ps(vout2, valpha);
            __mm256_store_ps(c_ptr0, vout0);
            vout2 = _mm256_blend_pd(vsum0, vout2, vgt0);
            vgt0 = _mm256_cmpgt_epi32(_mm256_castps_si256(vout3), vzero);
            vsum0 = _mm256_mul_ps(vout3, valpha);
            __mm256_store_ps(c_ptr1, vout1);
            vout3 = _mm256_blend_pd(vsum0, vout3, vgt0);
            vgt0 = _mm256_cmpgt_epi32(_mm256_castps_si256(vout4), vzero);
            vsum0 = _mm256_mul_ps(vout4, valpha);
            __mm256_store_ps(c_ptr2, vout2);
            vout4 = _mm256_blend_pd(vsum0, vout4, vgt0);
            vgt0 = _mm256_cmpgt_epi32(_mm256_castps_si256(vout5), vzero);
            vsum0 = _mm256_mul_ps(vout5, valpha);
            __mm256_store_ps(c_ptr3, vout3);
            vout5 = _mm256_blend_pd(vsum0, vout5, vgt0);
            vgt0 = _mm256_cmpgt_epi32(_mm256_castps_si256(vout01), vzero);
            vsum0 = _mm256_mul_ps(vout01, valpha);
            __mm256_store_ps(c_ptr4, vout4);
            vout01 = _mm256_blend_pd(vsum0, vout01, vgt0);
            vgt0 = _mm256_cmpgt_epi32(_mm256_castps_si256(vout01), vzero);
            vsum0 = _mm256_mul_ps(vout11, valpha);
            __mm256_store_ps(c_ptr5, vout5);
            vout2 = _mm256_mul_ps(vout21, valpha);
            vout3 = _mm256_mul_ps(vout31, valpha);
            vout11 = _mm256_blend_pd(vsum0, vout11, vgt0);
            vgt0 = _mm256_cmpgt_epi32(_mm256_castps_si256(vout21), vzero);
            __mm256_store_ps(c_ptr0 + 8, vout01);
            vout4 = _mm256_mul_ps(vout41, valpha);
            vout5 = _mm256_mul_ps(vout51, valpha);
            vout21 = _mm256_blend_pd(vout2, vout21, vgt0);
            vgt0 = _mm256_cmpgt_epi32(_mm256_castps_si256(vout31), vzero);
            __mm256_store_ps(c_ptr1 + 8, vout11);
            __m256i vgt1 =
                mm256_cmpgt_epi32(_mm256_castps_si256(vout41), vzero);
            __m256i vgt2 =
                mm256_cmpgt_epi32(_mm256_castps_si256(vout51), vzero);
            __mm256_store_ps(c_ptr2 + 8, vout21);
            vout31 = _mm256_blend_pd(vout3, vout31, vgt0);
            vout41 = _mm256_blend_pd(vout4, vout41, vgt1);
            vout51 = _mm256_blend_pd(vout5, vout51, vgt2);
          }
          __mm256_store_ps(c_ptr3 + 8, vout31);
          c_ptr0 += 16;
          __mm256_store_ps(c_ptr4 + 8, vout41);
          c_ptr1 += 16;
          __mm256_store_ps(c_ptr5 + 8, vout51);
          c_ptr2 += 16;
          c_ptr3 += 16;
          c_ptr4 += 16;
          c_ptr5 += 16;
        }
      }
    }
  }
}
#else
/**
 * @brief gemm compute of 4x4-unit
 * input A_packed: M * K, format M / 4 * K * 4
 * input B: K * N
 * output C: M * N
 * parameter beta: C = beta * C + A_packed * B
 * parameter bias: if has_bias then C = beta * C + A_packed * B + bias
 * parameter act_param: acitvation proccess
 */
void gemm_prepack_4x4(bool is_transB,
                      int M,
                      int N,
                      int K,
                      const float *A_packed,
                      const float *B,
                      int ldb,
                      float beta,
                      float *C,
                      int ldc,
                      const float *bias,
                      bool has_bias,
                      const operators::ActivationParam act_param,
                      ARMContext *ctx) {
  size_t llc_size = ctx->llc_size() > 0 ? ctx->llc_size() : 512 * 1024;
  // auto workspace = ctx->workspace_data<float16_t>();
  int threads = ctx->threads();
  llc_size = llc_size * 9 / 10;

  auto act_type = act_param.active_type;
  float local_alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    act_acquire(act_type,
                flag_act,
                local_alpha,
                act_param.Relu_clipped_coef,
                act_param.Leaky_relu_alpha);
  }
  //! MBLOCK * x (result) + MBLOCK * k (A) + x * k (B) = l2
  X_BLOCK_COMPUTE(llc_size, MBLOCK, NBLOCK, KBLOCK, beta)
  auto b_size = (x_block + 15) / 16 * K;
  // (x_block + 15) / 16 * K
  auto workspace =
      static_cast<float *>(TargetMalloc(TARGET(kx86), b_size * sizeof(float)));
  // memset(workspace, 0, b_size * sizeof(float));

  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    if (xmax > N) {
      xmax = N;
    }
    int bblocks = (xmax - x0 + NBLOCK - 1) / NBLOCK;
    remain = xmax - x0 - (bblocks - 1) * NBLOCK;
    if (remain > 0 && remain != 16) {
      flag_p_remain = true;
    }
    //! load bpanel
    float *b_pannel = workspace;
    if (is_transB) {
      loadb_trans(b_pannel, B, ldb, 0, K, x0, xmax);
    } else {
      loadb(b_pannel, B, ldb, 0, K, x0, xmax);
    }
#pragma omp parallel for num_threads(threads)
    for (unsigned int y = 0; y < M; y += MBLOCK) {
      unsigned int ymax = y + MBLOCK;
      if (ymax > M) {
        ymax = M;
      }

      float bias_local[4] = {0};
      if (has_bias) {
        int k = 0;
        for (int i = y; i < ymax; i++) {
          bias_local[k] = bias[y + k];
          k++;
        }
        for (; k < 4; k++) {
          bias_local[k] = 0.f;
        }
      }
      // prepare out data
      float cout0[NBLOCK];
      float cout1[NBLOCK];
      float cout2[NBLOCK];
      float cout3[NBLOCK];

      float *c_ptr0 = C + y * ldc + x0;
      float *c_ptr1 = c_ptr0 + ldc;
      float *c_ptr2 = c_ptr1 + ldc;
      float *c_ptr3 = c_ptr2 + ldc;

      const float *a_ptr_l = A_packed + y * K;
      const float *b_ptr = b_pannel;
      for (int xb = 0; xb < bblocks; xb++) {
        if ((y + 3) >= ymax) {
          switch ((y + 3) - ymax) {
            case 2:
              c_ptr1 = cout1;
            case 1:
              c_ptr2 = cout2;
            case 0:
              c_ptr3 = cout3;
            default:
              break;
          }
        }
        if (flag_p_remain && (xb == bblocks - 1)) {
          // 4 x 1
          for (int i = 0; i < remain; i++) {
            const float *a_ptr = a_ptr_l;
            __m128 vout0 = _mm_loadu_ps(bias_local);
            for (int i = 0; i < K - 2; i += 2) {
              __m128 vina0 = _mm_load_ps(a_ptr);
              __m128 vinb0 = _mm_set1_ps(b_ptr[0]);
              __m128 vina1 = _mm_load_ps(a_ptr + 4);
              __m128 vinb1 = _mm_set1_ps(b_ptr[1]);
              vout0 = _mm_add_ps(_mm_mul_ps(vina0, vinb0), vout0);
              a_ptr += 8;
              b_ptr += 2;
              vout0 = _mm_add_ps(_mm_mul_ps(vina1, vinb1), vout0);
            }
            if (K % 2) {
              __m128 vina0 = _mm_load_ps(a_ptr);
              __m128 vinb0 = _mm_set1_ps(b_ptr[0]);
              a_ptr += 4;
              b_ptr++;
              vout0 = _mm_add_ps(_mm_mul_ps(vina0, vinb0), vout0);
            }
            if (flag_act == 1) {  // relu
              __m128 vzero = _mm_set1_ps(0.f);
              vout0 = _mm_max_ps(vout0, vzero);
            } else if (flag_act == 2) {  // relu6
              __m128 vzero = _mm_set1_ps(0.f);
              __m128 valpha = _mm_set1_ps(local_alpha);
              vout0 = _mm_max_ps(vout0, vzero);
              vout0 = _mm_min_ps(vout0, valpha);
            } else if (flag_act == 3) {  // leakyrelu
              __m128 vzero = _mm_set1_ps(0.f);
              __m128 valpha = _mm_set1_ps(local_alpha);
              __m128i vgt0 = _mm_cmpgt_epi32(_mm_castps_si128(vout0), vzero);
              __m128 vsum0 = _mm_mul_ps(vout0, valpha);
              vout0 = _mm_blend_pd(vsum0, vout0, vgt0);
            }
            if (has_beta) {
              *c_ptr0++ =
                  (reinterpret_cast<float *>(&vout0))[0] + c_ptr0[0] * beta;
              *c_ptr1++ =
                  (reinterpret_cast<float *>(&vout0))[1] + c_ptr1[0] * beta;
              *c_ptr2++ =
                  (reinterpret_cast<float *>(&vout0))[2] + c_ptr2[0] * beta;
              *c_ptr3++ =
                  (reinterpret_cast<float *>(&vout0))[3] + c_ptr3[0] * beta;
            } else {
              *c_ptr0++ = (reinterpret_cast<float *>(&vout0))[0];
              *c_ptr1++ = (reinterpret_cast<float *>(&vout0))[1];
              *c_ptr2++ = (reinterpret_cast<float *>(&vout0))[2];
              *c_ptr3++ = (reinterpret_cast<float *>(&vout0))[3];
            }
          }
        } else {
          // 4 x 4
          const float *a_ptr = a_ptr_l;
          __m128 vout0 = _mm_set1_ps(bias_local[0]);
          __m128 vout1 = _mm_set1_ps(bias_local[1]);
          __m128 vout2 = _mm_set1_ps(bias_local[2]);
          __m128 vout3 = _mm_set1_ps(bias_local[3]);
          if (has_beta) {
            __m128 vbeta = _mm_set1_ps(beta);
            __m128 va = _mm_load_ps(c_ptr0);
            __m128 vb = _mm_load_ps(c_ptr1);
            __m128 vc = _mm_load_ps(c_ptr2);
            vout0 = _mm_fmadd_ps(va, vbeta, vout0);
            __m128 va = _mm_load_ps(c_ptr3);
            vout1 = _mm_fmadd_ps(vb, vbeta, vout1);
            vout2 = _mm_fmadd_ps(vc, vbeta, vout2);
            vout3 = _mm_fmadd_ps(va, vbeta, vout3);
          }
          for (int i = 0; i < K - 2; i += 2) {
            __m128 vina0 = _mm_set1_ps(a_ptr[0]);
            __m128 vinb0 = _mm_load_ps(b_ptr);
            __m128 vina1 = _mm_set1_ps(a_ptr[1]);
            __m128 vina2 = _mm_set1_ps(a_ptr[2]);
            vout0 = _mm_fmadd_ps(vina0, vinb0, vout0);
            vina0 = _mm_set1_ps(a_ptr[3]);
            vout1 = _mm_fmadd_ps(vina1, vinb0, vout1);
            vina1 = _mm_set1_ps(a_ptr[4]);
            vout2 = _mm_fmadd_ps(vina2, vinb0, vout2);
            vina2 = _mm_set1_ps(a_ptr[5]);
            vout3 = _mm_fmadd_ps(vina0, vinb0, vout3);
            vinb0 = _mm_load_ps(b_ptr + 4);
            vina0 = _mm_set1_ps(a_ptr[6]);
            b_ptr += 8;
            vout0 = _mm_fmadd_ps(vina1, vinb0, vout0);
            vina1 = _mm_set1_ps(a_ptr[7]);
            vout1 = _mm_fmadd_ps(vina2, vinb0, vout1);
            vout2 = _mm_fmadd_ps(vina0, vinb0, vout2);
            a_ptr += 8;
            vout2 = _mm_fmadd_ps(vina1, vinb0, vout3);
          }
          if (K % 2) {
            __m128 vina0 = _mm_set1_ps(a_ptr[0]);
            __m128 vinb0 = _mm_load_ps(b_ptr);
            __m128 vina1 = _mm_set1_ps(a_ptr[1]);
            __m128 vina2 = _mm_set1_ps(a_ptr[2]);
            b_ptr += 4;
            vout0 = _mm_fmadd_ps(vina0, vinb0, vout0);
            __m128 vina0 = _mm_set1_ps(a_ptr[3]);
            vout1 = _mm_fmadd_ps(vina1, vinb0, vout1);
            vout2 = _mm_fmadd_ps(vina2, vinb0, vout2);
            a_ptr += 4;
            vout3 = _mm_fmadd_ps(vina0, vinb0, vout3);
          }
          if (flag_act == 0) {  // no relu
            __mm_storeu_ps(c_ptr0, vout0);
            __mm_storeu_ps(c_ptr1, vout1);
          } else if (flag_act == 1) {  // relu
            __m128 vzero = _mm_set1_ps(0.f);
            vout0 = _mm_max_ps(vout0, vzero);
            vout1 = _mm_max_ps(vout1, vzero);
            vout2 = _mm_max_ps(vout2, vzero);
            vout3 = _mm_max_ps(vout3, vzero);
            __mm_storeu_ps(c_ptr0, vout0);
            __mm_storeu_ps(c_ptr1, vout1);
          } else if (flag_act == 2) {  // relu6
            __m128 vzero = _mm_set1_ps(0.f);
            __m128 valpha = _mm_set1_ps(local_alpha);
            vout0 = _mm_max_ps(vout0, vzero);
            vout1 = _mm_max_ps(vout1, vzero);
            vout2 = _mm_max_ps(vout2, vzero);
            vout3 = _mm_max_ps(vout3, vzero);
            vout0 = _mm_min_ps(vout0, valpha);
            vout1 = _mm_min_ps(vout1, valpha);
            vout2 = _mm_min_ps(vout2, valpha);
            vout3 = _mm_min_ps(vout3, valpha);
            __mm_storeu_ps(c_ptr0, vout0);
            __mm_storeu_ps(c_ptr1, vout1);
          } else if (flag_act == 3) {  // leakyrelu
            __m128 vzero = _mm_set1_ps(0.f);
            __m128 valpha = _mm_set1_ps(local_alpha);
            __m128i vgt0 = _mm_cmpgt_epi32(_mm_castps_si128(vout0), vzero);
            __m128 vsum0 = _mm_mul_ps(vout0, valpha);
            vout0 = _mm_blend_pd(vsum0, vout0, vgt0);
            vgt0 = _mm_cmpgt_epi32(_mm_castps_si128(vout1), vzero);
            vsum0 = _mm_mul_ps(vout1, valpha);
            vout1 = _mm_blend_pd(vsum0, vout1, vgt0);
            vgt0 = _mm_cmpgt_epi32(_mm_castps_si128(vout2), vzero);
            vsum0 = _mm_mul_ps(vout2, valpha);
            __mm_storeu_ps(c_ptr0, vout0);
            vout2 = _mm_blend_pd(vsum0, vout2, vgt0);
            vgt0 = _mm_cmpgt_epi32(_mm_castps_si128(vout3), vzero);
            vsum0 = _mm_mul_ps(vout3, valpha);
            __mm_storeu_ps(c_ptr1, vout1);
            vout3 = _mm_blend_pd(vsum0, vout3, vgt0);
          }
          c_ptr0 += 4;
          __mm_storeu_ps(c_ptr2, vout2);
          c_ptr1 += 4;
          __mm_storeu_ps(c_ptr3, vout3);
          c_ptr2 += 4;
          c_ptr3 += 4;
        }
      }
    }
  }
}
#endif
}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
