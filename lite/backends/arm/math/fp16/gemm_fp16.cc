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

#include "lite/backends/arm/math/fp16/gemm_fp16.h"
#include <arm_neon.h>
namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
#ifdef __aarch64__
void prepackA_8x16(float16_t *out,
                   const float16_t *in,
                   float16_t alpha,
                   const int ldin,
                   const int m0,
                   const int mmax,
                   const int k0,
                   const int kmax);
void prepackA_trans_8x16(float16_t *out,
                         const float16_t *in,
                         float16_t alpha,
                         const int ldin,
                         const int m0,
                         const int mmax,
                         const int k0,
                         const int kmax);
void gemm_prepack_8x16(bool is_transB,
                       int M,
                       int N,
                       int K,
                       const float16_t *A_packed,
                       const float16_t *B,
                       int ldb,
                       float16_t beta,
                       float16_t *C,
                       int ldc,
                       const float16_t *bias,
                       bool has_bias,
                       const operators::ActivationParam act_param,
                       ARMContext *ctx);
#endif

/**
 * \brief input data is not transpose
 * for arm-v7a, transform data to block x k x 6 layout
 * for arm-v8a, transform data to block x k x 8 layout
 */
void prepackA_fp16(void *out,
                   const void *in,
                   float16_t alpha,
                   const int ldin,
                   const int m0,
                   const int mmax,
                   const int k0,
                   const int kmax,
                   bool is_trans,
                   ARMContext *ctx) {
#ifdef __aarch64__
#define PREPACKA_PARAMS                                                     \
  static_cast<float16_t *>(out), static_cast<const float16_t *>(in), alpha, \
      ldin, m0, mmax, k0, kmax
  if (is_trans) {
    prepackA_trans_8x16(PREPACKA_PARAMS);
  } else {
    prepackA_8x16(PREPACKA_PARAMS);
  }
#else
#endif
}

void prepackA_fp16(TensorLite *tout,
                   const TensorLite &tin,
                   float16_t alpha,
                   int m,
                   int k,
                   int group,
                   bool is_trans,
                   ARMContext *ctx) {
  int hblock = get_hblock_fp16(ctx);
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
    const float16_t *weights_group = tin.data<float16_t>() + g * m * k;
    float16_t *weights_trans_ptr =
        tout->mutable_data<float16_t>() + g * group_size_round_up;
    prepackA_fp16(weights_trans_ptr,
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

/// a: m*k  b: k*n  c: m*n
void gemm_prepack_fp16(bool is_transB,
                       int M,
                       int N,
                       int K,
                       const float16_t *A_packed,
                       const float16_t *B,
                       int ldb,
                       float16_t beta,
                       float16_t *C,
                       int ldc,
                       const float16_t *bias,
                       bool has_bias,
                       const operators::ActivationParam act_param,
                       ARMContext *ctx) {
#ifdef __aarch64__
  gemm_prepack_8x16(is_transB,
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
#else   // armv7
#endif  // arm64
}
#ifdef __aarch64__
#define TRANS_C8                                                   \
  /* g0h0g2h2g4h4g6h6 */                                           \
  "trn1 v14.8h, v6.8h, v7.8h            \n"                        \
  "trn2 v15.8h, v6.8h, v7.8h            \n" /* a0b0c0d0a4b4c4d4 */ \
  "trn1 v0.4s, v8.4s, v10.4s            \n" /* a2b2c2d2a6b6c6d6 */ \
  "trn2 v1.4s, v8.4s, v10.4s            \n" /* a1b1c1d1a5b5c5d5 */ \
  "trn1 v2.4s, v9.4s, v11.4s            \n" /* a3b3c3d3a7b7c7d7 */ \
  "trn2 v3.4s, v9.4s, v11.4s            \n" /* e0f0g0h0a4b4c4d4 */ \
  "trn1 v4.4s, v12.4s, v14.4s           \n" /* e2f2g2h2a6b6c6d6 */ \
  "trn2 v5.4s, v12.4s, v14.4s           \n" /* e1f1g1h1a5b5c5d5 */ \
  "trn1 v6.4s, v13.4s, v15.4s           \n" /* e3f3g3h3a7b7c7d7 */ \
  "trn2 v7.4s, v13.4s, v15.4s           \n" /* 0-4 */              \
  "trn1 v8.2d, v0.2d, v4.2d             \n"                        \
  "trn2 v9.2d, v0.2d, v4.2d             \n" /* 2-6 */              \
  "trn1 v10.2d, v1.2d, v5.2d            \n"                        \
  "trn2 v11.2d, v1.2d, v5.2d            \n" /* 1-5 */              \
  "trn1 v12.2d, v2.2d, v6.2d            \n"                        \
  "trn2 v13.2d, v2.2d, v6.2d            \n" /* 3-7 */              \
  "trn1 v14.2d, v3.2d, v7.2d            \n"                        \
  "trn2 v15.2d, v3.2d, v7.2d            \n"

#endif
#ifdef __aarch64__
void prepackA_8x16(float16_t *out,
                   const float16_t *in,
                   float16_t alpha,
                   const int ldin,
                   const int m0,
                   const int mmax,
                   const int k0,
                   const int kmax) {
  int x_len = kmax - k0;
  float16_t zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(float16_t) * x_len);

  float16_t *dout = out;
  const float16_t *inptr = in;
  bool has_alpha = fabsf(alpha - 1.f) > 1e-8f;

  int cnt = x_len >> 3;
  int remain = x_len & 7;
  int cnt_4 = remain >> 2;
  remain = remain & 3;
  float16x8_t valpha = vdupq_n_f16(alpha);
#pragma omp parallel for
  for (int y = m0; y < mmax; y += 8) {
    float16_t *outptr = dout + (y - m0) * x_len;
    const float16_t *inptr0 = inptr + y * ldin + k0;
    const float16_t *inptr1 = inptr0 + ldin;
    const float16_t *inptr2 = inptr1 + ldin;
    const float16_t *inptr3 = inptr2 + ldin;
    const float16_t *inptr4 = inptr3 + ldin;
    const float16_t *inptr5 = inptr4 + ldin;
    const float16_t *inptr6 = inptr5 + ldin;
    const float16_t *inptr7 = inptr6 + ldin;
    if ((y + 7) >= mmax) {
      ptr_acquire_a8<float16_t>(zerobuff,
                                inptr1,
                                inptr2,
                                inptr3,
                                inptr4,
                                inptr5,
                                inptr6,
                                inptr7,
                                (y + 7),
                                mmax);
    }
    int cnt_col = cnt;
    // clang-format off
    asm volatile(
        "prfm   pldl1keep, [%[inptr0]]              \n"
        "prfm   pldl1keep, [%[inptr1]]              \n"
        "prfm   pldl1keep, [%[inptr2]]              \n"
        "prfm   pldl1keep, [%[inptr3]]              \n"
        "cmp %w[cnt], #1                            \n"
        "prfm   pldl1keep, [%[inptr4]]              \n"
        "prfm   pldl1keep, [%[inptr5]]              \n"
        "prfm   pldl1keep, [%[inptr6]]              \n"
        "prfm   pldl1keep, [%[inptr7]]              \n"
        "blt 1f                                     \n"
        "0:                                         \n"
        "ld1 {v0.8h}, [%[inptr0]], #16              \n"
        "ld1 {v1.8h}, [%[inptr1]], #16              \n"
        "ld1 {v2.8h}, [%[inptr2]], #16              \n"
        "ld1 {v3.8h}, [%[inptr3]], #16              \n"
        "ld1 {v4.8h}, [%[inptr4]], #16              \n"
        // a0b0a2b2a4b4a6b6
        "trn1 v8.8h, v0.8h, v1.8h                   \n"
        "ld1 {v5.8h}, [%[inptr5]], #16              \n"
        "trn2 v9.8h, v0.8h, v1.8h                   \n"
        "ld1 {v6.8h}, [%[inptr6]], #16              \n"
        // c0d0c2d2c4d4c6d6
        "cmp %w[has_alpha], #1                      \n"
        "trn1 v10.8h, v2.8h, v3.8h                  \n"
        "ld1 {v7.8h}, [%[inptr7]], #16              \n"
        "trn2 v11.8h, v2.8h, v3.8h                  \n"
        // e0f0e2f2e4f4e6f6
        "trn1 v12.8h, v4.8h, v5.8h                  \n"
        "trn2 v13.8h, v4.8h, v5.8h                  \n"
        TRANS_C8
        "bne 10f                                    \n"
        "fmul v8.8h, v8.8h, %[valpha].8h            \n"
        "fmul v9.8h, v9.8h, %[valpha].8h            \n"
        "fmul v10.8h, v10.8h, %[valpha].8h          \n"
        "fmul v11.8h, v11.8h, %[valpha].8h          \n"
        "fmul v12.8h, v12.8h, %[valpha].8h          \n"
        "fmul v13.8h, v13.8h, %[valpha].8h          \n"
        "fmul v14.8h, v14.8h, %[valpha].8h          \n"
        "fmul v15.8h, v15.8h, %[valpha].8h          \n"
        "10:                                        \n"
        // 0
        "st1 {v8.8h}, [%[outptr]], #16              \n"
        "st1 {v12.8h}, [%[outptr]], #16             \n"
        "st1 {v10.8h}, [%[outptr]], #16             \n"
        "st1 {v14.8h}, [%[outptr]], #16             \n"
        "subs %w[cnt], %w[cnt], #1                  \n"
        "st1 {v9.8h}, [%[outptr]], #16              \n"
        "st1 {v13.8h}, [%[outptr]], #16             \n"
        "st1 {v11.8h}, [%[outptr]], #16             \n"
        "st1 {v15.8h}, [%[outptr]], #16             \n"
        "bne 0b                                     \n"
        "1:                                         \n"
        "cmp %w[cnt_4], #1                          \n"
        "blt 2f                                     \n"
        "ld1 {v0.4h}, [%[inptr0]], #8               \n"
        "ld1 {v1.4h}, [%[inptr1]], #8               \n"
        "ld1 {v2.4h}, [%[inptr2]], #8               \n"
        "ld1 {v3.4h}, [%[inptr3]], #8               \n"
        "ld1 {v4.4h}, [%[inptr4]], #8               \n"
        // a0b0a2b2a4b4a6b6
        "trn1 v8.4h, v0.4h, v1.4h                   \n"
        "trn2 v9.4h, v0.4h, v1.4h                   \n"
        "ld1 {v5.4h}, [%[inptr5]], #8               \n"
        // c0d0c2d2c4d4c6d6
        "trn1 v10.4h, v2.4h, v3.4h                  \n"
        "trn2 v11.4h, v2.4h, v3.4h                  \n"
        "ld1 {v6.4h}, [%[inptr6]], #8              \n"
        // e0f0e2f2e4f4e6f6
        "trn1 v12.4h, v4.4h, v5.4h                  \n"
        "trn2 v13.4h, v4.4h, v5.4h                  \n"
        "ld1 {v7.4h}, [%[inptr7]], #8              \n"
        // a0b0c0d0a4b4c4d4
        "trn1 v0.2s, v8.2s, v10.2s                  \n"
        // a2b2c2d2a6b6c6d6
        "trn2 v1.2s, v8.2s, v10.2s                  \n"
        // g0h0g2h2g4h4g6h6
        "trn1 v14.4h, v6.4h, v7.4h                  \n"
        "trn2 v15.4h, v6.4h, v7.4h                  \n"
        // a1b1..a5b5..
        "trn1 v2.2s, v9.2s, v11.2s                  \n"
        // a3b3..a7b7..
        "trn2 v3.2s, v9.2s, v11.2s                  \n"
        // e0f0g0h0..
        "cmp %w[has_alpha], #1                      \n"
        "trn1 v4.2s, v12.2s, v14.2s                 \n"
        "trn2 v5.2s, v12.2s, v14.2s                 \n"
        "trn1 v6.2s, v13.2s, v15.2s                 \n"
        "trn2 v7.2s, v13.2s, v15.2s                 \n"
        "bne 11f                                    \n"
        "fmul v0.8h, v0.8h, %[valpha].8h            \n"
        "fmul v1.8h, v1.8h, %[valpha].8h            \n"
        "fmul v2.8h, v2.8h, %[valpha].8h            \n"
        "fmul v3.8h, v3.8h, %[valpha].8h            \n"
        "fmul v4.8h, v4.8h, %[valpha].8h            \n"
        "fmul v5.8h, v5.8h, %[valpha].8h            \n"
        "fmul v6.8h, v6.8h, %[valpha].8h            \n"
        "fmul v7.8h, v7.8h, %[valpha].8h            \n"
        "11:                                        \n"
        // 0
        "st1 {v0.2s}, [%[outptr]], #8               \n"
        "st1 {v4.2s}, [%[outptr]], #8               \n"
        // 1
        "st1 {v2.2s}, [%[outptr]], #8               \n"
        "st1 {v6.2s}, [%[outptr]], #8               \n"
        // 2
        "st1 {v1.2s}, [%[outptr]], #8               \n"
        "st1 {v5.2s}, [%[outptr]], #8               \n"
        // 3
        "st1 {v3.2s}, [%[outptr]], #8               \n"
        "st1 {v7.2s}, [%[outptr]], #8               \n"
        "2:                                         \n"
        : [inptr0] "+r"(inptr0),
          [inptr1] "+r"(inptr1),
          [inptr2] "+r"(inptr2),
          [inptr3] "+r"(inptr3),
          [inptr4] "+r"(inptr4),
          [inptr5] "+r"(inptr5),
          [inptr6] "+r"(inptr6),
          [inptr7] "+r"(inptr7),
          [outptr] "+r"(outptr),
          [cnt] "+r"(cnt_col)
        : [cnt_4] "r"(cnt_4),
          [has_alpha] "r"(has_alpha),
          [valpha] "w"(valpha)
        : "v0",
          "v1",
          "v2",
          "v3",
          "v4",
          "v5",
          "v6",
          "v7",
          "v8",
          "v9",
          "v10",
          "v11",
          "v12",
          "v13",
          "v14",
          "v15",
          "cc",
          "memory");
    // clang-format on
    for (int x = 0; x < remain; x++) {
      if (has_alpha) {
        *outptr++ = *inptr0++ * alpha;
        *outptr++ = *inptr1++ * alpha;
        *outptr++ = *inptr2++ * alpha;
        *outptr++ = *inptr3++ * alpha;
        *outptr++ = *inptr4++ * alpha;
        *outptr++ = *inptr5++ * alpha;
        *outptr++ = *inptr6++ * alpha;
        *outptr++ = *inptr7++ * alpha;
      } else {
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
  }
}

void prepackA_trans_8x16(float16_t *out,
                         const float16_t *in,
                         float16_t alpha,
                         const int ldin,
                         const int m0,
                         const int mmax,
                         const int k0,
                         const int kmax) {
  float16_t *outptr = out;
  const float16_t *inptr = in + k0 * ldin + m0;

  uint16_t mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int x_len = mmax - m0;
  int y_len = kmax - k0;
  int cnt = x_len >> 3;
  uint16_t right_remain = x_len & 7;
  int stride_out = 16 * y_len;
  bool has_alpha = fabsf(alpha - 1.f) > 1e-8f;

  uint16x8_t vzero = vdupq_n_u16(0);
  uint16x8_t vmask =
      vcltq_u16(vld1q_u16(mask_buffer), vdupq_n_u16(right_remain));
  float16x8_t valpha = vdupq_n_f16(alpha);

#pragma omp parallel for
  for (int y = 0; y < y_len - 3; y += 4) {
    const float16_t *ptr0 = inptr + y * ldin;
    const float16_t *ptr1 = ptr0 + ldin;
    const float16_t *ptr2 = ptr1 + ldin;
    const float16_t *ptr3 = ptr2 + ldin;
    float16_t *outptr_row_col = outptr + y * 8;
    int cnt_col = cnt;
    asm volatile(
        "cmp %w[cnt], #1                            \n"
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "prfm   pldl1keep, [%[ptr1]]                \n"
        "prfm   pldl1keep, [%[ptr2]]                \n"
        "prfm   pldl1keep, [%[ptr3]]                \n"
        "blt 1f                                     \n"
        "0:                                         \n"
        "cmp %w[has_alpha], #1                      \n"
        "ld1 {v0.8h}, [%[ptr0]], #16                \n"
        "ld1 {v1.8h}, [%[ptr1]], #16                \n"
        "ld1 {v2.8h}, [%[ptr2]], #16                \n"
        "ld1 {v3.8h}, [%[ptr3]], #16                \n"
        "bne 3f                                     \n"
        "fmul v0.8h, v0.8h, %[valpha].8h            \n"
        "fmul v1.8h, v1.8h, %[valpha].8h            \n"
        "fmul v2.8h, v2.8h, %[valpha].8h            \n"
        "fmul v3.8h, v3.8h, %[valpha].8h            \n"
        "3:                                         \n"
        "subs %w[cnt], %w[cnt], #1                  \n"
        "str q0, [%[outptr]]                        \n"
        "str q1, [%[outptr], #16]                   \n"
        "str q2, [%[outptr], #32]                   \n"
        "str q3, [%[outptr], #48]                   \n"
        "add %[outptr], %[outptr], %[stride]        \n"
        "bne 0b                                     \n"
        "1:                                         \n"
        "cmp %w[right_remain], #1                   \n"
        "blt 2f                                     \n"
        "ld1 {v0.8h}, [%[ptr0]]                     \n"
        "ld1 {v1.8h}, [%[ptr1]]                     \n"
        "ld1 {v2.8h}, [%[ptr2]]                     \n"
        "ld1 {v3.8h}, [%[ptr3]]                     \n"
        "cmp %w[has_alpha], #1                      \n"
        "bif v0.16b, %[vzero].16b, %[vmask].16b     \n"
        "bif v1.16b, %[vzero].16b, %[vmask].16b     \n"
        "bif v2.16b, %[vzero].16b, %[vmask].16b     \n"
        "bif v3.16b, %[vzero].16b, %[vmask].16b     \n"
        "bne 4f                                     \n"
        "fmul v0.8h, v0.8h, %[valpha].8h            \n"
        "fmul v1.8h, v1.8h, %[valpha].8h            \n"
        "fmul v2.8h, v2.8h, %[valpha].8h            \n"
        "fmul v3.8h, v3.8h, %[valpha].8h            \n"
        "4:                                         \n"
        "str q0, [%[outptr]]                        \n"
        "str q1, [%[outptr], #16]                   \n"
        "str q2, [%[outptr], #32]                   \n"
        "str q3, [%[outptr], #48]                   \n"
        "2:                                         \n"
        : [ptr0] "+r"(ptr0),
          [ptr1] "+r"(ptr1),
          [ptr2] "+r"(ptr2),
          [ptr3] "+r"(ptr3),
          [outptr] "+r"(outptr_row_col),
          [cnt] "+r"(cnt_col)
        : [right_remain] "r"(right_remain),
          [stride] "r"(stride_out),
          [vzero] "w"(vzero),
          [has_alpha] "r"(has_alpha),
          [valpha] "w"(valpha),
          [vmask] "w"(vmask)
        : "cc", "memory", "v0", "v1", "v2", "v3");
  }
#pragma omp parallel for
  for (int y = 4 * (y_len / 4); y < y_len; ++y) {
    const float16_t *ptr0 = inptr + y * ldin;
    float16_t *outptr_row_col = outptr + y * 8;
    int cnt_col = cnt;
    asm volatile(
        "cmp %w[cnt], #1                            \n"
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "blt 1f                                     \n"
        "0:                                         \n"
        "cmp %w[has_alpha], #1                      \n"
        "ld1 {v0.8h}, [%[ptr0]], #16                \n"
        "bne 3f                                     \n"
        "fmul v0.8h, v0.8h, %[valpha].8h            \n"
        "3:                                         \n"
        "subs %w[cnt], %w[cnt], #1                  \n"
        "str q0, [%[outptr]]                        \n"
        "add %[outptr], %[outptr], %[stride]        \n"
        "bne 0b                                     \n"
        "1:                                         \n"
        "cmp %w[right_remain], #1                   \n"
        "blt 2f                                     \n"
        "cmp %w[has_alpha], #1                      \n"
        "ld1 {v0.8h}, [%[ptr0]]                     \n"
        "bif v0.16b, %[vzero].16b, %[vmask].16b     \n"
        "bne 4f                                     \n"
        "fmul v0.8h, v0.8h, %[valpha].8h            \n"
        "4:                                         \n"
        "str q0, [%[outptr]]                        \n"
        "2:                                         \n"
        : [ptr0] "+r"(ptr0), [outptr] "+r"(outptr_row_col), [cnt] "+r"(cnt_col)
        : [right_remain] "r"(right_remain),
          [stride] "r"(stride_out),
          [vzero] "w"(vzero),
          [has_alpha] "r"(has_alpha),
          [valpha] "w"(valpha),
          [vmask] "w"(vmask)
        : "cc", "memory", "v0", "v1", "v2", "v3");
  }
}
#else
#endif

/**
* \brief input data is transpose
* for arm-v7a, transform data to block x k x 8 layout
* for arm-v8a, transform data to block x k x 16 layout
*/
#ifdef __aarch64__
void loadb(float16_t *out,
           const float16_t *in,
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
  int rem_cnt = right_remain >> 2;
  int rem_rem = right_remain & 3;

  int cnt_num = (x_len >= 16) ? 16 : (x_len >= 4 ? 4 : 1);
  int stride_out = cnt_num * y_len * 2;
  int stride = y_len << 3;  // 4 * y_len * 2
  // rem_cnt > 0 ? (16 - 4) : (16 - 1)
  int stride_w = (rem_cnt > 0) ? 24 : 30;  // (16 - 4) * 2
  int stride_w2 = 6;                       // (4 - 1) * 2;
  if (y_len >= 4) {
    stride_w = stride_w << 2;    // stride_w * 4
    stride_w2 = stride_w2 << 2;  // stride_w2 * 4
  }
  int cnt_y = 4 * (y_len / 4);
  int stride_k = y_len << 1;

#pragma omp parallel for
  for (int y = 0; y < y_len - 3; y += 4) {
    const uint16_t *ptr0 = inptr + y * ldin;
    const uint16_t *ptr1 = ptr0 + ldin;
    const uint16_t *ptr2 = ptr1 + ldin;
    const uint16_t *ptr3 = ptr2 + ldin;

    uint16_t *outptr_row_col = outptr_row + y * cnt_num;
    int cnt_col = cnt;
    bool y_line = (y > 0 && cnt > 0);
    int cnt_rem_num = rem_cnt;
    int rem_rem_rem = rem_rem;
    int temp = y_line ? stride_w * (y / 4) : 0;
    int temp2 = (y > 0 && rem_cnt > 0) ? stride_w2 * (y / 4) : 0;
    if (cnt > 0) {
      for (int i = 0; i < cnt; i++) {
        uint16x8_t v0 = vld1q_u16(ptr0);
        uint16x8_t v01 = vld1q_u16(ptr0 + 8);
        uint16x8_t v1 = vld1q_u16(ptr1);
        uint16x8_t v11 = vld1q_u16(ptr1 + 8);
        uint16x8_t v2 = vld1q_u16(ptr2);
        uint16x8_t v21 = vld1q_u16(ptr2 + 8);
        vst1q_u16(outptr_row_col, v0);
        uint16x8_t v3 = vld1q_u16(ptr3);
        vst1q_u16(outptr_row_col + 8, v01);
        uint16x8_t v31 = vld1q_u16(ptr3 + 8);
        vst1q_u16(outptr_row_col + 16, v1);
        ptr0 += 16;
        vst1q_u16(outptr_row_col + 24, v11);
        ptr1 += 16;
        vst1q_u16(outptr_row_col + 32, v2);
        ptr2 += 16;
        vst1q_u16(outptr_row_col + 40, v21);
        ptr3 += 16;
        vst1q_u16(outptr_row_col + 48, v3);
        vst1q_u16(outptr_row_col + 56, v31);
        outptr_row_col += (stride_out / 2);
      }
      outptr_row_col -= (temp / 2);
    }
    if (rem_cnt > 0) {
      for (int i = 0; i < rem_cnt; i++) {
        uint16x4_t v0 = vld1_u16(ptr0);
        uint16x4_t v1 = vld1_u16(ptr1);
        uint16x4_t v2 = vld1_u16(ptr2);
        uint16x4_t v3 = vld1_u16(ptr3);
        ptr0 += 4;
        vst1_u16(outptr_row_col, v0);
        ptr1 += 4;
        vst1_u16(outptr_row_col + 4, v1);
        ptr2 += 4;
        vst1_u16(outptr_row_col + 8, v2);
        ptr3 += 4;
        vst1_u16(outptr_row_col + 12, v3);
        outptr_row_col += (stride / 2);
      }
      outptr_row_col -= (temp2 / 2);
    }
    if (rem_rem > 0) {
      for (int i = 0; i < rem_rem; i++) {
        outptr_row_col[0] = *ptr0++;
        outptr_row_col[1] = *ptr1++;
        outptr_row_col[2] = *ptr2++;
        outptr_row_col[3] = *ptr3++;
        outptr_row_col += (stride_k / 2);
      }
    }
  }

#pragma omp parallel for
  for (int y = cnt_y; y < y_len; ++y) {
    const uint16_t *ptr0 = inptr + y * ldin;
    uint16_t *outptr_row_col = outptr_row + y * cnt_num;
    int cnt_col = cnt;
    bool y_line = (y > 0 && cnt > 0);
    int cnt_rem_num = rem_cnt;
    int rem_rem_rem = rem_rem;
    // (y - cnt_y) * (16 - 4) * 2 || (y - cnt_y) * (16 - 1) * 2
    int valid_w = rem_cnt > 0 ? 24 : 30;
    int temp = y_line ? (stride_w * (y / 4) + (y - cnt_y) * valid_w) : 0;
    // (y - cnt_y) * (4 - 1) * 2
    int temp2 =
        (y > 0 && rem_cnt > 0) ? (stride_w2 * (y / 4) + (y - cnt_y) * 6) : 0;
    if (cnt > 0) {
      for (int i = 0; i < cnt; i++) {
        uint16x8_t v0 = vld1q_u16(ptr0);
        uint16x8_t v1 = vld1q_u16(ptr0 + 8);
        ptr0 += 16;
        vst1q_u16(outptr_row_col, v0);
        vst1q_u16(outptr_row_col + 8, v1);
        outptr_row_col += (stride_out / 2);
      }
      outptr_row_col -= (temp / 2);
    }
    if (rem_cnt > 0) {
      for (int i = 0; i < rem_cnt; i++) {
        uint16x4_t v0 = vld1_u16(ptr0);
        ptr0 += 4;
        vst1_u16(outptr_row_col, v0);
        outptr_row_col += (stride / 2);
      }
      outptr_row_col -= (temp2 / 2);
    }
    if (rem_rem > 0) {
      for (int i = 0; i < rem_rem; i++) {
        *outptr_row_col = *ptr0++;
        outptr_row_col += (stride_k / 2);
      }
    }
  }
}

void loadb_trans(float16_t *out,
                 const float16_t *in,
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
    int cnt_col = cnt;
    // clang-format off
    asm volatile(
        "prfm   pldl1keep, [%[inptr0]]        \n"
        "prfm   pldl1keep, [%[inptr1]]        \n"
        "prfm   pldl1keep, [%[inptr2]]        \n"
        "prfm   pldl1keep, [%[inptr3]]        \n"
        "cmp %w[cnt], #1                      \n"
        "prfm   pldl1keep, [%[inptr4]]        \n"
        "prfm   pldl1keep, [%[inptr5]]        \n"
        "prfm   pldl1keep, [%[inptr6]]        \n"
        "prfm   pldl1keep, [%[inptr7]]        \n"
        "blt 1f                               \n"
        "0:                                   \n"
        "ld1 {v0.8h}, [%[inptr0]], #16        \n"
        "ld1 {v1.8h}, [%[inptr1]], #16        \n"
        "ld1 {v2.8h}, [%[inptr2]], #16        \n"
        "ld1 {v3.8h}, [%[inptr3]], #16        \n"
        "prfm   pldl1keep, [%[inptr8]]        \n"
        "ld1 {v4.8h}, [%[inptr4]], #16        \n"
        "prfm   pldl1keep, [%[inptr9]]        \n"
        "ld1 {v5.8h}, [%[inptr5]], #16        \n"
        // a0b0a2b2a4b4a6b6
        "trn1 v8.8h, v0.8h, v1.8h             \n"
        "trn2 v9.8h, v0.8h, v1.8h             \n"
        "prfm   pldl1keep, [%[inptr10]]       \n"
        "ld1 {v6.8h}, [%[inptr6]], #16        \n"
        "prfm   pldl1keep, [%[inptr11]]       \n"
        "ld1 {v7.8h}, [%[inptr7]], #16        \n"
        // c0d0c2d2c4d4c6d6
        "trn1 v10.8h, v2.8h, v3.8h            \n"
        "trn2 v11.8h, v2.8h, v3.8h            \n"
        "prfm   pldl1keep, [%[inptr12]]       \n"
        "prfm   pldl1keep, [%[inptr13]]       \n"
        // e0f0e2f2...
        "trn1 v12.8h, v4.8h, v5.8h            \n"
        "trn2 v13.8h, v4.8h, v5.8h            \n"
        "prfm   pldl1keep, [%[inptr14]]       \n"
        "prfm   pldl1keep, [%[inptr15]]       \n"
        TRANS_C8
        "ld1 {v0.8h}, [%[inptr8]], #16        \n"
        "str q8, [%[outptr]]                  \n"
        "ld1 {v1.8h}, [%[inptr9]], #16        \n"
        "str q12, [%[outptr], #32]            \n"
        "ld1 {v2.8h}, [%[inptr10]], #16       \n"
        "str q10, [%[outptr], #64]            \n"
        "ld1 {v3.8h}, [%[inptr11]], #16       \n"
        "str q14, [%[outptr], #96]            \n"
        "ld1 {v4.8h}, [%[inptr12]], #16       \n"
        "str q9, [%[outptr], #128]            \n"
        "ld1 {v5.8h}, [%[inptr13]], #16       \n"
        "str q13, [%[outptr], #160]           \n"
        "ld1 {v6.8h}, [%[inptr14]], #16       \n"
        "str q11, [%[outptr], #192]           \n"
        "ld1 {v7.8h}, [%[inptr15]], #16       \n"
        "str q15, [%[outptr], #224]           \n"
        // a0b0a2b2a4b4a6b6
        "trn1 v8.8h, v0.8h, v1.8h             \n"
        "trn2 v9.8h, v0.8h, v1.8h             \n"
        "prfm   pldl1keep, [%[inptr0]]        \n"
        "prfm   pldl1keep, [%[inptr1]]        \n"
        // c0d0c2d2c4d4c6d6
        "trn1 v10.8h, v2.8h, v3.8h            \n"
        "trn2 v11.8h, v2.8h, v3.8h            \n"
        "prfm   pldl1keep, [%[inptr2]]        \n"
        "prfm   pldl1keep, [%[inptr3]]        \n"
        // e0f0e2f2...
        "trn1 v12.8h, v4.8h, v5.8h            \n"
        "trn2 v13.8h, v4.8h, v5.8h            \n"
        "prfm   pldl1keep, [%[inptr4]]        \n"
        "prfm   pldl1keep, [%[inptr5]]        \n"
        "prfm   pldl1keep, [%[inptr6]]        \n"
        "prfm   pldl1keep, [%[inptr7]]        \n"
        TRANS_C8
        "subs %w[cnt], %w[cnt], #1            \n"
        "str q8, [%[outptr], #16]             \n"
        "str q12, [%[outptr], #48]            \n"
        "str q10, [%[outptr], #80]            \n"
        "str q14, [%[outptr], #112]           \n"
        "str q9, [%[outptr], #144]            \n"
        "str q13, [%[outptr], #176]           \n"
        "str q11, [%[outptr], #208]           \n"
        "str q15, [%[outptr], #240]           \n"
        "add %[outptr], %[outptr], #256\n"
        "bne 0b                               \n"
        "1:                                   \n"
        : [inptr0] "+r"(inptr0),
          [inptr1] "+r"(inptr1),
          [inptr2] "+r"(inptr2),
          [inptr3] "+r"(inptr3),
          [inptr4] "+r"(inptr4),
          [inptr5] "+r"(inptr5),
          [inptr6] "+r"(inptr6),
          [inptr7] "+r"(inptr7),
          [inptr8] "+r"(inptr8),
          [inptr9] "+r"(inptr9),
          [inptr10] "+r"(inptr10),
          [inptr11] "+r"(inptr11),
          [inptr12] "+r"(inptr12),
          [inptr13] "+r"(inptr13),
          [inptr14] "+r"(inptr14),
          [inptr15] "+r"(inptr15),
          [outptr] "+r"(outptr),
          [cnt] "+r"(cnt_col)
        :
        : "cc",
          "memory",
          "v0",
          "v1",
          "v2",
          "v3",
          "v4",
          "v5",
          "v6",
          "v7",
          "v8",
          "v9",
          "v10",
          "v11",
          "v12",
          "v13",
          "v14",
          "v15");
    // clang-format on
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
  for (; y < nmax - 3; y += 4) {
    const uint16_t *inptr0 = inptr + y * ldin + k0;
    const uint16_t *inptr1 = inptr0 + ldin;
    const uint16_t *inptr2 = inptr1 + ldin;
    const uint16_t *inptr3 = inptr2 + ldin;
    //! cope with row index exceed real size, set to zero buffer
    int cnt_col = cnt;
    // clang-format off
    asm volatile(
        "cmp %w[cnt], #1                      \n"
        "prfm   pldl1keep, [%[inptr0]]        \n"
        "prfm   pldl1keep, [%[inptr1]]        \n"
        "prfm   pldl1keep, [%[inptr2]]        \n"
        "prfm   pldl1keep, [%[inptr3]]        \n"
        "ld1 {v0.8h}, [%[inptr0]], #16        \n"
        "ld1 {v1.8h}, [%[inptr1]], #16        \n"
        "ld1 {v2.8h}, [%[inptr2]], #16        \n"
        "ld1 {v3.8h}, [%[inptr3]], #16        \n"
        "blt 1f                               \n"
        "0:                                   \n"
        // a0b0a2b2a4b4a6b6
        "trn1 v8.8h, v0.8h, v1.8h             \n"
        "trn2 v9.8h, v0.8h, v1.8h             \n"
        // c0d0c2d2c4d4c6d6
        "trn1 v10.8h, v2.8h, v3.8h            \n"
        "trn2 v11.8h, v2.8h, v3.8h            \n"
        "ld1 {v0.8h}, [%[inptr0]], #16        \n"
        // a0b0c0d0a4b4c4d4 a2b2c2d2a6b6c6d6
        "trn1 v12.4s, v8.4s, v10.4s           \n"
        "trn2 v13.4s, v8.4s, v10.4s           \n"
        "ld1 {v1.8h}, [%[inptr1]], #16        \n"
        // a1b1c1d1a5b5c5d5d5 a3b3c3d3a7b7c7d7
        "trn1 v14.4s, v9.4s, v11.4s           \n"
        "trn2 v15.4s, v9.4s, v11.4s           \n"
        "ld1 {v2.8h}, [%[inptr2]], #16        \n"
        // a0b0c0d0a1b1c1d1
        "trn1 v8.2d, v12.2d, v14.2d           \n"
        "trn2 v9.2d, v12.2d, v14.2d           \n"
        // a2b2c2d2a3b3c3d3
        "trn1 v10.2d, v13.2d, v15.2d          \n"
        "trn2 v11.2d, v13.2d, v15.2d          \n"
        "ld1 {v3.8h}, [%[inptr3]], #16        \n"
        "str q8, [%[outptr]]                  \n"
        "subs %w[cnt], %w[cnt], #1            \n"
        "str q10, [%[outptr], #0x10]          \n"
        "str q9,  [%[outptr], #0x20]          \n"
        "str q11, [%[outptr], #0x30]          \n"
        "add %[outptr], %[outptr], #64        \n"
        "bne 0b                               \n"
        "1:                                   \n"
        "sub %[inptr0], %[inptr0], #16        \n"
        "sub %[inptr1], %[inptr1], #16        \n"
        "sub %[inptr2], %[inptr2], #16        \n"
        "sub %[inptr3], %[inptr3], #16        \n"
        : [inptr0] "+r"(inptr0),
          [inptr1] "+r"(inptr1),
          [inptr2] "+r"(inptr2),
          [inptr3] "+r"(inptr3),
          [outptr] "+r"(outptr),
          [cnt] "+r"(cnt_col)
        :
        : "cc",
          "memory",
          "v0",
          "v1",
          "v2",
          "v3",
          "v4",
          "v5",
          "v6",
          "v7",
          "v8",
          "v9",
          "v10",
          "v11",
          "v12",
          "v13",
          "v14",
          "v15");
    // clang-format on
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
#endif
#ifdef __aarch64__
#define FMLA_N00_4                        \
  "fmla v8.4h,  v2.4h, v0.h[0]        \n" \
  "fmla v10.4h, v2.4h, v0.h[1]        \n" \
  "fmla v12.4h, v2.4h, v0.h[2]        \n" \
  "fmla v14.4h, v2.4h, v0.h[3]        \n"

#define FMLA_N01_4                        \
  "fmla v16.4h, v2.4h, v0.h[4]        \n" \
  "fmla v18.4h, v2.4h, v0.h[5]        \n" \
  "fmla v20.4h, v2.4h, v0.h[6]        \n" \
  "fmla v22.4h, v2.4h, v0.h[7]        \n"

#define FMLA_N10_4                        \
  "fmla v8.4h,  v3.4h, v1.h[0]        \n" \
  "fmla v10.4h, v3.4h, v1.h[1]        \n" \
  "fmla v12.4h, v3.4h, v1.h[2]        \n" \
  "fmla v14.4h, v3.4h, v1.h[3]        \n"

#define FMLA_N11_4                        \
  "fmla v16.4h, v3.4h, v1.h[4]        \n" \
  "fmla v18.4h, v3.4h, v1.h[5]        \n" \
  "fmla v20.4h, v3.4h, v1.h[6]        \n" \
  "fmla v22.4h, v3.4h, v1.h[7]        \n"

#define LEAKY_0_4                           \
  "fcmge v0.4h,  v8.4h,  %[vzero].4h    \n" \
  "fmul  v1.4h,  v8.4h,  %[valpha].4h   \n" \
  "fcmge v2.4h,  v10.4h, %[vzero].4h    \n" \
  "fmul  v3.4h,  v10.4h, %[valpha].4h   \n" \
  "fcmge v4.4h,  v12.4h, %[vzero].4h    \n" \
  "fmul  v5.4h,  v12.4h, %[valpha].4h   \n" \
  "fcmge v6.4h,  v14.4h, %[vzero].4h    \n" \
  "fmul  v7.4h,  v14.4h, %[valpha].4h   \n" \
  "bif   v8.8b,  v1.8b,  v0.8b          \n" \
  "bif   v10.8b, v3.8b,  v2.8b          \n" \
  "bif   v12.8b, v5.8b,  v4.8b          \n" \
  "bif   v14.8b, v7.8b,  v6.8b          \n"

#define LEAKY_1_4                           \
  "fcmge v0.4h,  v16.4h, %[vzero].4h    \n" \
  "fmul  v1.4h,  v16.4h, %[valpha].4h   \n" \
  "fcmge v2.4h,  v18.4h, %[vzero].4h    \n" \
  "fmul  v3.4h,  v18.4h, %[valpha].4h   \n" \
  "fcmge v4.4h,  v20.4h, %[vzero].4h    \n" \
  "fmul  v5.4h,  v20.4h, %[valpha].4h   \n" \
  "fcmge v6.4h,  v22.4h, %[vzero].4h    \n" \
  "fmul  v7.4h,  v22.4h, %[valpha].4h   \n" \
  "bif   v16.8b,  v1.8b,  v0.8b         \n" \
  "bif   v18.8b,  v3.8b,  v2.8b         \n" \
  "bif   v20.8b,  v5.8b,  v4.8b         \n" \
  "bif   v22.8b,  v7.8b,  v6.8b         \n"

#define FMAX_4                            \
  "fmax v8.4h,  v8.4h,  %[vzero].4h   \n" \
  "fmax v10.4h, v10.4h, %[vzero].4h   \n" \
  "fmax v12.4h, v12.4h, %[vzero].4h   \n" \
  "fmax v14.4h, v14.4h, %[vzero].4h   \n" \
  "fmax v16.4h, v16.4h, %[vzero].4h   \n" \
  "fmax v18.4h, v18.4h, %[vzero].4h   \n" \
  "fmax v20.4h, v20.4h, %[vzero].4h   \n" \
  "fmax v22.4h, v22.4h, %[vzero].4h   \n"

#define FMIN_4                             \
  "fmin v8.4h,  v8.4h,  %[valpha].4h   \n" \
  "fmin v10.4h, v10.4h, %[valpha].4h   \n" \
  "fmin v12.4h, v12.4h, %[valpha].4h   \n" \
  "fmin v14.4h, v14.4h, %[valpha].4h   \n" \
  "fmin v16.4h, v16.4h, %[valpha].4h   \n" \
  "fmin v18.4h, v18.4h, %[valpha].4h   \n" \
  "fmin v20.4h, v20.4h, %[valpha].4h   \n" \
  "fmin v22.4h, v22.4h, %[valpha].4h   \n"

#define FMLA_N00_8                        \
  "fmla v8.8h,  v2.8h, v0.h[0]        \n" \
  "fmla v10.8h, v2.8h, v0.h[1]        \n" \
  "fmla v12.8h, v2.8h, v0.h[2]        \n" \
  "fmla v14.8h, v2.8h, v0.h[3]        \n" \
  "fmla v16.8h, v2.8h, v0.h[4]        \n" \
  "fmla v18.8h, v2.8h, v0.h[5]        \n" \
  "fmla v20.8h, v2.8h, v0.h[6]        \n" \
  "fmla v22.8h, v2.8h, v0.h[7]        \n"

#define FMLA_N01_8                        \
  "fmla v9.8h,  v3.8h, v0.h[0]        \n" \
  "fmla v11.8h, v3.8h, v0.h[1]        \n" \
  "fmla v13.8h, v3.8h, v0.h[2]        \n" \
  "fmla v15.8h, v3.8h, v0.h[3]        \n" \
  "fmla v17.8h, v3.8h, v0.h[4]        \n" \
  "fmla v19.8h, v3.8h, v0.h[5]        \n" \
  "fmla v21.8h, v3.8h, v0.h[6]        \n" \
  "fmla v23.8h, v3.8h, v0.h[7]        \n"

#define FMLA_N10_8                         \
  "fmla v8.8h,  v4.8h, v1.h[0]         \n" \
  "fmla v10.8h, v4.8h, v1.h[1]        \n"  \
  "fmla v12.8h, v4.8h, v1.h[2]        \n"  \
  "fmla v14.8h, v4.8h, v1.h[3]        \n"  \
  "fmla v16.8h, v4.8h, v1.h[4]        \n"  \
  "fmla v18.8h, v4.8h, v1.h[5]        \n"  \
  "fmla v20.8h, v4.8h, v1.h[6]        \n"  \
  "fmla v22.8h, v4.8h, v1.h[7]        \n"

#define FMLA_N11_8                         \
  "fmla v9.8h,  v5.8h, v1.h[0]         \n" \
  "fmla v11.8h, v5.8h, v1.h[1]        \n"  \
  "fmla v13.8h, v5.8h, v1.h[2]        \n"  \
  "fmla v15.8h, v5.8h, v1.h[3]        \n"  \
  "fmla v17.8h, v5.8h, v1.h[4]        \n"  \
  "fmla v19.8h, v5.8h, v1.h[5]        \n"  \
  "fmla v21.8h, v5.8h, v1.h[6]        \n"  \
  "fmla v23.8h, v5.8h, v1.h[7]        \n"

#define FMAX_8                            \
  "fmax v8.8h,  v8.8h,  %[vzero].8h   \n" \
  "fmax v10.8h, v10.8h, %[vzero].8h   \n" \
  "fmax v12.8h, v12.8h, %[vzero].8h   \n" \
  "fmax v14.8h, v14.8h, %[vzero].8h   \n" \
  "fmax v16.8h, v16.8h, %[vzero].8h   \n" \
  "fmax v18.8h, v18.8h, %[vzero].8h   \n" \
  "fmax v20.8h, v20.8h, %[vzero].8h   \n" \
  "fmax v22.8h, v22.8h, %[vzero].8h   \n" \
  "fmax v9.8h,  v9.8h,  %[vzero].8h   \n" \
  "fmax v11.8h, v11.8h, %[vzero].8h   \n" \
  "fmax v13.8h, v13.8h, %[vzero].8h   \n" \
  "fmax v15.8h, v15.8h, %[vzero].8h   \n" \
  "fmax v17.8h, v17.8h, %[vzero].8h   \n" \
  "fmax v19.8h, v19.8h, %[vzero].8h   \n" \
  "fmax v21.8h, v21.8h, %[vzero].8h   \n" \
  "fmax v23.8h, v23.8h, %[vzero].8h   \n"

#define FMIN_8                             \
  "fmin v8.8h,  v8.8h,  %[valpha].8h   \n" \
  "fmin v10.8h, v10.8h, %[valpha].8h   \n" \
  "fmin v12.8h, v12.8h, %[valpha].8h   \n" \
  "fmin v14.8h, v14.8h, %[valpha].8h   \n" \
  "fmin v16.8h, v16.8h, %[valpha].8h   \n" \
  "fmin v18.8h, v18.8h, %[valpha].8h   \n" \
  "fmin v20.8h, v20.8h, %[valpha].8h   \n" \
  "fmin v22.8h, v22.8h, %[valpha].8h   \n" \
  "fmin v9.8h,  v9.8h,  %[valpha].8h   \n" \
  "fmin v11.8h, v11.8h, %[valpha].8h   \n" \
  "fmin v13.8h, v13.8h, %[valpha].8h   \n" \
  "fmin v15.8h, v15.8h, %[valpha].8h   \n" \
  "fmin v17.8h, v17.8h, %[valpha].8h   \n" \
  "fmin v19.8h, v19.8h, %[valpha].8h   \n" \
  "fmin v21.8h, v21.8h, %[valpha].8h   \n" \
  "fmin v23.8h, v23.8h, %[valpha].8h   \n"

void gemm_prepack_8x16(bool is_transB,
                       int M,
                       int N,
                       int K,
                       const float16_t *A_packed,
                       const float16_t *B,
                       int ldb,
                       float16_t beta,
                       float16_t *C,
                       int ldc,
                       const float16_t *bias,
                       bool has_bias,
                       const operators::ActivationParam act_param,
                       ARMContext *ctx) {
  size_t llc_size = ctx->llc_size() > 0 ? ctx->llc_size() : 512 * 1024;
  auto workspace = ctx->workspace_data<float16_t>();
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

  float16x8_t valpha = vdupq_n_f16(static_cast<float16_t>(local_alpha));
  //! MBLOCK * x (result) + MBLOCK * k (A) + x * k (B) = l2
  X_BLOCK_COMPUTE(llc_size, MBLOCK_FP16, NBLOCK_FP16, KBLOCK_FP16, beta)
  float16x8_t vbeta = vdupq_n_f16(beta);
  float16x8_t vzero = vdupq_n_f16(0.f);

  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    if (xmax > N) {
      xmax = N;
    }
    int bblocks = (xmax - x0 + NBLOCK_FP16 - 1) / NBLOCK_FP16;
    remain = xmax - x0 - (bblocks - 1) * NBLOCK_FP16;
    if (remain > 0 && remain != 16) {
      flag_p_remain = true;
    }
    //! load bpanel
    float16_t *b_pannel = workspace;
    if (is_transB) {
      loadb_trans(b_pannel, B, ldb, 0, K, x0, xmax);
    } else {
      loadb(b_pannel, B, ldb, 0, K, x0, xmax);
    }
#pragma omp parallel for num_threads(threads)
    for (unsigned int y = 0; y < M; y += MBLOCK_FP16) {
      unsigned int ymax = y + MBLOCK_FP16;
      if (ymax > M) {
        ymax = M;
      }

      float16_t bias_local[8] = {0};
      if (has_bias) {
        if (y + 7 >= ymax) {
          ptr_acquire_b8<float16_t>(bias_local, bias, y, (y + 7), ymax);
        } else {
          for (int i = 0; i < 8; i++) {
            bias_local[i] = bias[y + i];
          }
        }
      }
      float16x8_t vbias = vld1q_f16(bias_local);
      // prepare out data
      GEMM_PREPARE_C(float16_t, NBLOCK_FP16)
      const float16_t *a_ptr_l = A_packed + y * K;
      const float16_t *b_ptr = b_pannel;
      for (int xb = 0; xb < bblocks; xb++) {
        if ((y + 7) >= ymax) {
          switch ((y + 7) - ymax) {
            case 6:
              c_ptr1 = cout1;
            case 5:
              c_ptr2 = cout2;
            case 4:
              c_ptr3 = cout3;
            case 3:
              c_ptr4 = cout4;
            case 2:
              c_ptr5 = cout5;
            case 1:
              c_ptr6 = cout6;
            case 0:
              c_ptr7 = cout7;
            default:
              break;
          }
        }
        if (flag_p_remain && (xb == bblocks - 1)) {
          int cnt_rem = remain >> 2;
          int rem_rem = remain & 3;
          for (int i = 0; i < cnt_rem; i++) {
            const float16_t *a_ptr = a_ptr_l;
            int tail = tail_pre;
            int k = k_pre;
            // clang-format off
            asm volatile(
              "prfm   pldl1keep, [%[a_ptr]]       \n"
              "prfm   pldl1keep, [%[b_ptr]]       \n"
              "dup	v8.4h, %[vbias].h[0]          \n"
              "prfm   pldl1keep, [%[b_ptr], #64]  \n"
              "dup	v10.4h, %[vbias].h[1]         \n"
              "prfm   pldl1keep, [%[a_ptr], #64]  \n"
              "dup	v12.4h, %[vbias].h[2]         \n"
              "prfm   pldl1keep, [%[b_ptr], #128] \n"
              "dup	v14.4h, %[vbias].h[3]         \n"
              "prfm   pldl1keep, [%[b_ptr], #192] \n"
              "dup	v16.4h, %[vbias].h[4]         \n"
              "prfm   pldl1keep, [%[a_ptr], #128] \n"
              "dup	v18.4h, %[vbias].h[5]         \n"
              "prfm   pldl1keep, [%[b_ptr], #256] \n"
              "dup	v20.4h, %[vbias].h[6]         \n"
              "cmp    %w[has_beta], #1            \n"
              "prfm   pldl1keep, [%[a_ptr], #192] \n"
              "dup	v22.4h, %[vbias].h[7]         \n"
              "blt 1f                             \n"
              // process beta
              "ldr d0, [%[c_ptr0]]                \n"
              "ldr d2, [%[c_ptr1]]                \n"
              "ldr d4, [%[c_ptr2]]                \n"
              "ldr d6, [%[c_ptr3]]                \n"
              "ldr d1, [%[c_ptr4]]                \n"
              "fmla v8.4h, v0.4h, %[vbeta].4h     \n"
              "ldr d3, [%[c_ptr5]]                \n"
              "fmla v10.4h, v2.4h, %[vbeta].4h    \n"
              "ldr d5, [%[c_ptr6]]                \n"
              "fmla v12.4h, v4.4h, %[vbeta].4h    \n"
              "ldr d7, [%[c_ptr7]]                \n"
              "fmla v14.4h, v6.4h, %[vbeta].4h    \n"
              "fmla v16.4h, v1.4h, %[vbeta].4h    \n"
              "fmla v18.4h, v3.4h, %[vbeta].4h    \n"
              "fmla v20.4h, v5.4h, %[vbeta].4h    \n"
              "fmla v22.4h, v7.4h, %[vbeta].4h    \n"
              "1:                                 \n"
              "cmp %w[cnt], #1                    \n"
              "ldr q0, [%[a_ptr]], #16            \n"
              "ldr d2, [%[b_ptr]], #8             \n"
              "blt 2f                             \n"
              "0:                                 \n"
              // unrool 0
              FMLA_N00_4
              "ldr q1, [%[a_ptr]], #16            \n"
              "ldr d3, [%[b_ptr]], #8             \n"
              FMLA_N01_4
              "prfm   pldl1keep, [%[a_ptr], #64]  \n"
              "prfm   pldl1keep, [%[b_ptr], #128] \n"
              // unrool 1
              FMLA_N10_4
              "ldr q0, [%[a_ptr]], #16            \n"
              "ldr d2, [%[b_ptr]], #8             \n"
              "subs %w[cnt], %w[cnt], #1          \n"
              FMLA_N11_4
              "bne 0b                             \n"
              "2:                                 \n"
              "cmp %w[tail], #1                   \n"
              "beq 3f                             \n"
              // tail=2
              FMLA_N00_4
              "ldr q1, [%[a_ptr]], #16            \n"
              "ldr d3, [%[b_ptr]], #8             \n"
              FMLA_N01_4
              // unrool 1
              FMLA_N10_4
              FMLA_N11_4
              "b 6f                               \n"
              "3:                                 \n"
              // tail = 1
              FMLA_N00_4
              FMLA_N01_4
              "6:                                 \n"
              "cmp    %w[flag_act],   #1          \n"
              "beq 4f                             \n"
              "cmp    %w[flag_act],   #0          \n"
              "beq 7f                             \n"
              "cmp    %w[flag_act],   #2          \n"
              "beq 5f                             \n"
              // leakyRelu
              LEAKY_0_4
              LEAKY_1_4
              "b 7f                               \n"
              // relu
              "4:                                 \n"
              FMAX_4
              "b 7f                               \n"
              // relu6
              "5:                                 \n"
              FMAX_4
              FMIN_4
              "b 7f                               \n"
              // no relu
              "7:                                 \n"
              "st1 {v8.4h},  [%[c_ptr0]], #8      \n"
              "st1 {v10.4h}, [%[c_ptr1]], #8      \n"
              "st1 {v12.4h}, [%[c_ptr2]], #8      \n"
              "st1 {v14.4h}, [%[c_ptr3]], #8      \n"
              "st1 {v16.4h}, [%[c_ptr4]], #8      \n"
              "st1 {v18.4h}, [%[c_ptr5]], #8      \n"
              "st1 {v20.4h}, [%[c_ptr6]], #8      \n"
              "st1 {v22.4h}, [%[c_ptr7]], #8      \n"
              : [a_ptr] "+r"(a_ptr),
                [b_ptr] "+r"(b_ptr),
                [cnt] "+r"(k),
                [tail] "+r"(tail),
                [c_ptr0] "+r"(c_ptr0),
                [c_ptr1] "+r"(c_ptr1),
                [c_ptr2] "+r"(c_ptr2),
                [c_ptr3] "+r"(c_ptr3),
                [c_ptr4] "+r"(c_ptr4),
                [c_ptr5] "+r"(c_ptr5),
                [c_ptr6] "+r"(c_ptr6),
                [c_ptr7] "+r"(c_ptr7)
              : [has_beta] "r"(has_beta),
                [vbias] "w"(vbias),
                [vbeta] "w"(vbeta),
                [valpha] "w"(valpha),
                [vzero] "w"(vzero),
                [flag_act] "r"(flag_act)
              : "cc","memory",
                "v0","v1","v2","v3","v4","v5","v6","v7",
                "v8","v9","v10","v11","v12","v13",
                "v14","v15","v16","v17","v18","v19",
                "v20","v21","v22","v23"
            );
            // clang-format on
          }

          // remain process
          for (int i = 0; i < rem_rem; i++) {
            const float16_t *a_ptr = a_ptr_l;
            int tail = tail_pre;
            int k = k_pre;
            // clang-format off
            asm volatile(
              "prfm   pldl1keep, [%[a_ptr]]       \n"
              "prfm   pldl1keep, [%[b_ptr]]       \n"
              "dup	v8.4h, %[vbias].h[0]          \n"
              "prfm   pldl1keep, [%[b_ptr], #64]  \n"
              "dup	v10.4h, %[vbias].h[1]         \n"
              "prfm   pldl1keep, [%[a_ptr], #64]  \n"
              "dup	v12.4h, %[vbias].h[2]         \n"
              "prfm   pldl1keep, [%[b_ptr], #128] \n"
              "dup	v14.4h, %[vbias].h[3]         \n"
              "prfm   pldl1keep, [%[b_ptr], #192] \n"
              "dup	v16.4h, %[vbias].h[4]         \n"
              "prfm   pldl1keep, [%[a_ptr], #128] \n"
              "dup	v18.4h, %[vbias].h[5]         \n"
              "prfm   pldl1keep, [%[b_ptr], #256] \n"
              "dup	v20.4h, %[vbias].h[6]         \n"
              "cmp    %w[has_beta], #1            \n"
              "prfm   pldl1keep, [%[a_ptr], #192] \n"
              "dup	v22.4h, %[vbias].h[7]         \n"
              "blt 1f                             \n"
              // process beta
              "ldr d0, [%[c_ptr0]]                \n"
              "ldr d2, [%[c_ptr1]]                \n"
              "ldr d4, [%[c_ptr2]]                \n"
              "ldr d6, [%[c_ptr3]]                \n"
              "ldr d1, [%[c_ptr4]]                \n"
              "fmla v8.4h, v0.4h, %[vbeta].4h     \n"
              "ldr d3, [%[c_ptr5]]                \n"
              "fmla v10.4h, v2.4h, %[vbeta].4h    \n"
              "ldr d5, [%[c_ptr6]]                \n"
              "fmla v12.4h, v4.4h, %[vbeta].4h    \n"
              "ldr d7, [%[c_ptr7]]                \n"
              "fmla v14.4h, v6.4h, %[vbeta].4h    \n"
              "fmla v16.4h, v1.4h, %[vbeta].4h    \n"
              "fmla v18.4h, v3.4h, %[vbeta].4h    \n"
              "fmla v20.4h, v5.4h, %[vbeta].4h    \n"
              "fmla v22.4h, v7.4h, %[vbeta].4h    \n"
              "1:                                 \n"
              "cmp %w[cnt], #1                    \n"
              "movi v4.8h, #0                     \n"
              "movi v5.8h, #0                     \n"
              "ldp q0, q1, [%[a_ptr]], #32        \n"
              "ldr d2, [%[b_ptr]]                 \n"
              "blt 2f                             \n"
              "0:                                 \n"
              // unrool 0
              "add %[b_ptr], %[b_ptr], #4         \n"
              "prfm   pldl1keep, [%[a_ptr], #64]  \n"
              "prfm   pldl1keep, [%[b_ptr], #128] \n"
              "subs %w[cnt], %w[cnt], #1          \n"
              "fmla v4.8h, v0.8h, v2.h[0]         \n"
              "fmla v5.8h, v1.8h, v2.h[1]         \n"
              "ldp q0, q1, [%[a_ptr]], #32        \n"
              "ldr d2, [%[b_ptr]]                 \n"
              "bne 0b                             \n"
              "2:                                 \n"
              "cmp %w[tail], #1                   \n"
              "beq 3f                             \n"
              // tail=2
              "add %[b_ptr], %[b_ptr], #4         \n"
              "fmla v4.8h, v0.8h, v2.h[0]         \n"
              "fmla v5.8h, v1.8h, v2.h[1]         \n"
              "b 6f                               \n"
              "3:                                 \n"
              // tail = 1
              "sub %[a_ptr], %[a_ptr], #16        \n"
              "add %[b_ptr], %[b_ptr], #2         \n"
              "fmla v4.8h, v0.8h, v2.h[0]         \n"
              "6:                                 \n"
              "fadd v9.8h, v4.8h, v5.8h           \n"
              "cmp    %w[flag_act],   #1          \n"
              "ins  v0.h[0], v9.h[0]              \n"
              "ins  v1.h[0], v9.h[1]              \n"
              "ins  v2.h[0], v9.h[2]              \n"
              "ins  v3.h[0], v9.h[3]              \n"
              "beq 4f                             \n"
              "cmp    %w[flag_act],   #0          \n"
              "ins  v4.h[0], v9.h[4]              \n"
              "ins  v5.h[0], v9.h[5]              \n"
              "ins  v6.h[0], v9.h[6]              \n"
              "ins  v7.h[0], v9.h[7]              \n"
              "beq 7f                             \n"
              "cmp    %w[flag_act],   #2          \n"
              "fadd v8.4h, v8.4h, v0.4h           \n"
              "fadd v10.4h, v10.4h, v1.4h         \n"
              "fadd v12.4h, v12.4h, v2.4h         \n"
              "fadd v14.4h, v14.4h, v3.4h         \n"
              "beq 5f                             \n"
              "fadd v16.4h, v16.4h, v4.4h         \n"
              "fadd v18.4h, v18.4h, v5.4h         \n"
              "fadd v20.4h, v20.4h, v6.4h         \n"
              "fadd v22.4h, v22.4h, v7.4h         \n"
              // leakyRelu
              LEAKY_0_4
              LEAKY_1_4
              "b 8f                               \n"
              // relu
              "4:                                 \n"
              "ins  v4.h[0], v9.h[4]              \n"
              "ins  v5.h[0], v9.h[5]              \n"
              "ins  v6.h[0], v9.h[6]              \n"
              "ins  v7.h[0], v9.h[7]              \n"
              "fadd v8.4h, v8.4h, v0.4h           \n"
              "fadd v10.4h, v10.4h, v1.4h         \n"
              "fadd v12.4h, v12.4h, v2.4h         \n"
              "fadd v14.4h, v14.4h, v3.4h         \n"
              "fadd v16.4h, v16.4h, v4.4h         \n"
              "fadd v18.4h, v18.4h, v5.4h         \n"
              "fadd v20.4h, v20.4h, v6.4h         \n"
              "fadd v22.4h, v22.4h, v7.4h         \n"
              FMAX_4
              "b 8f                               \n"
              // relu6
              "5:                                 \n"
              "fadd v16.4h, v16.4h, v4.4h         \n"
              "fadd v18.4h, v18.4h, v5.4h         \n"
              "fadd v20.4h, v20.4h, v6.4h         \n"
              "fadd v22.4h, v22.4h, v7.4h         \n"
              FMAX_4
              FMIN_4
              "b 8f                               \n"
              // no relu
              "7:                                 \n"
              "fadd v8.4h,  v8.4h,  v0.4h         \n"
              "fadd v10.4h, v10.4h, v1.4h         \n"
              "fadd v12.4h, v12.4h, v2.4h         \n"
              "fadd v14.4h, v14.4h, v3.4h         \n"
              "fadd v16.4h, v16.4h, v4.4h         \n"
              "fadd v18.4h, v18.4h, v5.4h         \n"
              "fadd v20.4h, v20.4h, v6.4h         \n"
              "fadd v22.4h, v22.4h, v7.4h         \n"
              "8:                                 \n"
              "str  h8,     [%[c_ptr0]], #2       \n"
              "str  h10,    [%[c_ptr1]], #2       \n"
              "str  h12,    [%[c_ptr2]], #2       \n"
              "str  h14,    [%[c_ptr3]], #2       \n"
              "str  h16,    [%[c_ptr4]], #2       \n"
              "str  h18,    [%[c_ptr5]], #2       \n"
              "str  h20,    [%[c_ptr6]], #2       \n"
              "str  h22,    [%[c_ptr7]], #2       \n"
              : [a_ptr] "+r"(a_ptr),
                [b_ptr] "+r"(b_ptr),
                [cnt] "+r"(k),
                [tail] "+r"(tail),
                [c_ptr0] "+r"(c_ptr0),
                [c_ptr1] "+r"(c_ptr1),
                [c_ptr2] "+r"(c_ptr2),
                [c_ptr3] "+r"(c_ptr3),
                [c_ptr4] "+r"(c_ptr4),
                [c_ptr5] "+r"(c_ptr5),
                [c_ptr6] "+r"(c_ptr6),
                [c_ptr7] "+r"(c_ptr7)
              : [has_beta] "r"(has_beta),
                [vbias] "w"(vbias),
                [vbeta] "w"(vbeta),
                [valpha] "w"(valpha),
                [vzero] "w"(vzero),
                [flag_act] "r"(flag_act)
              : "cc","memory",
                "v0","v1","v2","v3","v4","v5","v6","v7",
                "v8","v9","v10","v11","v12","v13",
                "v14","v15","v16","v17","v18","v19",
                "v20","v21","v22","v23"
            );
            // clang-format on
          }
        } else {
          const float16_t *a_ptr = a_ptr_l;
          int tail = tail_pre;
          int k = k_pre;
          // clang-format off
          asm volatile(
            "prfm   pldl1keep, [%[a_ptr]]       \n"
            "prfm   pldl1keep, [%[b_ptr]]       \n"
            "dup	v8.8h, %[vbias].h[0]          \n"
            "dup	v9.8h, %[vbias].h[0]          \n"
            "prfm   pldl1keep, [%[b_ptr], #64]  \n"
            "dup	v10.8h, %[vbias].h[1]         \n"
            "dup	v11.8h, %[vbias].h[1]         \n"
            "prfm   pldl1keep, [%[a_ptr], #64]  \n"
            "dup	v12.8h, %[vbias].h[2]         \n"
            "dup	v13.8h, %[vbias].h[2]         \n"
            "prfm   pldl1keep, [%[b_ptr], #128] \n"
            "dup	v14.8h, %[vbias].h[3]         \n"
            "dup	v15.8h, %[vbias].h[3]         \n"
            "prfm   pldl1keep, [%[b_ptr], #192] \n"
            "dup	v16.8h, %[vbias].h[4]         \n"
            "dup	v17.8h, %[vbias].h[4]         \n"
            "prfm   pldl1keep, [%[a_ptr], #128] \n"
            "dup	v18.8h, %[vbias].h[5]         \n"
            "dup	v19.8h, %[vbias].h[5]         \n"
            "prfm   pldl1keep, [%[b_ptr], #256] \n"
            "dup	v20.8h, %[vbias].h[6]         \n"
            "dup	v21.8h, %[vbias].h[6]         \n"
            "cmp    %w[has_beta], #1            \n"
            "prfm   pldl1keep, [%[a_ptr], #192] \n"
            "dup	v22.8h, %[vbias].h[7]         \n"
            "dup	v23.8h, %[vbias].h[7]         \n"
            "blt 1f                             \n"
            // process beta
            "ldp q0, q1, [%[c_ptr0]]            \n"
            "ldp q2, q3, [%[c_ptr1]]            \n"
            "ldp q4, q5, [%[c_ptr2]]            \n"
            "ldp q6, q7, [%[c_ptr3]]            \n"
            "fmla v8.8h, v0.8h, %[vbeta].8h     \n"
            "fmla v9.8h, v1.8h, %[vbeta].8h     \n"
            "ldp q0, q1, [%[c_ptr4]]            \n"
            "fmla v10.8h, v2.8h, %[vbeta].8h    \n"
            "fmla v11.8h, v3.8h, %[vbeta].8h    \n"
            "ldp q2, q3, [%[c_ptr5]]            \n"
            "fmla v12.8h, v4.8h, %[vbeta].8h    \n"
            "fmla v13.8h, v5.8h, %[vbeta].8h    \n"
            "ldp q4, q5, [%[c_ptr6]]            \n"
            "fmla v14.8h, v6.8h, %[vbeta].8h    \n"
            "fmla v15.8h, v7.8h, %[vbeta].8h    \n"
            "ldp q6, q7, [%[c_ptr7]]            \n"
            "fmla v16.8h, v0.8h, %[vbeta].8h    \n"
            "fmla v17.8h, v1.8h, %[vbeta].8h    \n"
            "fmla v18.8h, v2.8h, %[vbeta].8h    \n"
            "fmla v19.8h, v3.8h, %[vbeta].8h    \n"
            "fmla v20.8h, v4.8h, %[vbeta].8h    \n"
            "fmla v21.8h, v5.8h, %[vbeta].8h    \n"
            "fmla v22.8h, v6.8h, %[vbeta].8h    \n"
            "fmla v23.8h, v7.8h, %[vbeta].8h    \n"
            "1:                                 \n"
            "cmp %w[cnt], #1                    \n"
            "ldr q0, [%[a_ptr]], #16            \n"
            "ldr q2, [%[b_ptr]], #16            \n"
            "blt 2f                             \n"
            "0:                                 \n"
            "ldr q3, [%[b_ptr]], #16            \n"
            // unrool 0
            FMLA_N00_8
            "prfm   pldl1keep, [%[a_ptr], #64]  \n"
            "prfm   pldl1keep, [%[b_ptr], #128] \n"
            "fmla v9.8h, v3.8h, v0.h[0]         \n"
            "fmla v11.8h, v3.8h, v0.h[1]        \n"
            "ldr q1, [%[a_ptr]], #16            \n"
            "fmla v13.8h, v3.8h, v0.h[2]        \n"
            "fmla v15.8h, v3.8h, v0.h[3]        \n"
            "ldr q4, [%[b_ptr]], #16            \n"
            "fmla v17.8h, v3.8h, v0.h[4]        \n"
            "fmla v19.8h, v3.8h, v0.h[5]        \n"
            "fmla v21.8h, v3.8h, v0.h[6]        \n"
            "fmla v23.8h, v3.8h, v0.h[7]        \n"
            // unrool 1
            "ldr q5, [%[b_ptr]], #16            \n"
            FMLA_N10_8
            "prfm   pldl1keep, [%[a_ptr], #64]  \n"
            "prfm   pldl1keep, [%[b_ptr], #128] \n"
            "fmla v9.8h, v5.8h, v1.h[0]         \n"
            "fmla v11.8h, v5.8h, v1.h[1]        \n"
            "ldr q0, [%[a_ptr]], #16            \n"
            "fmla v13.8h, v5.8h, v1.h[2]        \n"
            "fmla v15.8h, v5.8h, v1.h[3]        \n"
            "ldr q2, [%[b_ptr]], #16            \n"
            "subs %w[cnt], %w[cnt], #1          \n"
            "fmla v17.8h, v5.8h, v1.h[4]        \n"
            "fmla v19.8h, v5.8h, v1.h[5]        \n"
            "fmla v21.8h, v5.8h, v1.h[6]        \n"
            "fmla v23.8h, v5.8h, v1.h[7]        \n"
            "bne 0b                             \n"
            "2:                                 \n"
            "cmp %w[tail], #1                   \n"
            "ldr q3, [%[b_ptr]], #16            \n"
            "beq 3f                             \n"
            // tail=2
            FMLA_N00_8
            "prfm   pldl1keep, [%[a_ptr], #64]  \n"
            "prfm   pldl1keep, [%[b_ptr], #128] \n"
            "fmla v9.8h, v3.8h, v0.h[0]         \n"
            "fmla v11.8h, v3.8h, v0.h[1]        \n"
            "ldr q1, [%[a_ptr]], #16            \n"
            "fmla v13.8h, v3.8h, v0.h[2]        \n"
            "fmla v15.8h, v3.8h, v0.h[3]        \n"
            "ldr q4, [%[b_ptr]], #16            \n"
            "fmla v17.8h, v3.8h, v0.h[4]        \n"
            "fmla v19.8h, v3.8h, v0.h[5]        \n"
            "fmla v21.8h, v3.8h, v0.h[6]        \n"
            "fmla v23.8h, v3.8h, v0.h[7]        \n"
            // unrool 1
            "ldr q5, [%[b_ptr]], #16            \n"
            FMLA_N10_8
            FMLA_N11_8
            "b 6f                               \n"
            "3:                                 \n"
            // tail = 1
            FMLA_N00_8
            FMLA_N01_8
            "6:                                 \n"
            "cmp    %w[flag_act],   #1          \n"
            "beq 4f                             \n"
            "cmp    %w[flag_act],   #0          \n"
            "beq 7f                             \n"
            "cmp    %w[flag_act],   #2          \n"
            "beq 5f                             \n"
            // leakyRelu
            "fcmge v0.8h, v8.8h, %[vzero].8h    \n"
            "fmul v1.8h, v8.8h, %[valpha].8h     \n"
            "fcmge v2.8h, v9.8h, %[vzero].8h    \n"
            "fmul v3.8h, v9.8h, %[valpha].8h     \n"
            "fcmge v4.8h, v10.8h, %[vzero].8h   \n"
            "fmul v5.8h, v10.8h, %[valpha].8h    \n"
            "fcmge v6.8h, v11.8h, %[vzero].8h   \n"
            "fmul v7.8h, v11.8h, %[valpha].8h    \n"
            "bif  v8.16b, v1.16b, v0.16b        \n"
            "bif  v9.16b, v3.16b, v2.16b        \n"
            "bif  v10.16b, v5.16b, v4.16b       \n"
            "bif  v11.16b, v7.16b, v6.16b       \n"
            "fcmge v0.8h, v12.8h, %[vzero].8h   \n"
            "fmul v1.8h, v12.8h, %[valpha].8h    \n"
            "fcmge v2.8h, v13.8h, %[vzero].8h   \n"
            "fmul v3.8h, v13.8h, %[valpha].8h    \n"
            "fcmge v4.8h, v14.8h, %[vzero].8h   \n"
            "fmul v5.8h, v14.8h, %[valpha].8h    \n"
            "fcmge v6.8h, v15.8h, %[vzero].8h   \n"
            "fmul v7.8h, v15.8h, %[valpha].8h    \n"
            "bif  v12.16b, v1.16b, v0.16b       \n"
            "bif  v13.16b, v3.16b, v2.16b       \n"
            "bif  v14.16b, v5.16b, v4.16b       \n"
            "bif  v15.16b, v7.16b, v6.16b       \n"
            "fcmge v0.8h, v16.8h, %[vzero].8h   \n"
            "fmul v1.8h, v16.8h, %[valpha].8h    \n"
            "fcmge v2.8h, v17.8h, %[vzero].8h   \n"
            "fmul v3.8h, v17.8h, %[valpha].8h    \n"
            "fcmge v4.8h, v18.8h, %[vzero].8h   \n"
            "fmul v5.8h, v18.8h, %[valpha].8h    \n"
            "fcmge v6.8h, v19.8h, %[vzero].8h   \n"
            "fmul v7.8h, v19.8h, %[valpha].8h    \n"
            "bif  v16.16b, v1.16b, v0.16b       \n"
            "bif  v17.16b, v3.16b, v2.16b       \n"
            "bif  v18.16b, v5.16b, v4.16b       \n"
            "bif  v19.16b, v7.16b, v6.16b       \n"
            "fcmge v0.8h, v20.8h, %[vzero].8h   \n"
            "fmul v1.8h, v20.8h, %[valpha].8h    \n"
            "fcmge v2.8h, v21.8h, %[vzero].8h   \n"
            "fmul v3.8h, v21.8h, %[valpha].8h    \n"
            "fcmge v4.8h, v22.8h, %[vzero].8h   \n"
            "fmul v5.8h, v22.8h, %[valpha].8h    \n"
            "fcmge v6.8h, v23.8h, %[vzero].8h   \n"
            "fmul v7.8h, v23.8h, %[valpha].8h    \n"
            "bif  v20.16b, v1.16b, v0.16b       \n"
            "bif  v21.16b, v3.16b, v2.16b       \n"
            "bif  v22.16b, v5.16b, v4.16b       \n"
            "bif  v23.16b, v7.16b, v6.16b       \n"
            "b 7f                               \n"
            // relu
            "4:                                 \n"
            FMAX_8
            "b 7f                               \n"
            // relu6
            "5:                                 \n"
            FMAX_8
            FMIN_8
            "b 7f                               \n"
            // no relu
            "7:                                 \n"
            "stp q8, q9, [%[c_ptr0]], #32       \n"
            "stp q10, q11, [%[c_ptr1]], #32     \n"
            "stp q12, q13, [%[c_ptr2]], #32     \n"
            "stp q14, q15, [%[c_ptr3]], #32     \n"
            "stp q16, q17, [%[c_ptr4]], #32     \n"
            "stp q18, q19, [%[c_ptr5]], #32     \n"
            "stp q20, q21, [%[c_ptr6]], #32     \n"
            "stp q22, q23, [%[c_ptr7]], #32     \n"
            : [a_ptr] "+r"(a_ptr),
              [b_ptr] "+r"(b_ptr),
              [cnt] "+r"(k),
              [tail] "+r"(tail),
              [c_ptr0] "+r"(c_ptr0),
              [c_ptr1] "+r"(c_ptr1),
              [c_ptr2] "+r"(c_ptr2),
              [c_ptr3] "+r"(c_ptr3),
              [c_ptr4] "+r"(c_ptr4),
              [c_ptr5] "+r"(c_ptr5),
              [c_ptr6] "+r"(c_ptr6),
              [c_ptr7] "+r"(c_ptr7)
            : [has_beta] "r"(has_beta),
              [vbias] "w"(vbias),
              [vbeta] "w"(vbeta),
              [valpha] "w"(valpha),
              [vzero] "w"(vzero),
              [flag_act] "r"(flag_act)
            : "cc","memory",
              "v0","v1","v2","v3","v4","v5","v6","v7",
              "v8","v9","v10","v11","v12","v13",
              "v14","v15","v16","v17","v18","v19",
              "v20","v21","v22","v23"
          );
          // clang-format on
        }
      }
    }
  }
}
#else
#endif
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
