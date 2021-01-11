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

#include "lite/backends/arm/math/fp16/gemm_prepacked_fp16.h"
#include <arm_neon.h>
namespace paddle {
namespace lite {
namespace arm {
namespace math {
#ifdef __aarch64__
void prepackA_8x16(__fp16 *out,
                   const __fp16 *in,
                   const int ldin,
                   const int m0,
                   const int mmax,
                   const int k0,
                   const int kmax);
void prepackA_trans_8x16(__fp16 *out,
                         const __fp16 *in,
                         const int ldin,
                         const int m0,
                         const int mmax,
                         const int k0,
                         const int kmax);
void sgemm_prepack_8x16(const __fp16 *A_packed,
                        const __fp16 *B,
                        const __fp16 *bias,
                        __fp16 *C,
                        int M,
                        int N,
                        int K,
                        bool is_bias,
                        bool is_relu,
                        bool transB,
                        Context *ctx);
#endif

/**
 * \brief input data is not transpose
 * for arm-v7a, transform data to block x k x 6 layout
 * for arm-v8a, transform data to block x k x 8 layout
 */
void prepackA(void *out,
              const void *in,
              const int ldin,
              const int m0,
              const int mmax,
              const int k0,
              const int kmax,
              bool is_trans,
              Context *ctx) {
#ifdef __aarch64__
  if (is_trans) {
    prepackA_trans_8x16(static_cast<__fp16 *>(out),
                        static_cast<const __fp16 *>(in),
                        ldin,
                        m0,
                        mmax,
                        k0,
                        kmax);
  } else {
    prepackA_8x16(static_cast<__fp16 *>(out),
                  static_cast<const __fp16 *>(in),
                  ldin,
                  m0,
                  mmax,
                  k0,
                  kmax);
  }
#else
#endif
}

void prepackA_fp16(TensorLite *tout,
                   const TensorLite &tin,
                   int m,
                   int k,
                   int group,
                   bool is_trans,
                   ARMContext *ctx) {
  int hblock = get_hblock(ctx->get_arch());
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int group_size_round_up = ((m_roundup * k + 15) / 16) * 16;
  if (tout.valid_size() < group_size_round_up * group) {
    tout.reshape(Shape(group_size_round_up * group));
  }
  int lda = k;
  if (is_trans) {
    lda = m;
  }
  for (int g = 0; g < group; ++g) {
    const __fp16 *weights_group =
        static_cast<const __fp16 *>(tin.data()) + g * m * k;
    __fp16 *weights_trans_ptr =
        static_cast<__fp16 *>(tout.mutable_data()) + g * group_size_round_up;
    prepackA(weights_trans_ptr, weights_group, lda, 0, m, 0, k, is_trans, ctx);
  }
}

/// a: m*k  b: k*n  c: m*n
void sgemm_prepack_fp16(const __fp16 *A_packed,
                        const __fp16 *B,
                        const __fp16 *bias,
                        __fp16 *C,
                        int M,
                        int N,
                        int K,
                        bool is_bias,
                        bool is_relu,
                        bool is_transB,
                        Context *ctx) {
#ifdef __aarch64__
  sgemm_prepack_8x16(
      A_packed, B, bias, C, M, N, K, is_bias, is_relu, is_transB, ctx);
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
  "trn1 v3.4s, v9.4s, v11.4s            \n" /* e0f0g0h0a4b4c4d4 */ \
  "trn1 v4.4s, v12.4s, v14.4s           \n" /* e2f2g2h2a6b6c6d6 */ \
  "trn2 v5.4s, v12.4s, v14.4s           \n" /* e1f1g1h1a5b5c5d5 */ \
  "trn1 v6.4s, v13.4s, v15.4s           \n" /* e3f3g3h3a7b7c7d7 */ \
  "trn1 v7.4s, v13.4s, v15.4s           \n" /* 0-4 */              \
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
void prepackA_8x16(__fp16 *out,
                   const __fp16 *in,
                   const int ldin,
                   const int m0,
                   const int mmax,
                   const int k0,
                   const int kmax) {
  int x_len = kmax - k0;
  uint16_t zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(uint16_t) * x_len);

  uint16_t *dout = reinterpret_cast<uint16_t *>(out);
  const uint16_t *inptr = reinterpret_cast<const uint16_t *>(in);

  int cnt = x_len >> 3;
  int remain = x_len & 7;
  int stride = x_len * 8;
  int cnt_4 = remain >> 2;
  remain = remain & 3;
#pragma omp parallel for
  for (int y = m0; y < mmax; y += 8) {
    uint16_t *outptr = dout;
    const uint16_t *inptr0 = inptr + y * ldin + k0;
    const uint16_t *inptr1 = inptr0 + ldin;
    const uint16_t *inptr2 = inptr1 + ldin;
    const uint16_t *inptr3 = inptr2 + ldin;
    const uint16_t *inptr4 = inptr3 + ldin;
    const uint16_t *inptr5 = inptr4 + ldin;
    const uint16_t *inptr6 = inptr5 + ldin;
    const uint16_t *inptr7 = inptr6 + ldin;
    if ((y + 7) >= mmax) {
      switch ((y + 7) - mmax) {
        case 6:
          inptr1 = zerobuff;
        case 5:
          inptr2 = zerobuff;
        case 4:
          inptr3 = zerobuff;
        case 3:
          inptr4 = zerobuff;
        case 2:
          inptr5 = zerobuff;
        case 1:
          inptr6 = zerobuff;
        case 0:
          inptr7 = zerobuff;
        default:
          break;
      }
    }
    int cnt_col = cnt;
    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "prfm   pldl1keep, [%[ptr1]]                \n"
        "prfm   pldl1keep, [%[ptr2]]                \n"
        "prfm   pldl1keep, [%[ptr3]]                \n"
        "cmp %w[cnt], #1                            \n"
        "prfm   pldl1keep, [%[ptr4]]                \n"
        "prfm   pldl1keep, [%[ptr5]]                \n"
        "prfm   pldl1keep, [%[ptr6]]                \n"
        "prfm   pldl1keep, [%[ptr7]]                \n"
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
        "trn1 v10.8h, v2.8h, v3.8h                  \n"
        "ld1 {v7.8h}, [%[inptr7]], #16              \n"
        "trn2 v11.8h, v2.8h, v3.8h                  \n"
        // e0f0e2f2e4f4e6f6
        "trn1 v12.8h, v4.8h, v5.8h                  \n"
        "trn2 v13.8h, v4.8h, v5.8h                  \n" TRANS_C8
        // 0
        "st1 v8.8h, [%[outptr]], #16                \n"
        "st1 v12.8h, [%[outptr]], #16               \n"
        "st1 v10.8h, [%[outptr]], #16               \n"
        "st1 v14.8h, [%[outptr]], #16               \n"
        "subs %w[cnt], %w[cnt], #1                  \n"
        "st1 v9.8h, [%[outptr]], #16                \n"
        "st1 v13.8h, [%[outptr]], #16               \n"
        "st1 v11.8h, [%[outptr]], #16               \n"
        "st1 v15.8h, [%[outptr]], #16               \n"
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
        "trn1 v3.2s, v9.2s, v11.2s                  \n"
        // e0f0g0h0..
        "trn1 v4.2s, v12.2s, v14.2s                 \n"
        "st1 {v0.2s}, [%[outptr]], #8               \n"
        "trn2 v5.2s, v12.2s, v14.2s                 \n"
        "st1 {v4.2s}, [%[outptr]], #8               \n"
        "trn1 v6.2s, v13.2s, v15.2s                 \n"
        "st1 {v2.2s}, [%[outptr]], #8               \n"
        "trn2 v7.2s, v13.2s, v15.2s                 \n"
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
        : [cnt_4] "r"(cnt_4)
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
    dout += stride;
  }
}

void prepackA_trans_8x16(__fp16 *out,
                         const __fp16 *in,
                         const int ldin,
                         const int m0,
                         const int mmax,
                         const int k0,
                         const int kmax) {
  uint16_t *outptr = reinterpret_cast<uint16_t *>(out);
  const uint16_t *inptr =
      reinterpret_cast<const uint16_t *>(in) + k0 * ldin + m0;

  uint16_t mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int x_len = mmax - m0;
  int y_len = kmax - k0;
  int cnt = x_len >> 3;
  uint16_t right_remain = x_len & 7;
  int right_pad = 8 - right_remain;
  if (right_remain == 0) {
    right_pad = 0;
  }
  int stride_out = 8 * y_len;

  uint16x8_t vzero = vdupq_n_u16(0);
  uint16x8_t vmask =
      vcltq_u16(vld1q_u16(mask_buffer), vdupq_n_u16(right_remain));

#pragma omp parallel for
  for (int y = 0; y < y_len - 3; y += 4) {
    const uint16_t *ptr0 = inptr + y * ldin;
    const uint16_t *ptr1 = ptr0 + ldin;
    const uint16_t *ptr2 = ptr1 + ldin;
    const uint16_t *ptr3 = ptr2 + ldin;
    uint16_t *outptr_row_col = outptr + y * 8;
    int cnt_col = cnt;
    asm volatile(
        "cmp %w[cnt], #1                            \n"
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "prfm   pldl1keep, [%[ptr1]]                \n"
        "prfm   pldl1keep, [%[ptr2]]                \n"
        "prfm   pldl1keep, [%[ptr3]]                \n"
        "blt 1f                                     \n"
        "0:                                         \n"
        "ld1 {v0.8h}, [%[ptr0]], #16                \n"
        "ld1 {v1.8h}, [%[ptr1]], #16                \n"
        "ld1 {v2.8h}, [%[ptr2]], #16                \n"
        "ld1 {v3.8h}, [%[ptr3]], #16                \n"
        "subs %w[cnt], %w[cnt], #1                  \n"
        "str q0, [%[outptr]]                        \n"
        "str q1, [%[outptr], #16]                   \n"
        "str q2, [%[outptr], #32]                   \n"
        "str q3, [%[outptr], #48]                   \n"
        "add %[outputr], %[outptr], %w[stride]      \n"
        "bne 0b                                     \n"
        "1:                                         \n"
        "cmp %w[right_remain], #1                   \n"
        "blt 2f                                     \n"
        "ld1 {v0.8h}, [%[ptr0]]                     \n"
        "ld1 {v1.8h}, [%[ptr1]]                     \n"
        "ld1 {v2.8h}, [%[ptr2]]                     \n"
        "ld1 {v3.8h}, [%[ptr3]]                     \n"
        "bif v0.16b, %[vzero].16b, %[vmask].16b     \n"
        "bif v1.16b, %[vzero].16b, %[vmask].16b     \n"
        "bif v2.16b, %[vzero].16b, %[vmask].16b     \n"
        "bif v3.16b, %[vzero].16b, %[vmask].16b     \n"
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
          [vmask] "w"(vmask)
        : "cc", "memory", "v0", "v1", "v2", "v3");
  }
#pragma omp parallel for
  for (int y = 4 * (y_len / 4); y < y_len; ++y) {
    const uint16_t *ptr0 = inptr + y * ldin;
    uint16_t *outptr_row_col = outptr + y * 8;
    int cnt_col = cnt;
    asm volatile(
        "cmp %w[cnt], #1                            \n"
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "blt 1f                                     \n"
        "0:                                         \n"
        "ld1 {v0.8h}, [%[ptr0]], #16                \n"
        "subs %w[cnt], %w[cnt], #1                  \n"
        "str q0, [%[outptr]]                        \n"
        "add %[outputr], %[outptr], %w[stride]      \n"
        "bne 0b                                     \n"
        "1:                                         \n"
        "cmp %w[right_remain], #1                   \n"
        "blt 2f                                     \n"
        "ld1 {v0.8h}, [%[ptr0]]                     \n"
        "bif v0.16b, %[vzero].16b, %[vmask].16b     \n"
        "str q0, [%[outptr]]                        \n"
        "2:                                         \n"
        : [ptr0] "+r"(ptr0), [outptr] "+r"(outptr_row_col), [cnt] "+r"(cnt_col)
        : [right_remain] "r"(right_remain),
          [stride] "r"(stride_out),
          [vzero] "w"(vzero),
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
void loadb(__fp16 *out,
           const __fp16 *in,
           const int ldin,
           const int k0,
           const int kmax,
           const int n0,
           const int nmax) {
  uint16_t *outptr = reinterpret_cast<uint16_t *>(out);
  const uint16_t *inptr =
      reinterpret_cast<const uint16_t *>(in) + k0 * ldin + n0;
  uint16_t mask_buffer[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  int x_len = nmax - n0;
  int y_len = kmax - k0;
  int cnt = x_len >> 4;
  int right_remain = x_len & 15;

  uint16_t *outptr_row = outptr;
  int stride_out = 16 * y_len;

  uint16x8_t vzero = vdupq_n_u16(0);
  uint16x8_t vmask1 =
      vcltq_u16(vld1q_u16(mask_buffer), vdupq_n_u16(right_remain));
  uint16x8_t vmask2 =
      vcltq_u16(vld1q_u16(mask_buffer + 8), vdupq_n_u16(right_remain));

#pragma omp parallel for
  for (int y = 0; y < y_len - 3; y += 4) {
    const uint16_t *ptr0 = inptr + y * ldin;
    const uint16_t *ptr1 = ptr0 + ldin;
    const uint16_t *ptr2 = ptr1 + ldin;
    const uint16_t *ptr3 = ptr2 + ldin;

    uint16_t *outptr_row_col = outptr_row + y * 16;
    int cnt_col = cnt;
    asm volatile(
        "cmp %w[cnt], #1                            \n"
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "prfm   pldl1keep, [%[ptr1]]                \n"
        "prfm   pldl1keep, [%[ptr2]]                \n"
        "prfm   pldl1keep, [%[ptr3]]                \n"
        "blt 1f                                     \n"
        "0:                                         \n"
        "ldp q0, q1, [%[ptr0]], #32                 \n"
        "ldp q2, q3, [%[ptr1]], #32                 \n"
        "ldp q4, q5, [%[ptr2]], #32                 \n"
        "ldp q6, q7, [%[ptr3]], #32                 \n"
        "subs %w[cnt], %w[cnt], #1                  \n"
        "stp q0, q1, [%[outptr]]                    \n"
        "stp q2, q3, [%[outptr], #32]               \n"
        "stp q4, q5, [%[outptr], #64]               \n"
        "stp q6, q7, [%[outptr], #96]               \n"
        "add %[outputr], %[outptr], %w[stride]      \n"
        "bne 0b                                     \n"
        "1:                                         \n"
        "cmp %w[right_remain], #1                   \n"
        "blt 2f                                     \n"
        "ldp q0, q1, [%[ptr0]], #32                 \n"
        "ldp q2, q3, [%[ptr1]], #32                 \n"
        "ldp q4, q5, [%[ptr2]], #32                 \n"
        "ldp q6, q7, [%[ptr3]], #32                 \n"
        "bif v0.16b, %[vzero].16b, %[vmask1].16b    \n"
        "bif v1.16b, %[vzero].16b, %[vmask2].16b    \n"
        "bif v2.16b, %[vzero].16b, %[vmask1].16b    \n"
        "bif v3.16b, %[vzero].16b, %[vmask2].16b    \n"
        "bif v4.16b, %[vzero].16b, %[vmask1].16b    \n"
        "bif v5.16b, %[vzero].16b, %[vmask2].16b    \n"
        "stp q0, q1, [%[outptr]]                    \n"
        "bif v6.16b, %[vzero].16b, %[vmask1].16b    \n"
        "stp q2, q3, [%[outptr], #32]               \n"
        "bif v7.16b, %[vzero].16b, %[vmask2].16b    \n"
        "stp q4, q5, [%[outptr], #64]               \n"
        "stp q6, q7, [%[outptr], #96]               \n"
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
          [vmask1] "w"(vmask1),
          [vmask2] "w"(vmask2)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
  }

#pragma omp parallel for
  for (int y = 4 * (y_len / 4); y < y_len; ++y) {
    const uint16_t *ptr0 = inptr + y * ldin;
    uint16_t *outptr_row_col = outptr_row + y * 16;
    int cnt_col = cnt;

    asm volatile(
        "cmp %w[cnt], #1                            \n"
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "blt 1f                                     \n"
        "0:                                         \n"
        "ldp q0, q1, [%[ptr0]], #32                 \n"
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "subs %w[cnt], %w[cnt], #1                  \n"
        "stp q0, q1, [%[outptr]]                    \n"
        "add %[outputr], %[outptr], %w[stride]      \n"
        "bne 0b                                     \n"
        "1:                                         \n"
        "cmp %w[right_remain], #1                   \n"
        "blt 2f                                     \n"
        "ldp q0, q1, [%[ptr0]], #32                 \n"
        "bif v0.16b, %[vzero].16b, %[vmask1].16b    \n"
        "bif v1.16b, %[vzero].16b, %[vmask2].16b    \n"
        "stp q0, q1, [%[outptr]]                    \n"
        "2:                                         \n"
        : [ptr0] "+r"(ptr0), [outptr] "+r"(outptr_row_col), [cnt] "+r"(cnt_col)
        : [right_remain] "r"(right_remain),
          [stride] "r"(stride_out),
          [vzero] "w"(vzero),
          [vmask1] "w"(vmask1),
          [vmask2] "w"(vmask2)
        : "cc", "memory", "v0", "v1")
  }
}

void loadb_trans(__fp16 *out,
                 const __fp16 *in,
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
  uint16_t mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint16x8_t vzero = vdupq_n_u16(0);
  uint16x8_t vmask = vcltq_u16(vld1q_u16(mask_buffer), vdupq_n_u16(remain));

  //! data B is not transposed, transpose B to k * 16
  for (int y = n0; y < nmax; y += 16) {
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
    if ((y + 15) >= nmax) {
      switch ((y + 15) - nmax) {
        case 14:
          inptr1 = zerobuff;
        case 13:
          inptr2 = zerobuff;
        case 12:
          inptr3 = zerobuff;
        case 11:
          inptr4 = zerobuff;
        case 10:
          inptr5 = zerobuff;
        case 9:
          inptr6 = zerobuff;
        case 8:
          inptr7 = zerobuff;
        case 7:
          inptr8 = zerobuff;
        case 6:
          inptr9 = zerobuff;
        case 5:
          inptr10 = zerobuff;
        case 4:
          inptr11 = zerobuff;
        case 3:
          inptr12 = zerobuff;
        case 2:
          inptr13 = zerobuff;
        case 1:
          inptr14 = zerobuff;
        case 0:
          inptr15 = zerobuff;
        default:
          break;
      }
    }
    int cnt_col = cnt;
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
        "prfm   pldl1keep, [%[inptr15]]       \n" TRANS_C8
        "ld1 {v0.8h}, [%[inptr8]], #16        \n"
        "str q8, [%[outptr]]                  \n"
        "ld1 {v1.8h}, [%[inptr9]], #16        \n"
        "str q12, [%[outptr], #32]            \n"
        "ld1 {v2.8h}, [%[inptr10]], #16       \n"
        "st1 q10, [%[outptr], #64]            \n"
        "ld1 {v3.8h}, [%[inptr11]], #16       \n"
        "st1 q14, [%[outptr], #96]            \n"
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
        "prfm   pldl1keep, [%[inptr7]]        \n" TRANS_C8
        "sub %w[cnt], %w[cnt], #1             \n"
        "str q8, [%[outptr], #16]             \n"
        "str q12, [%[outptr], #48]            \n"
        "str q10, [%[outptr], #80]            \n"
        "str q14, [%[outptr], #112]           \n"
        "str q9, [%[outptr], #144]            \n"
        "str q13, [%[outptr], #176]           \n"
        "str q11, [%[outptr], #208]           \n"
        "str q15, [%[outptr], #240]           \n"
        "add %[outptr], %[outptr], #256\n"
        "bne 1b                               \n"
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
}

#else
#endif
#ifdef __aarch64__
void sgemm_prepack_8x16(const __fp16 *A_packed,
                        const __fp16 *B,
                        const __fp16 *bias,
                        __fp16 *C,
                        int M,
                        int N,
                        int K,
                        bool is_bias,
                        bool is_relu,
                        bool transB,
                        Context *ctx) {
  size_t llc_size = ctx->llc_size() / 4;
  auto workspace = ctx->workspace_data<__fp16>();
  int threads = ctx->threads();
  //! MBLOCK * x (result) + MBLOCK * k (A) + x * k (B) = l2
  int x_block = (llc_size - (MBLOCK * K)) / (sizeof(__fp16) * (K + MBLOCK));
  x_block /= NBLOCK_FP16;
  x_block *= NBLOCK_FP16;
  int x_num = (N + (x_block - 1)) / x_block;
  x_block = (N + x_num - 1) / x_num;
  x_block = (x_block + NBLOCK_FP16 - 1) / NBLOCK_FP16;
  x_block *= NBLOCK_FP16;
  x_block = x_block < NBLOCK_FP16 ? NBLOCK_FP16 : x_block;

  // unroll 2 loop
  int tail_pre = (K & (KBLOCK_FP16 - 1));
  int k_pre = ((K + KBLOCK_FP16 - 1) / KBLOCK_FP16) - 1;

  bool flag_p_remain = false;
  int remain = 0;
  float16x8_t vzero = vdupq_n_f16(0.f);

  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    if (xmax > N) {
      xmax = N;
    }
    int bblocks = (xmax - x0 + NBLOCK_FP16 - 1) / NBLOCK_FP16;
    remain = xmax - x0 - (bblocks - 1) * NBLOCK_FP16;
    if (remain > 0) {
      flag_p_remain = true;
    }
    //! load bpanel
    __fp16 *b_pannel = static_cast<__fp16 *>(workspace);
    if (transB) {
      loadb_trans(b_pannel, B, K, 0, K, x0, xmax);
    } else {
      loadb(b_pannel, B, N, 0, K, x0, xmax);
    }
#pragma omp parallel for num_threads(threads)
    for (unsigned int y = 0; y < M; y += MBLOCK) {
      unsigned int ymax = y + MBLOCK;
      if (ymax > M) {
        ymax = M;
      }
      __fp16 bias_local[8] = {0};
      if (is_bias) {
        if (y + 7 >= ymax) {
          switch ((y + 7) - ymax) {
            case 0:
              bias_local[7] = bias[y + 6];
            case 1:
              bias_local[6] = bias[y + 5];
            case 2:
              bias_local[5] = bias[y + 4];
            case 3:
              bias_local[4] = bias[y + 4];
            case 4:
              bias_local[3] = bias[y + 3];
            case 5:
              bias_local[2] = bias[y + 2];
            case 6:
              bias_local[1] = bias[y + 1] default : break;
          }
        } else {
          bias_local[0] = bias[y];
          bias_local[1] = bias[y + 1];
          bias_local[2] = bias[y + 2];
          bias_local[3] = bias[y + 3];
          bias_local[4] = bias[y + 4];
          bias_local[5] = bias[y + 5];
          bias_local[6] = bias[y + 6];
          bias_local[7] = bias[y + 7];
        }
      }

      __fp16 cout0[NBLOCK_FP16] = {0};
      __fp16 cout1[NBLOCK_FP16] = {0};
      __fp16 cout2[NBLOCK_FP16] = {0};
      __fp16 cout3[NBLOCK_FP16] = {0};
      __fp16 cout4[NBLOCK_FP16] = {0};
      __fp16 cout5[NBLOCK_FP16] = {0};
      __fp16 cout6[NBLOCK_FP16] = {0};
      __fp16 cout7[NBLOCK_FP16] = {0};

      __fp16 *c_ptr0 = C + y * N + x0;
      __fp16 *c_ptr1 = c_ptr0 + N;
      __fp16 *c_ptr2 = c_ptr1 + N;
      __fp16 *c_ptr3 = c_ptr2 + N;
      __fp16 *c_ptr4 = c_ptr3 + N;
      __fp16 *c_ptr5 = c_ptr4 + N;
      __fp16 *c_ptr6 = c_ptr5 + N;
      __fp16 *c_ptr7 = c_ptr6 + N;

      __fp16 *pout0 = c_ptr0;
      __fp16 *pout1 = c_ptr1;
      __fp16 *pout2 = c_ptr2;
      __fp16 *pout3 = c_ptr3;
      __fp16 *pout4 = c_ptr4;
      __fp16 *pout5 = c_ptr5;
      __fp16 *pout6 = c_ptr6;
      __fp16 *pout7 = c_ptr7;

      const __fp16 *a_ptr_l = A_packed + y * K;
      const __fp16 *b_ptr = b_pannel;
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
          pout0 = c_ptr0;
          pout1 = c_ptr1;
          pout2 = c_ptr2;
          pout3 = c_ptr3;
          pout4 = c_ptr4;
          pout5 = c_ptr5;
          pout6 = c_ptr6;
          pout7 = c_ptr7;

          c_ptr0 = cout0;
          c_ptr1 = cout1;
          c_ptr2 = cout2;
          c_ptr3 = cout3;
          c_ptr4 = cout4;
          c_ptr5 = cout5;
          c_ptr6 = cout6;
          c_ptr7 = cout7;
        }
        const __fp16 *a_ptr = a_ptr_l;
        asm volatile(
            "prfm   pldl1keep, [%[a_ptr]]        \n"
            "prfm   pldl1keep, [%[b_ptr]]        \n"
            "prfm   pldl1keep, [%[a_ptr], #64]   \n"
            "prfm   pldl1keep, [%[b_ptr], #64]   \n"
            "prfm   pldl1keep, [%[a_ptr], #128]   \n"
            "prfm   pldl1keep, [%[b_ptr], #128]   \n"
            "prfm   pldl1keep, [%[a_ptr], #192]   \n"
            "prfm   pldl1keep, [%[b_ptr], #192]   \n"
            "prfm   pldl1keep, [%[a_ptr], #256]   \n"
            "prfm   pldl1keep, [%[b_ptr], #256]   \n"
            "prfm   pldl1keep, [%[a_ptr], #320]   \n"
            "prfm   pldl1keep, [%[b_ptr], #320]   \n"
            "prfm   pldl1keep, [%[a_ptr], #384]   \n"
            "prfm   pldl1keep, [%[b_ptr], #384]   \n"
            :
            : [a_ptr] "r"(a_ptr), [b_ptr] "r"(b_ptr)
            : "memory");
        float16x8_t vout00 = vdupq_n_f16(bias_local[0]);
        float16x8_t vout01 = vdupq_n_f16(bias_local[0]);
        float16x8_t va = vld1q_f16(a_ptr);
        float16x8_t vout10 = vdupq_n_f16(bias_local[1]);
        float16x8_t vb0 = vld1q_f16(b_ptr);
        float16x8_t vout11 = vdupq_n_f16(bias_local[1]);
        float16x8_t vout20 = vdupq_n_f16(bias_local[2]);
        float16x8_t vout21 = vdupq_n_f16(bias_local[2]);
        float16x8_t vout30 = vdupq_n_f16(bias_local[3]);
        float16x8_t vout31 = vdupq_n_f16(bias_local[3]);
        float16x8_t vout40 = vdupq_n_f16(bias_local[4]);
        float16x8_t vout41 = vdupq_n_f16(bias_local[4]);
        float16x8_t vout50 = vdupq_n_f16(bias_local[5]);
        float16x8_t vout51 = vdupq_n_f16(bias_local[5]);
        float16x8_t vout60 = vdupq_n_f16(bias_local[6]);
        float16x8_t vout61 = vdupq_n_f16(bias_local[6]);
        float16x8_t vout70 = vdupq_n_f16(bias_local[7]);
        float16x8_t vout71 = vdupq_n_f16(bias_local[7]);
        for (int i = 0; i < k_pre; i++) {
          for (int j = 0; j < 8; j++) {
            vout00 = vfmaq_laneq_f16(vout00, vb0, va, 0);  // out0 = b0 * a00[0]
            vout10 = vfmaq_laneq_f16(vout10, vb0, va, 1);  // out1 = b0 * a00[1]
            vout20 = vfmaq_laneq_f16(vout20, vb0, va, 2);  // out2 = b0 * a00[2]
            vout30 = vfmaq_laneq_f16(vout30, vb0, va, 3);  // out3 = b0 * a00[3]
            float16x8_t vb1 = vld1q_f16(b_ptr + 8);
            vout40 = vfmaq_laneq_f16(vout40, vb0, va, 4);  // out4 = b0 * a00[4]
            vout50 = vfmaq_laneq_f16(vout50, vb0, va, 5);  // out5 = b0 * a00[5]
            vout60 = vfmaq_laneq_f16(vout60, vb0, va, 6);  // out6 = b0 * a00[6]
            vout70 = vfmaq_laneq_f16(vout70, vb1, va, 7);  // out7 = b0 * a00[7]
            a_ptr += 8;
            b_ptr += 16;
            vout01 = vfmaq_laneq_f16(vout01, vb1, va, 0);  // out0 = b0 * a00[0]
            vout11 = vfmaq_laneq_f16(vout11, vb1, va, 1);  // out1 = b0 * a00[1]
            vout21 = vfmaq_laneq_f16(vout21, vb1, va, 2);  // out2 = b0 * a00[2]
            vout31 = vfmaq_laneq_f16(vout31, vb1, va, 3);  // out3 = b0 * a00[3]
            va = vld1q_f16(a_ptr);
            vout41 = vfmaq_laneq_f16(vout41, vb1, va, 4);  // out4 = b0 * a00[4]
            vout51 = vfmaq_laneq_f16(vout51, vb1, va, 5);  // out5 = b0 * a00[5]
            vb0 = vld1q_f16(b_ptr);
            vout61 = vfmaq_laneq_f16(vout61, vb1, va, 6);  // out6 = b0 * a00[6]
            vout71 = vfmaq_laneq_f16(vout71, vb1, va, 7);  // out7 = b0 * a00[7]
          }
        }

        if (tail_pre > 0) {
          for (int i = 0; i < tail_pre; i++) {
            vout00 = vfmaq_laneq_f16(vout00, vb0, va, 0);  // out0 = b0 * a00[0]
            vout10 = vfmaq_laneq_f16(vout10, vb0, va, 1);  // out1 = b0 * a00[1]
            vout20 = vfmaq_laneq_f16(vout20, vb0, va, 2);  // out2 = b0 * a00[2]
            vout30 = vfmaq_laneq_f16(vout30, vb0, va, 3);  // out3 = b0 * a00[3]
            float16x8_t vb1 = vld1q_f16(b_ptr + 8);
            vout40 = vfmaq_laneq_f16(vout40, vb0, va, 4);  // out4 = b0 * a00[4]
            vout50 = vfmaq_laneq_f16(vout50, vb0, va, 5);  // out5 = b0 * a00[5]
            vout60 = vfmaq_laneq_f16(vout60, vb0, va, 6);  // out6 = b0 * a00[6]
            vout70 = vfmaq_laneq_f16(vout70, vb1, va, 7);  // out7 = b0 * a00[7]
            a_ptr += 8;
            b_ptr += 16;
            vout01 = vfmaq_laneq_f16(vout01, vb1, va, 0);  // out0 = b0 * a00[0]
            vout11 = vfmaq_laneq_f16(vout11, vb1, va, 1);  // out1 = b0 * a00[1]
            vout21 = vfmaq_laneq_f16(vout21, vb1, va, 2);  // out2 = b0 * a00[2]
            vout31 = vfmaq_laneq_f16(vout31, vb1, va, 3);  // out3 = b0 * a00[3]
            va = vld1q_f16(a_ptr);
            vout41 = vfmaq_laneq_f16(vout41, vb1, va, 4);  // out4 = b0 * a00[4]
            vout51 = vfmaq_laneq_f16(vout51, vb1, va, 5);  // out5 = b0 * a00[5]
            vb0 = vld1q_f16(b_ptr);
            vout61 = vfmaq_laneq_f16(vout61, vb1, va, 6);  // out6 = b0 * a00[6]
            vout71 = vfmaq_laneq_f16(vout71, vb1, va, 7);  // out7 = b0 * a00[7]
          }
        }

        if (is_relu) {
          vout00 = vmaxq_f16(vout00, vzero);
          vout01 = vmaxq_f16(vout01, vzero);
          vout10 = vmaxq_f16(vout10, vzero);
          vout11 = vmaxq_f16(vout11, vzero);
          vout20 = vmaxq_f16(vout20, vzero);
          vout21 = vmaxq_f16(vout21, vzero);
          vout30 = vmaxq_f16(vout30, vzero);
          vout31 = vmaxq_f16(vout31, vzero);
          vout40 = vmaxq_f16(vout40, vzero);
          vout41 = vmaxq_f16(vout41, vzero);
          vout50 = vmaxq_f16(vout50, vzero);
          vout51 = vmaxq_f16(vout51, vzero);
          vout60 = vmaxq_f16(vout60, vzero);
          vout61 = vmaxq_f16(vout61, vzero);
          vout70 = vmaxq_f16(vout70, vzero);
          vout71 = vmaxq_f16(vout71, vzero);
        }
        vst1q_f16(c_ptr0, vout00);
        vst1q_f16(c_ptr1, vout10);
        c_ptr0 += 8;
        vst1q_f16(c_ptr2, vout20);
        c_ptr1 += 8;
        vst1q_f16(c_ptr3, vout30);
        c_ptr2 += 8;
        vst1q_f16(c_ptr4, vout40);
        c_ptr3 += 8;
        vst1q_f16(c_ptr5, vout50);
        c_ptr4 += 8;
        vst1q_f16(c_ptr6, vout60);
        c_ptr5 += 8;
        vst1q_f16(c_ptr7, vout70);
        c_ptr6 += 8;
        vst1q_f16(c_ptr0, vout01);
        c_ptr7 += 8;
        vst1q_f16(c_ptr1, vout11);
        c_ptr0 += 8;
        vst1q_f16(c_ptr2, vout21);
        c_ptr1 += 8;
        vst1q_f16(c_ptr3, vout31);
        c_ptr2 += 8;
        vst1q_f16(c_ptr4, vout41);
        c_ptr3 += 8;
        vst1q_f16(c_ptr5, vout51);
        c_ptr4 += 8;
        vst1q_f16(c_ptr6, vout61);
        c_ptr5 += 8;
        vst1q_f16(c_ptr7, vout71);
        c_ptr6 += 8;
        c_ptr7 += 8;
        if (flag_p_remain && (xb == bblocks - 1)) {
          for (int i = 0; i < remain; ++i) {
            *pout0++ = cout0[i];
            *pout1++ = cout1[i];
            *pout2++ = cout2[i];
            *pout3++ = cout3[i];
            *pout4++ = cout4[i];
            *pout5++ = cout5[i];
            *pout6++ = cout6[i];
            *pout7++ = cout7[i];
          }
        }
      }
    }
  }
}
#else
#endif
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
