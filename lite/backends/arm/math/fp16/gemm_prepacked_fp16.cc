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
        "trn2 v9.8h, v0.8h, v1.8h                   \n"
        "ld1 {v5.8h}, [%[inptr5]], #16              \n"
        // c0d0c2d2c4d4c6d6
        "trn1 v10.8h, v2.8h, v3.8h                  \n"
        "trn2 v11.8h, v2.8h, v3.8h                  \n"
        "ld1 {v6.8h}, [%[inptr6]], #16              \n"
        // e0f0e2f2e4f4e6f6
        "trn1 v12.8h, v4.8h, v5.8h                  \n"
        "trn2 v13.8h, v4.8h, v5.8h                  \n"
        "ld1 {v7.8h}, [%[inptr7]], #16              \n"
        // a0b0c0d0a4b4c4d4
        "trn1 v0.4s, v8.4s, v10.4s                  \n"
        // a2b2c2d2a6b6c6d6
        "trn2 v1.4s, v8.4s, v10.4s                  \n"
        // g0h0g2h2g4h4g6h6
        "trn1 v14.8h, v6.8h, v7.8h                  \n"
        "trn2 v15.8h, v6.8h, v7.8h                  \n"
        // a1b1..a5b5..
        "trn1 v2.4s, v9.4s, v11.4s                  \n"
        // a3b3..a7b7..
        "trn1 v3.4s, v9.4s, v11.4s                  \n"
        // c0d0..
        "trn1 v4.4s, v12.4s, v14.4s                 \n"
        "trn2 v5.4s, v12.4s, v14.4s                 \n"
        "trn1 v6.4s, v13.4s, v15.4s                 \n"
        "trn2 v7.4s, v13.4s, v15.4s                 \n"
        // 0-4
        "trn1 v8.2d, v0.2d, v4.2d                   \n"
        "trn2 v9.2d, v0.2d, v4.2d                   \n"
        // 2-6
        "trn1 v10.2d, v1.2d, v5.2d                  \n"
        "trn2 v11.2d, v1.2d, v5.2d                  \n"
        // 1-5
        "trn1 v12.2d, v2.2d, v6.2d                   \n"
        "trn2 v13.2d, v2.2d, v6.2d                   \n"
        // 3-7
        "trn1 v14.2d, v3.2d, v7.2d                   \n"
        "trn2 v15.2d, v3.2d, v7.2d                   \n"
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
  uint16_t right_remain = x_len - 8 * (x_len / 8);
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
  int right_remain = x_len - 16 * (x_len / 16);
  int right_pad = 16 - right_remain;
  const size_t copy_len_remain = sizeof(__fp16) * right_remain;
  const size_t copy_len_pad = sizeof(__fp16) * right_pad;
  const size_t size_ldin = sizeof(__fp16) * ldin;

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
    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "prfm   pldl1keep, [%[ptr0], #64]   \n"
        "prfm   pldl1keep, [%[ptr1]]        \n"
        "prfm   pldl1keep, [%[ptr1], #64]   \n"
        "prfm   pldl1keep, [%[ptr2]]        \n"
        "prfm   pldl1keep, [%[ptr2], #64]   \n"
        "prfm   pldl1keep, [%[ptr3]]        \n"
        "prfm   pldl1keep, [%[ptr3], #64]   \n"
        :
        : [ptr0] "r"(ptr0), [ptr1] "r"(ptr1), [ptr2] "r"(ptr2), [ptr3] "r"(ptr3)
        : "memory");

    uint16_t *outptr_row_col = outptr_row + y * 16;

    int i = 0;
    for (; i < x_len - 11; i += 16) {
      uint16x8_t vr00 = vld1q_u16(ptr0);
      uint16x8_t vr02 = vld1q_u16(ptr0 + 8);

      uint16x8_t vr10 = vld1q_u16(ptr1);
      uint16x8_t vr12 = vld1q_u16(ptr1 + 8);

      vst1q_u16(outptr_row_col, vr00);
      vst1q_u16(outptr_row_col + 8, vr02);

      uint16x8_t vr20 = vld1q_u16(ptr2);
      uint16x8_t vr22 = vld1q_u16(ptr2 + 8);

      vst1q_u16(outptr_row_col + 12, vr10);
      vst1q_u16(outptr_row_col + 20, vr12);

      uint16x8_t vr30 = vld1q_u16(ptr3);
      uint16x8_t vr32 = vld1q_u16(ptr3 + 8);

      vst1q_u16(outptr_row_col + 24, vr20);
      vst1q_u16(outptr_row_col + 32, vr22);

      vst1q_u16(outptr_row_col + 36, vr30);
      vst1q_u16(outptr_row_col + 44, vr32);

      ptr0 += 16;
      ptr1 += 16;
      ptr2 += 16;
      ptr3 += 16;

      outptr_row_col += stride_out;
    }
    if (right_remain > 0) {
      uint16x8_t vr00 = vld1q_u16(ptr0);
      uint16x8_t vr02 = vld1q_u16(ptr0 + 8);
      uint16x8_t vr10 = vld1q_u16(ptr1);
      uint16x8_t vr12 = vld1q_u16(ptr1 + 8);

      uint16x8_t vr00_1 = vbslq_u16(vmask1, vr00, vzero);
      uint16x8_t vr02_1 = vbslq_u16(vmask2, vr02, vzero);
      uint16x8_t vr20 = vld1q_u16(ptr2);
      uint16x8_t vr22 = vld1q_u16(ptr2 + 8);

      vst1q_u16(outptr_row_col, vr00_1);
      vst1q_u16(outptr_row_col + 8, vr02_1);
      uint16x8_t vr10_1 = vbslq_u16(vmask1, vr10, vzero);
      uint16x8_t vr12_1 = vbslq_u16(vmask2, vr12, vzero);

      uint16x8_t vr30 = vld1q_u16(ptr3);
      uint16x8_t vr32 = vld1q_u16(ptr3 + 8);
      vst1q_u16(outptr_row_col + 12, vr10_1);
      vst1q_u16(outptr_row_col + 20, vr12_1);

      uint16x8_t vr20_1 = vbslq_u16(vmask1, vr20, vzero);
      uint16x8_t vr22_1 = vbslq_u16(vmask2, vr22, vzero);
      uint16x8_t vr30_1 = vbslq_u16(vmask1, vr30, vzero);
      uint16x8_t vr32_1 = vbslq_u16(vmask2, vr32, vzero);

      vst1q_u16(outptr_row_col + 24, vr20_1);
      vst1q_u16(outptr_row_col + 32, vr22_1);
      vst1q_u16(outptr_row_col + 36, vr30_1);
      vst1q_u16(outptr_row_col + 44, vr32_1);
    }
  }

#pragma omp parallel for
  for (int y = 4 * (y_len / 4); y < y_len; ++y) {
    const uint16_t *ptr0 = inptr + y * ldin;
    uint16_t *outptr_row_col = outptr_row + y * 16;

    int i = 0;
    for (; i < x_len - 11; i += 16) {
      uint16x8_t vr0 = vld1q_u16(ptr0);
      uint16x8_t vr2 = vld1q_u16(ptr0 + 8);
      vst1q_u16(outptr_row_col, vr0);
      vst1q_u16(outptr_row_col + 8, vr2);

      ptr0 += 16;
      outptr_row_col += stride_out;
    }
    if (right_remain > 0) {
      uint16x8_t vr0 = vld1q_u16(ptr0);
      uint16x8_t vr2 = vld1q_u16(ptr0 + 8);

      uint16x8_t vr0_1 = vbslq_u16(vmask1, vr0, vzero);
      uint16x8_t vr2_1 = vbslq_u16(vmask2, vr2, vzero);

      vst1q_u16(outptr_row_col, vr0);
      vst1q_u16(outptr_row_col + 8, vr2);
    }
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
  uint16_t zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(uint16_t) * x_len);
  uint16_t *outptr = reinterpret_cast<uint16_t *>(out);
  uint32_t *tmpptr = reinterpret_cast<uint32_t *>(outptr);
  const uint16_t *inptr = reinterpret_cast<const uint16_t *>(in);

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

    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "prfm   pldl1keep, [%[ptr0], #64]   \n"
        "prfm   pldl1keep, [%[ptr1]]        \n"
        "prfm   pldl1keep, [%[ptr1], #64]   \n"
        "prfm   pldl1keep, [%[ptr2]]        \n"
        "prfm   pldl1keep, [%[ptr2], #64]   \n"
        "prfm   pldl1keep, [%[ptr3]]        \n"
        "prfm   pldl1keep, [%[ptr3], #64]   \n"
        "prfm   pldl1keep, [%[ptr4]]        \n"
        "prfm   pldl1keep, [%[ptr4], #64]   \n"
        "prfm   pldl1keep, [%[ptr5]]        \n"
        "prfm   pldl1keep, [%[ptr5], #64]   \n"
        "prfm   pldl1keep, [%[ptr6]]        \n"
        "prfm   pldl1keep, [%[ptr6], #64]   \n"
        "prfm   pldl1keep, [%[ptr7]]        \n"
        "prfm   pldl1keep, [%[ptr7], #64]   \n"
        "prfm   pldl1keep, [%[ptr8]]        \n"
        "prfm   pldl1keep, [%[ptr8], #64]   \n"
        "prfm   pldl1keep, [%[ptr9]]        \n"
        "prfm   pldl1keep, [%[ptr9], #64]   \n"
        "prfm   pldl1keep, [%[ptr10]]        \n"
        "prfm   pldl1keep, [%[ptr10], #64]   \n"
        "prfm   pldl1keep, [%[ptr11]]        \n"
        "prfm   pldl1keep, [%[ptr11], #64]   \n"
        "prfm   pldl1keep, [%[ptr12]]        \n"
        "prfm   pldl1keep, [%[ptr12], #64]   \n"
        "prfm   pldl1keep, [%[ptr13]]        \n"
        "prfm   pldl1keep, [%[ptr13], #64]   \n"
        "prfm   pldl1keep, [%[ptr14]]        \n"
        "prfm   pldl1keep, [%[ptr14], #64]   \n"
        "prfm   pldl1keep, [%[ptr15]]        \n"
        "prfm   pldl1keep, [%[ptr15], #64]   \n"
        :
        : [ptr0] "r"(inptr0),
          [ptr1] "r"(inptr1),
          [ptr2] "r"(inptr2),
          [ptr3] "r"(inptr3),
          [ptr4] "r"(inptr4),
          [ptr5] "r"(inptr5),
          [ptr6] "r"(inptr6),
          [ptr7] "r"(inptr7),
          [ptr8] "r"(inptr8),
          [ptr9] "r"(inptr9),
          [ptr10] "r"(inptr10),
          [ptr11] "r"(inptr11),
          [ptr12] "r"(inptr12),
          [ptr13] "r"(inptr13),
          [ptr14] "r"(inptr14),
          [ptr15] "r"(inptr15)
        : "memory");

    int x = x_len;

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
    for (; x > 7; x -= 8) {
      uint16x8_t vin0 = vld1q_u16(inptr0);
      uint16x8_t vin1 = vld1q_u16(inptr1);
      uint16x8_t vin2 = vld1q_u16(inptr2);
      uint16x8_t vin3 = vld1q_u16(inptr3);
      uint16x8_t vin4 = vld1q_u16(inptr4);
      // v0=a0b0a2b2a4b4a6b6 v1=a1b1a3b3a5b5a7b7
      uint16x8x2_t vin0_1 = vtrnq_u16(vin0, vin1);
      uint16x8_t vin5 = vld1q_u16(inptr5);
      // v0=c0d0c2d2c4d4c6d6 v1=c1d1c3d3c5d5c7d7
      uint16x8x2_t vin2_3 = vtrnq_u16(vin2, vin3);
      uint16x8_t vin6 = vld1q_u16(inptr6);
      uint16x8_t vin7 = vld1q_u16(inptr7);
      inptr0 += 8;
      inptr1 += 8;
      // v0=e0f0e2f2d2c4d4c6d6 v1=c1d1c3d3c5d5c7d7
      uint16x8x2_t vin4_5 = vtrnq_u16(vin4, vin5);
      inptr2 += 8;
      inptr3 += 8;
      uint16x8_t vin8 = vld1q_u16(inptr8);
      uint16x8_t vin9 = vld1q_u16(inptr9);
      // v0=g0h0c2d2c4d4c6d6 v1=g1h1c3d3c5d5c7d7
      uint16x8x2_t vin6_7 = vtrnq_u16(vin6, vin7);

      vst1q_u16(outptr, vin0_1.val[0]);
      uint16x8_t vin10 = vld1q_u16(inptr10);
      uint16x8_t vin11 = vld1q_u16(inptr11);
      inptr4 += 8;
      outptr += 8;
      uint16x8x2_t vin8_9 = vtrnq_u16(vin8, vin9);
      uint16x8_t vin12 = vld1q_u16(inptr12);
      uint16x8_t vin13 = vld1q_u16(inptr13);
      vst1q_u16(outptr, vin2_3.val[0]);
      uint16x8x2_t vin10_11 = vtrnq_u16(vin10, vin11);
      inptr5 += 8;
      outptr += 8;
      uint16x8_t vin14 = vld1q_u16(inptr14);
      uint16x8_t vin15 = vld1q_u16(inptr15);
      uint16x8x2_t vin12_13 = vtrnq_u16(vin12, vin13);
      vst1q_u16(outptr, vin0_1.val[1]);
      inptr6 += 8;
      outptr += 8;
      uint16x8x2_t vin14_15 = vtrnq_u16(vin12, vin13);
      vst1q_u16(outptr, vin2_3.val[1]);
      inptr7 += 8;
      outptr += 8;

      vst1q_u16(outptr, vin4_5.val[0]);
      inptr8 += 8;
      outptr += 8;
      vst1q_u16(outptr, vin6_7.val[0]);
      inptr9 += 8;
      outptr += 8;

      // vout0=a0b0a2b2a4b4a6b6
      uint32x4_t vout0 = vld1q_u32(tmpptr);
      vst1q_u16(outptr, vin4_5.val[1]);
      inptr10 += 8;
      outptr += 8;
      // vout1=c0d0c2d2c4d4c6d6
      uint32x4_t vout1 = vld1q_u32(tmpptr + 4);
      vst1q_u16(outptr, vin6_7.val[1]);
      inptr11 += 8;
      outptr += 8;

      // vout2=a1b1a3b3a5b5a7b7
      uint32x4_t vout2 = vld1q_u32(tmpptr + 8);
      vst1q_u16(outptr, vin8_9.val[0]);
      inptr12 += 8;
      outptr += 8;

      // vout3=c1d1c3d3c5d5c7d7
      uint32x4_t vout3 = vld1q_u32(tmpptr + 12);
      vst1q_u16(outptr, vin10_11.val[0]);
      inptr13 += 8;
      outptr += 8;
      // val0 = a0b0c0d0q4b4c4d4 val1=a2b2c2d2a6b6c6d6
      uint32x4x2_t vout0_1 = vtrnq_u32(vout0, vout1);
      vst1q_u16(outptr, vin8_9.val[1]);
      inptr14 += 8;
      outptr += 8;

      uint32x4_t vout4 = vld1q_u32(tmpptr + 16);
      vst1q_u16(outptr, vin10_11.val[1]);
      inptr14 += 8;
      outptr += 8;

      uint32x4_t vout5 = vld1q_u32(tmpptr + 20);
      vst1q_u16(outptr, vin12_13.val[0]);
      inptr15 += 8;
      outptr += 8;

      // val0 = a1b1c1d1q5b5c5d5 val1=a3b3c3d3a7b7c7d7
      uint32x4x2_t vout2_3 = vtrnq_u32(vout2, vout3);
      vst1q_u16(outptr, vin14_15.val[0]);
      outptr += 8;

      uint32x4_t vout6 = vld1q_u32(tmpptr + 24);
      vst1q_u16(outptr, vin12_13.val[1]);
      outptr += 8;

      uint32x4_t vout7 = vld1q_u32(tmpptr + 28);
      vst1q_u16(outptr, vin14_15.val[1]);
      outptr += 8;
      // 0
      uint32x4_t vout8 = vld1q_u32(tmpptr + 32);
      vst1_u32(tmpptr, vget_low_u32(vout0_1.val[0]));
      // val0 = e0f0g0h0e4f4g4h4 val1=e2f2g2h2e6f6g6h6
      uint32x4x2_t vout4_5 = vtrnq_u32(vout4, vout5);
      uint32x4_t vout9 = vld1q_u32(tmpptr + 36);

      vst1_u32(tmpptr + 2, vget_low_u32(vout4_5.val[0]));
      uint32x4x2_t vout6_7 = vtrnq_u32(vout6, vout7);

      uint32x4_t vout10 = vld1q_u32(tmpptr + 40);
      uint32x4x2_t vout8_9 = vtrnq_u32(vout8, vout9);
      uint32x4_t vout11 = vld1q_u32(tmpptr + 44);

      // 1
      vst1_u32(tmpptr + 8, vget_low_u32(vout2_3.val[0]));
      uint32x4_t vout12 = vld1q_u32(tmpptr + 48);
      vst1_u32(tmpptr + 10, vget_low_u32(vout6_7.val[0]));
      uint32x4_t vout13 = vld1q_u32(tmpptr + 52);
      vst1_u32(tmpptr + 4, vget_low_u32(vout8_9.val[0]));
      uint32x4x2_t vout10_11 = vtrnq_u32(vout10, vout11);

      // 2
      vst1_u32(tmpptr + 16, vget_low_u32(vout0_1.val[1]));
      uint32x4x2_t vout12_13 = vtrnq_u32(vout12, vout13);
      uint32x4_t vout14 = vld1q_u32(tmpptr + 56);
      vst1_u32(tmpptr + 18, vget_low_u32(vout4_5.val[1]));
      uint32x4_t vout15 = vld1q_u32(tmpptr + 60);

      // 3
      vst1_u32(tmpptr + 6, vget_low_u32(vout12_13.val[0]));
      uint32x4x2_t vout14_15 = vtrnq_u32(vout14, vout15);
      vst1_u32(tmpptr + 24, vget_low_u32(vout2_3.val[1]));
      vst1_u32(tmpptr + 26, vget_low_u32(vout6_7.val[1]));

      // 4
      vst1_u32(tmpptr + 12, vget_low_u32(vout10_11.val[0]));
      vst1_u32(tmpptr + 32, vget_high_u32(vout0_1.val[0]));
      vst1_u32(tmpptr + 14, vget_low_u32(vout14_15.val[0]));
      vst1_u32(tmpptr + 34, vget_high_u32(vout4_5.val[0]));
      // 5
      vst1_u32(tmpptr + 20, vget_low_u32(vout8_9.val[1]));
      vst1_u32(tmpptr + 40, vget_high_u32(vout2_3.val[0]));
      vst1_u32(tmpptr + 22, vget_low_u32(vout12_13.val[1]));
      vst1_u32(tmpptr + 42, vget_high_u32(vout6_7.val[0]));
      // 6
      vst1_u32(tmpptr + 28, vget_low_u32(vout10_11.val[1]));
      vst1_u32(tmpptr + 48, vget_high_u32(vout0_1.val[1]));
      vst1_u32(tmpptr + 30, vget_low_u32(vout14_15.val[1]));
      vst1_u32(tmpptr + 50, vget_high_u32(vout4_5.val[1]));
      // 7
      vst1_u32(tmpptr + 36, vget_high_u32(vout8_9.val[0]));
      vst1_u32(tmpptr + 56, vget_high_u32(vout2_3.val[1]));
      vst1_u32(tmpptr + 38, vget_high_u32(vout12_13.val[0]));
      vst1_u32(tmpptr + 58, vget_high_u32(vout6_7.val[1]));

      vst1_u32(tmpptr + 44, vget_high_u32(vout10_11.val[0]));
      vst1_u32(tmpptr + 46, vget_high_u32(vout14_15.val[0]));
      vst1_u32(tmpptr + 52, vget_high_u32(vout8_9.val[1]));
      vst1_u32(tmpptr + 54, vget_high_u32(vout12_13.val[1]));
      vst1_u32(tmpptr + 60, vget_high_u32(vout10_11.val[1]));
      vst1_u32(tmpptr + 62, vget_high_u32(vout14_15.val[1]));
      tmpptr += 64;
      /*  asm volatile (
        // Load up 16 elements (3 vectors) from each of 8 sources.
        "ld1 {v0.8h}, [%[inptr0]], #32\n" // v0=a0a1a2a3a4a5a6a7
        "ld1 {v1.8h}, [%[inptr1]], #32\n" // v1=b0b1b2b3b4b5b6b7
        "ld1 {v2.8h}, [%[inptr2]], #32\n" // v2=c0c1c2c3c4c5c6c7
        "ld1 {v3.8h}, [%[inptr3]], #32\n" // v3=d0d1d2d3d4d5d6d7
        "ld1 {v4.8h}, [%[inptr4]], #32\n" // v4=e0e1e2e3e4e5e6e7
        "trn1 v12.8h, v0.8h, v1.8h\n"     // v12=a0b0a2b2a4b4a6b6
        "trn2 v13.8h, v0.8h, v1.8h\n"     // v13=a1b1q3b3q5b5a7b7
        "ld1 {v5.8h}, [%[inptr5]], #32\n" // v5=f0f1f2b3b4b5b6b7
        "ld1 {v6.8h}, [%[inptr6]], #32\n" // v6=g0g1g2c3c4c5c6c7
        "trn1 v14.8h, v2.8h, v3.8h\n"     // v14=c0d0c2d2c4d4c6d6
        "trn2 v15.8h, v2.8h, v3.8h\n"     // v15=c1d1c3d3c5d5c7d7

        "ld1 {v7.8h}, [%[inptr7]], #32\n" // v7=h0h1h2d3d4d5d6d7
        "ld1 {v8.8h}, [%[inptr8]], #32\n"  // v8=i0i1i2a3a4a5a6a7
        "trn1 v16.8h, v4.8h, v5.8h\n"      // v16=e0f0a2b2a4b4a6b6
        "trn2 v17.8h, v4.8h, v5.8h\n"      // v17=e1f1q3b3q5b5a7b7
        "ld1 {v9.8h}, [%[inptr9]], #32\n"  // v9=j0j1j2b3b4b5b6b7
        "trn1 v0.4s, v12.4s, v14.4s\n"     // v0=a0b0c0d0a4b4c4d4
        "trn2 v1.4s, v12.4s, v14.4s\n"     // v1=a2b2c2d2a6b6c6d6
        "ld1 {v10.8h}, [%[inptr10]], #32\n"// v10=k0k1k2c3c4c5c6c7
        "trn1 v2.4s, v13.4s, v15.4s\n"     // v2=a1b1c1d1a5b5c5d5
        "trn2 v3.4s, v13.4s, v15.4s\n"     // v3=a3b3c3d3a7b7c7d7
        "ld1 {v11.8h}, [%[inptr11]], #32\n"// v11=l0l1l2d3d4d5d6d7
        "trn1 v18.8h, v6.8h, v7.8h\n"      // v18=g0h0a2b2a4b4a6b6
        "trn2 v19.8h, v6.8h, v7.8h\n"      // v19=g1h1q3b3q5b5a7b7
        "ld1 {v24.8h}, [%[inptr12]], #32\n"// v11=l0l1l2d3d4d5d6d7
        "st1 {d0}, [%[outptr]], #16\n"     // v0=a0b0c0d0
        "trn1 v20.8h, v8.8h, v9.8h\n"      // v20=i0j0a2b2a4b4a6b6
        "trn2 v21.8h, v8.8h, v9.8h\n"      // v21=i1j1q3b3q5b5a7b7
        "ld1 {v25.8h}, [%[inptr13]], #32\n"// v11=l0l1l2d3d4d5d6d7
        "trn1 v22.8h, v10.8h, v10.8h\n"    // v20=k0l0a2b2a4b4a6b6
        "trn2 v23.8h, v11.8h, v11.8h\n"    // v21=k1l1q3b3q5b5a7b7
        "ld1 {v26.8h}, [%[inptr14]], #32\n"// v11=l0l1l2d3d4d5d6d7
        "trn1 v4.4s, v16.4s, v18.4s\n"     // v4=e0f0g0h0a4b4c4d4
        "trn2 v5.4s, v16.4s, v18.4s\n"     // v5=e2f2g2h2a6b6c6d6
        "ld1 {v27.8h}, [%[inptr15]], #32\n"// v11=l0l1l2d3d4d5d6d7
        "trn1 v6.4s, v17.4s, v19.4s\n"     // v6=e1f1g1h1a5b5c5d5
        "trn2 v7.4s, v17.4s, v19.4s\n"     // v7=e3f3g3h3a7b7c7d7

        "trn1 v28.8h, v24.8h, v25.8h\n"     // v12=a0b0a2b2a4b4a6b6
        "trn2 v29.8h, v24.8h, v25.8h\n"     // v13=a1b1q3b3q5b5a7b7
        "trn1 v8.4s, v20.4s, v22.4s\n"     // v4=e0f0g0h0a4b4c4d4
        "st1 {d8}, [%[outptr]], #16\n"     // v0=e0f0g0h0
        "trn2 v9.4s, v20.4s, v22.4s\n"     // v5=e2f2g2h2a6b6c6d6
        "trn1 v30.8h, v26.8h, v27.8h\n"     // v12=a0b0a2b2a4b4a6b6
        "trn2 v31.8h, v26.8h, v27.8h\n"     // v13=a1b1q3b3q5b5a7b7
        "trn1 v8.4s, v20.4s, v22.4s\n"     // v4=e0f0g0h0a4b4c4d4
        "trn1 v10.4s, v21.4s, v23.4s\n"    // v6=e1f1g1h1a5b5c5d5
        "trn2 v11.4s, v21.4s, v23.4s\n"    // v7=e3f3g3h3a7b7c7d7
        "st1 {d16}, [%[outptr]], #16\n"    // v0=i0j0k0l0
        "trn1 v24.4s, v28.4s, v30.4s\n"    // v6=a0b0c0d0a4b4c4d4
        "trn2 v25.4s, v28.4s, v30.4s\n"    // v7=a2b2c2d2a6b6c6d6
        "st1 {d4}, [%[outptr]], #16\n"     // v0=a1b1c1d1
        "trn1 v26.4s, v29.4s, v31.4s\n"    // v6=e1f1g1h1a5b5c5d5
        "trn2 v27.4s, v29.4s, v31.4s\n"    // v7=e3f3g3h3a7b7c7d7
        "st1 {d12}, [%[outptr]], #16\n"    // v0=e1f1g1h1
        "st1 {d20}, [%[outptr]], #16\n"    // v0=e1f1g1h1
        "st1 {d2}, [%[outptr]], #16\n"    // v0=a2b2c2d2
        "st1 {d10}, [%[outptr]], #16\n"    // v0=a2b2c2d2
        "st1 {d18}, [%[outptr]], #16\n"    // v0=a2b2c2d2
        "st1 {d6}, [%[outptr]], #16\n"    // v0=a3b3c3d3
        "st1 {d14}, [%[outptr]], #16\n"    // v0=a3b3c3d3
        "st1 {d22}, [%[outptr]], #16\n"    // v0=a3b3c3d3
        "st1 {d1}, [%[outptr]], #16\n"    // v0=a4b4c4d4
        "st1 {d9}, [%[outptr]], #16\n"    // v0=a4b4c4d4
        "st1 {d17}, [%[outptr]], #16\n"    // v0=a4b4c4d4
        "st1 {d5}, [%[outptr]], #16\n"    // v0=a5b5c5d5
        "st1 {d13}, [%[outptr]], #16\n"    // v0=a5b5c5d5
        "st1 {d21}, [%[outptr]], #16\n"    // v0=a5b5c5d5
        "st1 {d3}, [%[outptr]], #16\n"    // v0=a6b6c6d6
        "st1 {d11}, [%[outptr]], #16\n"    // v0=a6b6c6d6
        "st1 {d19}, [%[outptr]], #16\n"    // v0=a6b6c6d6
        "st1 {d7}, [%[outptr]], #16\n"    // v0=a7b7c7d7
        "st1 {d15}, [%[outptr]], #16\n"    // v0=a7b7c7d7
        "st1 {d23}, [%[outptr]], #16\n"    // v0=a7b7c7d7
        : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
        [inptr3] "+r"(inptr3), \
         [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5), [inptr6] "+r"(inptr6),
        [inptr7] "+r"(inptr7), \
         [inptr8] "+r"(inptr8), [inptr9] "+r"(inptr9), [inptr10] "+r"(inptr10),
        [inptr11] "+r"(inptr11), \
         [inptr12] "+r"(inptr12), [inptr13] "+r"(inptr13), [inptr14]
        "+r"(inptr14), [inptr15] "+r"(inptr15), \
          [outptr] "+r"(outptr)
        :
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12",
                "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
        "v22", "v23",
                "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc",
        "memory"
        );*/
    }

    for (; x > 0; x--) {
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
