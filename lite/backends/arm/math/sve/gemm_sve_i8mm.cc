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

#include "lite/backends/arm/math/sve/gemm_sve_i8mm.h"
#include <arm_neon.h>
#include <arm_sve.h>
#include "lite/backends/arm/math/sve/funcs_sve.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace sve {
/**
 * \brief input data is not transpose, (mmax - m0) * (kmax-k0)
 * kup = (K + 7) / 8
 * for sve, transform data to block x kup x c8 x 8 layout
 * din = a00a01..a0k;a10a11..a1k;...:am0am1..amk
 * dout =
 * a00a01..a07a10a11..a17...a70a71..a77;a08a09..a015a18a19...;a80a81..a87a90a91...
 */
void prepackA_m8k8_int8_sve(int8_t* out,
                            const int8_t* in,
                            int ldin,
                            int m0,
                            int mmax,
                            int k0,
                            int kmax) {
  int x_len = (kmax - k0);
  int kup = ROUNDUP_SVE(x_len, 8);
  int8_t zerobuff[kup];  // NOLINT
  memset(zerobuff, 0, sizeof(int8_t) * kup);

  int8_t* dout = out;
  const int8_t* inptr = in;
  int cnt = x_len >> 4;
  int remain = x_len % 16;
  int rem_cnt = remain >> 3;
  int rem_rem = remain % 8;
  LITE_PARALLEL_COMMON_BEGIN(y, tid, mmax, m0, 8) {
    int8_t* outptr = dout + kup * (y - m0);
    const int8_t* inptr_row[8];
    inptr_row[0] = inptr + y * ldin + k0;
    for (int i = 1; i < 8; i++) {
      inptr_row[i] = inptr_row[i - 1] + ldin;
    }
    //! cope with row index exceed real size, set to zero buffer
    if ((y + 7) >= mmax) {
      switch ((y + 7) - mmax) {
        case 6:
          inptr_row[1] = zerobuff;
        case 5:
          inptr_row[2] = zerobuff;
        case 4:
          inptr_row[3] = zerobuff;
        case 3:
          inptr_row[4] = zerobuff;
        case 2:
          inptr_row[5] = zerobuff;
        case 1:
          inptr_row[6] = zerobuff;
        case 0:
          inptr_row[7] = zerobuff;
        default:
          break;
      }
    }
    int cnt_col = cnt;
    asm volatile(
        "ptrue p0.b\n"
        "cbz %x[cnt], 2f\n"
        "1: \n"
        "ld1b {z0.b}, p0/Z, [%x[inptr0]]\n"
        "ld1b {z1.b}, p0/Z, [%x[inptr1]]\n"
        "ld1b {z2.b}, p0/Z, [%x[inptr2]]\n"
        "ld1b {z3.b}, p0/Z, [%x[inptr3]]\n"
        "ld1b {z4.b}, p0/Z, [%x[inptr4]]\n"
        "ld1b {z5.b}, p0/Z, [%x[inptr5]]\n"
        "ld1b {z6.b}, p0/Z, [%x[inptr6]]\n"
        "ld1b {z7.b}, p0/Z, [%x[inptr7]]\n"
        "subs %x[cnt], %x[cnt], #1\n"
        // zip-64
        "trn1 z8.d,  z0.d, z1.d\n"  // a0-7b0-7
        "trn2 z12.d, z0.d, z1.d\n"  // a8-15b8-15
        "trn1 z9.d,  z2.d, z3.d\n"  // c0-7d0-7
        "trn2 z13.d, z2.d, z3.d\n"  // c8-15d8-15
        "trn1 z10.d, z4.d, z5.d\n"  // e0-7f0-7
        "trn1 z11.d, z6.d, z7.d\n"  // g0-7h0-7
        "trn2 z14.d, z4.d, z5.d\n"  // e8-15f8-15
        "trn2 z15.d, z6.d, z7.d\n"  // g8-15g8-15

        "add %x[inptr0], %x[inptr0], #0x10\n"
        "add %x[inptr1], %x[inptr1], #0x10\n"
        "add %x[inptr2], %x[inptr2], #0x10\n"
        "add %x[inptr3], %x[inptr3], #0x10\n"
        "add %x[inptr4], %x[inptr4], #0x10\n"
        "add %x[inptr5], %x[inptr5], #0x10\n"
        "add %x[inptr6], %x[inptr6], #0x10\n"
        "add %x[inptr7], %x[inptr7], #0x10\n"

        "st1b {z8.b},  p0, [%x[outptr]]\n"
        "st1b {z9.b},  p0, [%x[outptr], #1, MUL VL]\n"
        "st1b {z10.b}, p0, [%x[outptr], #2, MUL VL]\n"
        "st1b {z11.b}, p0, [%x[outptr], #3, MUL VL]\n"
        "st1b {z12.b}, p0, [%x[outptr], #4, MUL VL]\n"
        "st1b {z13.b}, p0, [%x[outptr], #5, MUL VL]\n"
        "st1b {z14.b}, p0, [%x[outptr], #6, MUL VL]\n"
        "st1b {z15.b}, p0, [%x[outptr], #7, MUL VL]\n"
        "addvl %x[outptr], %x[outptr], #8\n"
        "bne 1b\n"
        "2: \n"
        "cbz %x[rem_cnt], 3f\n"
        "ld1b {z0.b}, p0/Z, [%x[inptr0]]\n"
        "ld1b {z1.b}, p0/Z, [%x[inptr1]]\n"
        "ld1b {z2.b}, p0/Z, [%x[inptr2]]\n"
        "ld1b {z3.b}, p0/Z, [%x[inptr3]]\n"
        "ld1b {z4.b}, p0/Z, [%x[inptr4]]\n"
        "ld1b {z5.b}, p0/Z, [%x[inptr5]]\n"
        "ld1b {z6.b}, p0/Z, [%x[inptr6]]\n"
        "ld1b {z7.b}, p0/Z, [%x[inptr7]]\n"
        "add %x[inptr0], %x[inptr0], #0x08\n"
        "add %x[inptr1], %x[inptr1], #0x08\n"
        "add %x[inptr2], %x[inptr2], #0x08\n"
        "add %x[inptr3], %x[inptr3], #0x08\n"
        "add %x[inptr4], %x[inptr4], #0x08\n"
        "add %x[inptr5], %x[inptr5], #0x08\n"
        "add %x[inptr6], %x[inptr6], #0x08\n"
        "add %x[inptr7], %x[inptr7], #0x08\n"
        // zip-64
        "trn1 z8.d,  z0.d, z1.d\n"  // a0-7b0-7
        "trn1 z9.d,  z2.d, z3.d\n"  // c0-7d0-7
        "trn1 z10.d, z4.d, z5.d\n"  // e0-7f0-7
        "trn1 z11.d, z6.d, z7.d\n"  // g0-7h0-7
        "st1b {z8.b},  p0, [%x[outptr]]\n"
        "st1b {z9.b},  p0, [%x[outptr], #1, MUL VL]\n"
        "st1b {z10.b}, p0, [%x[outptr], #2, MUL VL]\n"
        "st1b {z11.b}, p0, [%x[outptr], #3, MUL VL]\n"
        "add %x[outptr], %x[outptr], #0x40\n"
        "3: \n"
        : [inptr0] "+r"(inptr_row[0]),
          [inptr1] "+r"(inptr_row[1]),
          [inptr2] "+r"(inptr_row[2]),
          [inptr3] "+r"(inptr_row[3]),
          [inptr4] "+r"(inptr_row[4]),
          [inptr5] "+r"(inptr_row[5]),
          [inptr6] "+r"(inptr_row[6]),
          [inptr7] "+r"(inptr_row[7]),
          [outptr] "+r"(outptr),
          [cnt] "+r"(cnt_col)
        : [rem_cnt] "r"(rem_cnt)
        : "cc",
          "memory",
          "p0",
          "z0",
          "z1",
          "z2",
          "z3",
          "z4",
          "z5",
          "z6",
          "z7",
          "z8",
          "z9",
          "z10",
          "z11",
          "z12",
          "z13",
          "z14",
          "z15");
    if (rem_rem) {
      for (int i = 0; i < 8; i++) {
        for (int j = rem_rem; j > 0; j--) {
          *outptr++ = *inptr_row[i]++;
        }
        for (int j = 0; j < 8 - rem_rem; j++) {
          *outptr++ = 0;
        }
      }
    }
  }
  LITE_PARALLEL_COMMON_END();
}

#define ZIP8x16_16x8                                                          \
  /* zip-8 */                                                                 \
  "trn1 z8.b,   z0.b, z1.b\n"                        /* a0b0a2b2...a14b14*/   \
  "trn2 z9.b,   z0.b, z1.b\n"                        /* a1b1a3b3...a15b15*/   \
  "trn1 z10.b,  z2.b, z3.b\n"                        /* c0d0c2d2...c14d14*/   \
  "trn2 z11.b,  z2.b, z3.b\n"                        /* c1d1c3d3...c15c15*/   \
  "trn1 z12.b,  z4.b, z5.b\n"                        /* e0f0e2f2...e14f14*/   \
  "trn2 z13.b,  z4.b, z5.b\n"                        /* e1f1e3f3...e15f15*/   \
  "trn1 z14.b,  z6.b, z7.b\n"                        /* g0h0g2h2...g14h14*/   \
  "trn2 z15.b,  z6.b, z7.b\n" /* g1h1g3h3...g15h15*/ /* zip-16 */             \
  "trn1 z0.h,   z8.h,  z10.h\n"                      /*  a0b0c0d0a4....*/     \
  "trn2 z1.h,   z8.h,  z10.h\n"                      /* a2b2c2d2a6...*/       \
  "trn1 z2.h,   z9.h,  z11.h\n"                      /* a1b1c1d1a5..*/        \
  "trn2 z3.h,   z9.h,  z11.h\n"                      /* a3b3c3d3a7...*/       \
  "trn1 z4.h,   z12.h, z14.h\n"                      /* e0f0g0h0e4...*/       \
  "trn2 z5.h,   z12.h, z14.h\n"                      /* e2f2g2h2e6...*/       \
  "trn1 z6.h,   z13.h, z15.h\n"                      /* e1f1g1h1e5...*/       \
  "trn2 z7.h,   z13.h, z15.h\n" /* e3f3g3h3e7...*/   /* zip-32 */             \
  "trn1 z8.s,   z0.s, z4.s\n"                        /* a0b0c0d0e0f0g0h0a8*/  \
  "trn2 z9.s,   z0.s, z4.s\n"                        /* a4b4c4d4e4f4g4h4a12*/ \
  "trn1 z10.s,  z1.s, z5.s\n"                        /* a2b2c2d2e2f2g2h2a10*/ \
  "trn2 z11.s,  z1.s, z5.s\n"                        /* a6b6c6d6e6646464a14*/ \
  "trn1 z12.s,  z2.s, z6.s\n"                        /* a1b1c1d1e1f1g1h1a9*/  \
  "trn2 z13.s,  z2.s, z6.s\n"                        /* a5b5c5d5e5f5g5hha13*/ \
  "trn1 z14.s,  z3.s, z7.s\n"                        /* a3b3c3d3e3f3g3h3a11*/ \
  "trn2 z15.s,  z3.s, z7.s\n"                        /* a7b7c7d7e7f7g7h7a15*/

void prepackA_m8k8_trans_int8_sve(int8_t* out,
                                  const int8_t* in,
                                  int ldin,
                                  int m0,
                                  int mmax,
                                  int k0,
                                  int kmax) {
  int x_len = (mmax - m0);
  int kup = ROUNDUP_SVE(x_len, 8);
  int8_t zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(int8_t) * kup);

  int8_t* dout = out;
  const int8_t* inptr = in;
  int cnt = x_len >> 4;
  int remain = x_len % 16;
  int rem_cnt = remain >> 3;
  int rem_rem = remain % 8;
  int stride = 8 * ROUNDUP_SVE(kmax - k0, 8);

  LITE_PARALLEL_COMMON_BEGIN(y, tid, kmax, k0, 8) {
    int8_t* outptr = dout + (y - k0) * 8;
    const int8_t* inptr_row[8];
    inptr_row[0] = inptr + y * ldin + m0;
    for (int i = 1; i < 8; i++) {
      inptr_row[i] = inptr_row[i - 1] + ldin;
    }
    //! cope with row index exceed real size, set to zero buffer
    if ((y + 7) >= kmax) {
      switch ((y + 7) - kmax) {
        case 6:
          inptr_row[1] = zerobuff;
        case 5:
          inptr_row[2] = zerobuff;
        case 4:
          inptr_row[3] = zerobuff;
        case 3:
          inptr_row[4] = zerobuff;
        case 2:
          inptr_row[5] = zerobuff;
        case 1:
          inptr_row[6] = zerobuff;
        case 0:
          inptr_row[7] = zerobuff;
        default:
          break;
      }
    }
    int cnt_col = cnt;
    asm volatile(
        "ptrue p0.b\n"
        "cbz %x[cnt], 2f\n"
        "1: \n"
        "ld1b {z0.b}, p0/Z, [%x[inptr0]]\n"
        "ld1b {z1.b}, p0/Z, [%x[inptr1]]\n"
        "ld1b {z2.b}, p0/Z, [%x[inptr2]]\n"
        "ld1b {z3.b}, p0/Z, [%x[inptr3]]\n"
        "ld1b {z4.b}, p0/Z, [%x[inptr4]]\n"
        "ld1b {z5.b}, p0/Z, [%x[inptr5]]\n"
        "ld1b {z6.b}, p0/Z, [%x[inptr6]]\n"
        "ld1b {z7.b}, p0/Z, [%x[inptr7]]\n"
        "subs %x[cnt], %x[cnt], #1\n" ZIP8x16_16x8
        // zip-64
        "trn1 z0.d,  z8.d,  z12.d\n"  // 0-1
        "trn2 z1.d,  z8.d,  z12.d\n"  // 8-9
        "trn1 z2.d,  z9.d,  z13.d\n"  // 4-5
        "trn2 z3.d,  z9.d,  z13.d\n"  // 12-13
        "trn1 z4.d,  z10.d, z14.d\n"  // 2-3
        "trn2 z5.d,  z10.d, z14.d\n"  // 10-11
        "trn1 z6.d,  z11.d, z15.d\n"  // 6-7
        "trn2 z7.d,  z11.d, z15.d\n"  // 14-15

        "add %x[inptr0], %x[inptr0], #0x10\n"
        "add %x[inptr1], %x[inptr1], #0x10\n"
        "add %x[inptr2], %x[inptr2], #0x10\n"
        "add %x[inptr3], %x[inptr3], #0x10\n"
        "add %x[inptr4], %x[inptr4], #0x10\n"
        "add %x[inptr5], %x[inptr5], #0x10\n"
        "add %x[inptr6], %x[inptr6], #0x10\n"
        "add %x[inptr7], %x[inptr7], #0x10\n"

        "st1b {z0.b}, p0, [%x[outptr]]\n"
        "st1b {z4.b}, p0, [%x[outptr], #1, MUL VL]\n"
        "st1b {z2.b}, p0, [%x[outptr], #2, MUL VL]\n"
        "st1b {z6.b}, p0, [%x[outptr], #3, MUL VL]\n"
        "add  %x[outptr], %x[outptr], %x[stride] \n"
        "st1b {z1.b}, p0, [%x[outptr], #0, MUL VL]\n"
        "st1b {z5.b}, p0, [%x[outptr], #1, MUL VL]\n"
        "st1b {z3.b}, p0, [%x[outptr], #2, MUL VL]\n"
        "st1b {z7.b}, p0, [%x[outptr], #3, MUL VL]\n"
        "add  %x[outptr], %x[outptr], %x[stride] \n"
        "bne 1b\n"
        "2: \n"
        "cbz %x[rem_cnt], 3f\n"
        "ld1b {z0.b}, p0/Z, [%x[inptr0]]\n"
        "ld1b {z1.b}, p0/Z, [%x[inptr1]]\n"
        "ld1b {z2.b}, p0/Z, [%x[inptr2]]\n"
        "ld1b {z3.b}, p0/Z, [%x[inptr3]]\n"
        "ld1b {z4.b}, p0/Z, [%x[inptr4]]\n"
        "ld1b {z5.b}, p0/Z, [%x[inptr5]]\n"
        "ld1b {z6.b}, p0/Z, [%x[inptr6]]\n"
        "ld1b {z7.b}, p0/Z, [%x[inptr7]]\n"
        "add %x[inptr0], %x[inptr0], #0x08\n"
        "add %x[inptr1], %x[inptr1], #0x08\n"
        "add %x[inptr2], %x[inptr2], #0x08\n"
        "add %x[inptr3], %x[inptr3], #0x08\n"
        "add %x[inptr4], %x[inptr4], #0x08\n"
        "add %x[inptr5], %x[inptr5], #0x08\n"
        "add %x[inptr6], %x[inptr6], #0x08\n"
        "add %x[inptr7], %x[inptr7], #0x08\n" ZIP8x16_16x8
        // zip-64
        "trn1 z0.d,  z8.d,  z12.d\n"  // 0-1
        "trn1 z2.d,  z9.d,  z13.d\n"  // 4-5
        "trn1 z4.d,  z10.d, z14.d\n"  // 2-3
        "trn1 z6.d,  z11.d, z15.d\n"  // 6-7

        "st1b {z0.b}, p0, [%x[outptr]]\n"
        "st1b {z4.b}, p0, [%x[outptr], #1, MUL VL]\n"
        "st1b {z2.b}, p0, [%x[outptr], #2, MUL VL]\n"
        "st1b {z6.b}, p0, [%x[outptr], #3, MUL VL]\n"
        "add  %x[outptr], %x[outptr], %x[stride] \n"
        "3: \n"
        : [inptr0] "+r"(inptr_row[0]),
          [inptr1] "+r"(inptr_row[1]),
          [inptr2] "+r"(inptr_row[2]),
          [inptr3] "+r"(inptr_row[3]),
          [inptr4] "+r"(inptr_row[4]),
          [inptr5] "+r"(inptr_row[5]),
          [inptr6] "+r"(inptr_row[6]),
          [inptr7] "+r"(inptr_row[7]),
          [outptr] "+r"(outptr),
          [cnt] "+r"(cnt_col)
        : [rem_cnt] "r"(rem_cnt), [stride] "r"(stride)
        : "cc",
          "memory",
          "p0",
          "z0",
          "z1",
          "z2",
          "z3",
          "z4",
          "z5",
          "z6",
          "z7",
          "z8",
          "z9",
          "z10",
          "z11",
          "z12",
          "z13",
          "z14",
          "z15");
    if (rem_rem) {
      for (int j = rem_rem; j > 0; j--) {
        for (int i = 0; i < 8; i++) {
          *outptr++ = *inptr_row[i]++;
        }
      }
      for (int j = 0; j < 8 - rem_rem; j++) {
        for (int i = 0; i < 8; i++) {
          *outptr++ = 0;
        }
      }
    }
  }
  LITE_PARALLEL_COMMON_END();
}

#define LOAD_DATA_SVE(index)                                    \
  auto vdata_##index##_ = svld1(all_true_pg, inptr_row[index]); \
  inptr_row[index] += 8;

#define STORE_DATA_SVE(index, offset) \
  svst1(all_true_pg, outptr + offset, vdata_##index##_);
/**
 * \brief input data is not transpose, (kmax-k0) * (nmax-n0)
 * kup = (K + 7) / 8
 * for sve, transform data to block x kup x 12 x 8 layout
 * din = a00a01..a0k;a10a11..a1k;...:am0am1..amk
 * dout = a00a10..a70;a01a11..a71;...;a0_11a1_11...a7_11;a80a90...a15_0;...
 */
void loadb_k8n12_int8_sve(int8_t* out,
                          const int8_t* in,
                          const int ldin,
                          const int k0,
                          const int kmax,
                          const int n0,
                          const int nmax) {
  int x_len = (nmax - n0);
  int y_len = (kmax - k0);
  int8_t zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(int8_t) * x_len);
  int8_t* dout = out;
  const int8_t* din = in + k0 * ldin + n0;
  int cnt = x_len / 12;
  int right_remain = x_len % 12;

  int rem_cnt = right_remain >> 2;
  int rem_rem = right_remain % 4;
  int rem_2 = rem_rem >> 1;
  int rem_2_rem = rem_rem % 2;
  rem_2 = (rem_rem + 1) >> 1;
  int kup = ROUNDUP_SVE(y_len, 8);
  int cnt_12 = (cnt > 0) ? 12 : 0;
  int cnt_4 = (rem_cnt > 0) ? 4 : 0;
  int cnt_2 = (rem_2 > 0) ? 2 : 0;
  int stride_12 = cnt_12 * kup;
  int stride_4 = cnt_4 * kup;
  int stride_2 = cnt_2 * kup;
  int stride_w_4 = stride_12 * cnt;
  int stride_w_2 = stride_w_4 + stride_4 * rem_cnt;

  LITE_PARALLEL_COMMON_BEGIN(y, tid, y_len, 0, 8) {
    const int8_t* inptr_row[8];
    inptr_row[0] = din + y * ldin;
    for (int i = 1; i < 8; i++) {
      inptr_row[i] = inptr_row[i - 1] + ldin;
    }

    int8_t* outptr_row_col = dout + y * cnt_12;  // 12;
    int8_t* outptr_row_4 = dout + stride_w_4 + y * cnt_4;
    int8_t* outptr_row_1 = dout + stride_w_2 + y * cnt_2;
    if (y + 7 > y_len) {
      switch (y_len - y) {
        case 1:
          inptr_row[1] = zerobuff;
        case 2:
          inptr_row[2] = zerobuff;
        case 3:
          inptr_row[3] = zerobuff;
        case 4:
          inptr_row[4] = zerobuff;
        case 5:
          inptr_row[5] = zerobuff;
        case 6:
          inptr_row[6] = zerobuff;
        case 7:
          inptr_row[7] = zerobuff;
        default:
          break;
      }
    }

    int cnt_col = cnt;
    int rem_cnt_col = rem_cnt;
    asm volatile(
        "ptrue p0.b\n"
        "cbz %x[cnt], 2f\n"
        "1: \n"
        "ld1b {z0.b}, p0/Z, [%x[inptr0]]\n"
        "ld1b {z1.b}, p0/Z, [%x[inptr1]]\n"
        "ld1b {z2.b}, p0/Z, [%x[inptr2]]\n"
        "ld1b {z3.b}, p0/Z, [%x[inptr3]]\n"
        "ld1b {z4.b}, p0/Z, [%x[inptr4]]\n"
        "ld1b {z5.b}, p0/Z, [%x[inptr5]]\n"
        "ld1b {z6.b}, p0/Z, [%x[inptr6]]\n"
        "ld1b {z7.b}, p0/Z, [%x[inptr7]]\n"
        "subs %x[cnt], %x[cnt], #1\n" ZIP8x16_16x8
        // zip-64
        "trn1 z0.d,  z8.d,  z12.d\n"  // 0-1
        "trn2 z1.d,  z8.d,  z12.d\n"  // 8-9
        "trn1 z2.d,  z9.d,  z13.d\n"  // 4-5
        "trn2 z3.d,  z9.d,  z13.d\n"  // 12-13
        "trn1 z4.d,  z10.d, z14.d\n"  // 2-3
        "trn2 z5.d,  z10.d, z14.d\n"  // 10-11
        "trn1 z6.d,  z11.d, z15.d\n"  // 6-7
        "trn2 z7.d,  z11.d, z15.d\n"  // 14-15

        "add %x[inptr0], %x[inptr0], #0x0c\n"
        "add %x[inptr1], %x[inptr1], #0x0c\n"
        "add %x[inptr2], %x[inptr2], #0x0c\n"
        "add %x[inptr3], %x[inptr3], #0x0c\n"
        "add %x[inptr4], %x[inptr4], #0x0c\n"
        "add %x[inptr5], %x[inptr5], #0x0c\n"
        "add %x[inptr6], %x[inptr6], #0x0c\n"
        "add %x[inptr7], %x[inptr7], #0x0c\n"

        "st1b {z0.b}, p0, [%x[outptr_12]]\n"
        "st1b {z4.b}, p0, [%x[outptr_12], #1, MUL VL]\n"
        "st1b {z2.b}, p0, [%x[outptr_12], #2, MUL VL]\n"
        "st1b {z6.b}, p0, [%x[outptr_12], #3, MUL VL]\n"
        "st1b {z1.b}, p0, [%x[outptr_12], #4, MUL VL]\n"
        "st1b {z5.b}, p0, [%x[outptr_12], #5, MUL VL]\n"
        "add %x[outptr_12], %x[outptr_12], %x[stride_12]\n"
        "bne 1b\n"
        "2: \n"
        "cbz %x[rem_cnt], 3f\n"
        "ld1b {z0.b}, p0/Z, [%x[inptr0]]\n"
        "ld1b {z1.b}, p0/Z, [%x[inptr1]]\n"
        "ld1b {z2.b}, p0/Z, [%x[inptr2]]\n"
        "ld1b {z3.b}, p0/Z, [%x[inptr3]]\n"
        "ld1b {z4.b}, p0/Z, [%x[inptr4]]\n"
        "ld1b {z5.b}, p0/Z, [%x[inptr5]]\n"
        "ld1b {z6.b}, p0/Z, [%x[inptr6]]\n"
        "ld1b {z7.b}, p0/Z, [%x[inptr7]]\n"
        "subs %x[rem_cnt], %x[rem_cnt], #1\n" ZIP8x16_16x8
        // zip-64
        "trn1 z0.d,  z8.d,  z12.d\n"  // 0-1
        "trn1 z4.d,  z10.d, z14.d\n"  // 2-3
        "add %x[inptr0], %x[inptr0], #0x04\n"
        "add %x[inptr1], %x[inptr1], #0x04\n"
        "add %x[inptr2], %x[inptr2], #0x04\n"
        "add %x[inptr3], %x[inptr3], #0x04\n"
        "add %x[inptr4], %x[inptr4], #0x04\n"
        "add %x[inptr5], %x[inptr5], #0x04\n"
        "add %x[inptr6], %x[inptr6], #0x04\n"
        "add %x[inptr7], %x[inptr7], #0x04\n"
        "st1b {z0.b}, p0, [%x[outptr_4]]\n"
        "st1b {z4.b}, p0, [%x[outptr_4], #1, MUL VL]\n"
        "add %x[outptr_4], %x[outptr_4], %x[stride_4]\n"
        "bne 2b\n"
        "3: \n"
        : [inptr0] "+r"(inptr_row[0]),
          [inptr1] "+r"(inptr_row[1]),
          [inptr2] "+r"(inptr_row[2]),
          [inptr3] "+r"(inptr_row[3]),
          [inptr4] "+r"(inptr_row[4]),
          [inptr5] "+r"(inptr_row[5]),
          [inptr6] "+r"(inptr_row[6]),
          [inptr7] "+r"(inptr_row[7]),
          [outptr_12] "+r"(outptr_row_col),
          [outptr_4] "+r"(outptr_row_4),
          [cnt] "+r"(cnt_col),
          [rem_cnt] "+r"(rem_cnt_col)
        : [stride_12] "r"(stride_12), [stride_4] "r"(stride_4)
        : "cc",
          "memory",
          "p0",
          "z0",
          "z1",
          "z2",
          "z3",
          "z4",
          "z5",
          "z6",
          "z7",
          "z8",
          "z9",
          "z10",
          "z11",
          "z12",
          "z13",
          "z14",
          "z15");
    if (rem_rem > 0) {
      for (int i = 0; i < rem_2; i++) {
        outptr_row_1[0] = *inptr_row[0]++;
        outptr_row_1[1] = *inptr_row[1]++;
        outptr_row_1[2] = *inptr_row[2]++;
        outptr_row_1[3] = *inptr_row[3]++;
        outptr_row_1[4] = *inptr_row[4]++;
        outptr_row_1[5] = *inptr_row[5]++;
        outptr_row_1[6] = *inptr_row[6]++;
        outptr_row_1[7] = *inptr_row[7]++;
        if (i == rem_2 - 1 && rem_2_rem) {
          outptr_row_1[8] = 0;
          outptr_row_1[9] = 0;
          outptr_row_1[10] = 0;
          outptr_row_1[11] = 0;
          outptr_row_1[12] = 0;
          outptr_row_1[13] = 0;
          outptr_row_1[14] = 0;
          outptr_row_1[15] = 0;
        } else {
          outptr_row_1[8] = *inptr_row[0]++;
          outptr_row_1[9] = *inptr_row[1]++;
          outptr_row_1[10] = *inptr_row[2]++;
          outptr_row_1[11] = *inptr_row[3]++;
          outptr_row_1[12] = *inptr_row[4]++;
          outptr_row_1[13] = *inptr_row[5]++;
          outptr_row_1[14] = *inptr_row[6]++;
          outptr_row_1[15] = *inptr_row[7]++;
        }
        outptr_row_1 += stride_2;
      }
    }
  }
  LITE_PARALLEL_COMMON_END();
}

void loadb_k8n12_trans_int8_sve(int8_t* out,
                                const int8_t* in,
                                const int ldin,
                                const int k0,
                                const int kmax,
                                const int n0,
                                const int nmax) {
  int x_len = kmax - k0;
  int y_len = nmax - n0;
  int size = ROUNDUP_SVE(x_len, 8);
  int8_t* dout = out;
  const int8_t* din = in + n0 * ldin + k0;
  int8_t zerobuff[size];  // NOLINT
  memset(zerobuff, 0, sizeof(int8_t) * size);
  int cnt = x_len >> 3;
  int remain = x_len & 7;
  const auto all_true_pg = svptrue<int8_t>();
  int cnt_y = y_len / 12;
  int rem_y = y_len % 12;
  int cnt_4 = rem_y >> 2;
  int cnt_1 = rem_y % 4;

  LITE_PARALLEL_BEGIN(y, tid, cnt_y) {
    const int8_t* inptr_row[12];
    int8_t* outptr = dout + y * 12 * size;
    inptr_row[0] = din + y * 12 * ldin;
    for (int i = 1; i < 12; i++) {
      inptr_row[i] = inptr_row[i - 1] + ldin;
    }
    for (int i = 0; i < cnt; i++) {
      LOAD_DATA_SVE(0)
      LOAD_DATA_SVE(1)
      LOAD_DATA_SVE(2)
      LOAD_DATA_SVE(3)
      LOAD_DATA_SVE(4)
      LOAD_DATA_SVE(5)
      LOAD_DATA_SVE(6)
      LOAD_DATA_SVE(7)
      LOAD_DATA_SVE(8)
      LOAD_DATA_SVE(9)
      LOAD_DATA_SVE(10)
      LOAD_DATA_SVE(11)
      STORE_DATA_SVE(0, 0)
      STORE_DATA_SVE(1, 8)
      STORE_DATA_SVE(2, 16)
      STORE_DATA_SVE(3, 24)
      STORE_DATA_SVE(4, 32)
      STORE_DATA_SVE(5, 40)
      STORE_DATA_SVE(6, 48)
      STORE_DATA_SVE(7, 56)
      STORE_DATA_SVE(8, 64)
      STORE_DATA_SVE(9, 72)
      STORE_DATA_SVE(10, 80)
      STORE_DATA_SVE(11, 88)
      outptr += 96;
    }
    if (remain > 0) {
      for (int i = 0; i < 12; i++) {
        for (int j = 0; j < remain; j++) {
          *outptr++ = *inptr_row[i]++;
        }
        for (int j = 0; j < 8 - remain; j++) {
          *outptr++ = 0;
        }
      }
    }
  }
  LITE_PARALLEL_END();
  const int8_t* input_4 = din + cnt_y * 12 * ldin;
  int8_t* output_4 = dout + cnt_y * 12 * size;
  LITE_PARALLEL_BEGIN(y, tid, cnt_4) {
    const int8_t* inptr_row[4];
    int8_t* outptr = output_4 + y * 4 * size;
    inptr_row[0] = input_4 + y * 4 * ldin;
    for (int i = 1; i < 4; i++) {
      inptr_row[i] = inptr_row[i - 1] + ldin;
    }
    for (int i = 0; i < cnt; i++) {
      LOAD_DATA_SVE(0)
      LOAD_DATA_SVE(1)
      LOAD_DATA_SVE(2)
      LOAD_DATA_SVE(3)
      STORE_DATA_SVE(0, 0)
      STORE_DATA_SVE(1, 8)
      STORE_DATA_SVE(2, 16)
      STORE_DATA_SVE(3, 24)
      outptr += 32;
    }
    if (remain > 0) {
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < remain; j++) {
          *outptr++ = *inptr_row[i]++;
        }
        for (int j = 0; j < 8 - remain; j++) {
          *outptr++ = 0;
        }
      }
    }
  }
  LITE_PARALLEL_END();
  const int8_t* input_1 = input_4 + cnt_4 * 4 * ldin;
  int8_t* output_1 = output_4 + cnt_4 * 4 * size;
  for (int y = 0; y < cnt_1; y += 2) {
    int8_t* outptr = output_1 + y * size;
    const int8_t* inptr_row0 = input_1 + y * ldin;
    const int8_t* inptr_row1 = inptr_row0 + ldin;
    if (y + 1 == cnt_1) {
      for (int i = 0; i < cnt; i++) {
        memcpy(outptr, inptr_row0, sizeof(int8_t) * 8);
        memset(outptr + 8, 0, sizeof(int8_t) * 8);
        inptr_row0 += 8;
        outptr += 16;
      }
      memcpy(outptr, inptr_row0, sizeof(int8_t) * remain);
      memset(outptr + remain, 0, sizeof(int8_t) * (16 - remain));
    } else {
      for (int i = 0; i < cnt; i++) {
        memcpy(outptr, inptr_row0, sizeof(int8_t) * 8);
        memcpy(outptr + 8, inptr_row1, sizeof(int8_t) * 8);
        inptr_row0 += 8;
        inptr_row1 += 8;
        outptr += 16;
      }
      memcpy(outptr, inptr_row0, sizeof(int8_t) * remain);
      memset(outptr + remain, 0, (8 - remain) * sizeof(int8_t));
      memcpy(outptr + 8, inptr_row1, sizeof(int8_t) * remain);
      memset(outptr + 8 + remain, 0, (8 - remain) * sizeof(int8_t));
    }
  }
}

void prepackA_int8_sve(int8_t* out,
                       const int8_t* in,
                       int ldin,
                       int m0,
                       int mmax,
                       int k0,
                       int kmax,
                       bool is_trans,
                       ARMContext* ctx) {
  if (is_trans) {
    prepackA_m8k8_trans_int8_sve(out, in, ldin, m0, mmax, k0, kmax);
  } else {
    prepackA_m8k8_int8_sve(out, in, ldin, m0, mmax, k0, kmax);
  }
}

void prepackA_int8_sve(TensorLite* tout,
                       const TensorLite& tin,
                       int m,
                       int k,
                       int group,
                       bool is_trans,
                       ARMContext* ctx) {
  int hblock = get_hblock_int8_sve(ctx);
  int m_roundup = ROUNDUP_SVE(m, hblock);
  // round up to 128 bits
  int kup = ROUNDUP_SVE(k, 8);
  int group_size_round_up = ((m_roundup * kup + 15) / 16) * 16;

  if (tout->numel() < group_size_round_up * group) {
    tout->Resize({1, 1, 1, group_size_round_up * group});
  }
  int lda = k;
  if (is_trans) {
    lda = m;
  }
  for (int g = 0; g < group; ++g) {
    const int8_t* weights_group = tin.data<int8_t>() + g * m * k;
    int8_t* weights_trans_ptr =
        tout->mutable_data<int8_t>() + g * group_size_round_up;
    prepackA_int8_sve(
        weights_trans_ptr, weights_group, lda, 0, m, 0, k, is_trans, ctx);
  }
}

#define SMMLA_PARAMS(Dtype)                                            \
  const int8_t *a_ptr, const int8_t *&b_ptr, const float *bias,        \
      Dtype *&c_ptr0, Dtype *&c_ptr1, Dtype *&c_ptr2, Dtype *&c_ptr3,  \
      Dtype *&c_ptr4, Dtype *&c_ptr5, Dtype *&c_ptr6, Dtype *&c_ptr7,  \
      const float32_t *scale, const float32_t *alpha, int k, int tail, \
      int flag_act, int last
template <typename Dtype>
inline void gemm_smmla_int8_kernel_8x1(SMMLA_PARAMS(Dtype));

template <typename Dtype>
inline void gemm_smmla_int8_kernel_8x4(SMMLA_PARAMS(Dtype));

template <typename Dtype>
inline void gemm_smmla_int8_kernel_8x12(SMMLA_PARAMS(Dtype));

// clang-format off
#define INIT_SMMLA_8x2                              \
  "ptrue p0.b \n"                                   \
  "prfm   pldl1keep, [%[a_ptr]]\n"                  \
  "prfm   pldl1keep, [%[b_ptr]]\n"                  \
  "mov    z8.s,   #0x0\n"                           \
  "mov    z14.s,  #0x0\n"                           \
  "mov    z20.s,  #0x0\n"                           \
  "mov    z26.s,  #0x0\n"                           \
  "ld1rqb {z0.b}, p0/Z, [%x[a_ptr]]\n"              \
  "prfm   pldl1keep, [%[a_ptr], #64]\n"             \
  "prfm   pldl1keep, [%[b_ptr], #64]\n"             \
  "ld1b   {z4.b}, p0/Z, [%x[b_ptr]]\n"              \
  "ld1rqb {z1.b}, p0/Z, [%x[a_ptr], #0x10]\n"       \
  "ld1b   {z5.b}, p0/Z, [%x[b_ptr],  #1, MUL VL]\n" \
  "prfm   pldl1keep, [%[a_ptr], #128]\n"            \
  "prfm   pldl1keep, [%[b_ptr], #128]\n"            \
  "ld1rqb {z2.b}, p0/Z, [%x[a_ptr], #0x20]\n"       \
  "prfm   pldl1keep, [%[a_ptr], #192]\n"            \
  "prfm   pldl1keep, [%[b_ptr], #192]\n"            \
  "prfm   pldl1keep, [%[a_ptr], #256]\n"            \
  "cbz    %x[k],  1f\n"

#define COMPUTE_SMMLA_8x2                           \
  "0: \n"                                           \
  "ld1rqb {z3.b}, p0/Z, [%x[a_ptr], #0x30]\n"       \
  "ld1rqb {z6.b}, p0/Z, [%x[a_ptr], #0x40]\n"       \
  "smmla  z8.s,  z0.b, z4.b\n"                      \
  "smmla  z14.s, z1.b, z4.b\n"                      \
  "ld1rqb {z7.b}, p0/Z, [%x[a_ptr], #0x50]\n"       \
  "ld1rqb {z9.b}, p0/Z, [%x[a_ptr], #0x60]\n"       \
  "smmla  z20.s, z2.b, z4.b\n"                      \
  "smmla  z26.s, z3.b, z4.b\n"                      \
  "ld1rqb {z10.b}, p0/Z, [%x[a_ptr], #0x70]\n"      \
  "addvl  %x[b_ptr], %x[b_ptr], #2\n"               \
  "add    %x[a_ptr], %x[a_ptr], #0x80\n"            \
  "subs   %x[k], %x[k], #1\n"                       \
  "ld1b   {z4.b}, p0/Z, [%x[b_ptr]]\n"              \
  "ld1rqb {z0.b}, p0/Z, [%x[a_ptr]]\n"              \
  "smmla  z8.s,  z6.b,  z5.b\n"                     \
  "smmla  z14.s, z7.b,  z5.b\n"                     \
  "ld1rqb {z1.b}, p0/Z, [%x[a_ptr], #0x10]\n"       \
  "smmla  z20.s, z9.b,  z5.b\n"                     \
  "smmla  z26.s, z10.b, z5.b\n"                     \
  "ld1rqb {z2.b}, p0/Z, [%x[a_ptr], #0x20]\n"       \
  "ld1b   {z5.b}, p0/Z, [%x[b_ptr],  #1, MUL VL]\n" \
  "bne 0b\n"

#define COMPUTE_SMMLA_8x2_REMAIN               \
  "1: \n"                                      \
  "cmp %x[rem_cnt], #2\n"                      \
  "beq 2f\n"                                   \
  "ld1rqb {z3.b}, p0/Z, [%x[a_ptr], #0x30]\n"  \
  "addvl  %x[b_ptr], %x[b_ptr], #1\n"          \
  "add    %x[a_ptr], %x[a_ptr], #0x40\n"       \
  "smmla  z8.s,  z0.b, z4.b\n"                 \
  "smmla  z14.s, z1.b, z4.b\n"                 \
  "smmla  z20.s, z2.b, z4.b\n"                 \
  "smmla  z26.s, z3.b, z4.b\n"                 \
  "b 3f\n"                                     \
  "2: \n"                                      \
  "ld1rqb {z3.b}, p0/Z, [%x[a_ptr], #0x30]\n"  \
  "ld1rqb {z6.b}, p0/Z, [%x[a_ptr], #0x40]\n"  \
  "smmla  z8.s,  z0.b, z4.b\n"                 \
  "smmla  z14.s, z1.b, z4.b\n"                 \
  "ld1rqb {z7.b}, p0/Z, [%x[a_ptr], #0x50]\n"  \
  "ld1rqb {z9.b}, p0/Z, [%x[a_ptr], #0x60]\n"  \
  "smmla  z20.s, z2.b, z4.b\n"                 \
  "smmla  z26.s, z3.b, z4.b\n"                 \
  "ld1rqb {z10.b}, p0/Z, [%x[a_ptr], #0x70]\n" \
  "addvl  %x[b_ptr], %x[b_ptr], #2\n"          \
  "add    %x[a_ptr], %x[a_ptr], #0x80\n"       \
  "smmla  z8.s,  z6.b,  z5.b\n"                \
  "smmla  z14.s, z7.b,  z5.b\n"                \
  "smmla  z20.s, z9.b,  z5.b\n"                \
  "smmla  z26.s, z10.b, z5.b\n"

#define CVT_SMMLA_INT32_TO_FP32_8x2           \
  "3: \n"                                     \
  "mov z6.s, #0x0\n"                          \
  "ld1rqw {z0.s}, p0/Z, [%x[bias]]\n"         \
  "ld1rqw {z1.s}, p0/Z, [%x[bias], #0x10]\n"  \
  "trn1 z2.d,  z8.d,  z6.d\n"                 \
  "trn2 z3.d,  z8.d,  z6.d\n"                 \
  "trn1 z8.d,  z14.d, z6.d\n"                 \
  "trn2 z9.d,  z14.d, z6.d\n"                 \
  "trn1 z14.d, z20.d, z6.d\n"                 \
  "trn2 z15.d, z20.d, z6.d\n"                 \
  "trn1 z20.d, z26.d, z6.d\n"                 \
  "trn2 z21.d, z26.d, z6.d\n"                 \
  "ld1rqw {z4.s}, p0/Z, [%x[scale]]\n"        \
  "dup    z30.s,  z0.s[0]\n"                  \
  "dup    z29.s,  z0.s[1]\n"                  \
  "ld1rqw {z5.s}, p0/Z, [%x[scale], #0x10]\n" \
  "scvtf  z2.s,  p0/m, z2.s\n"                \
  "scvtf  z3.s,  p0/m, z3.s\n"                \
  "dup    z28.s,  z0.s[2]\n"                  \
  "dup    z27.s,  z0.s[3]\n"                  \
  "scvtf  z8.s,  p0/m, z8.s\n"                \
  "scvtf  z9.s,  p0/m, z9.s\n"                \
  "dup    z26.s,  z1.s[0]\n"                  \
  "dup    z25.s,  z1.s[1]\n"                  \
  "scvtf  z14.s, p0/m, z14.s\n"               \
  "scvtf  z15.s, p0/m, z15.s\n"               \
  "dup    z24.s,  z1.s[2]\n"                  \
  "dup    z23.s,  z1.s[3]\n"                  \
  "scvtf  z20.s, p0/m, z20.s\n"               \
  "scvtf  z21.s, p0/m, z21.s\n"               \
  "fmla   z30.s, z2.s, z4.s[0]\n"             \
  "fmla   z29.s, z3.s, z4.s[1]\n"             \
  "fmla   z28.s, z8.s, z4.s[2]\n"             \
  "fmla   z27.s, z9.s, z4.s[3]\n"             \
  "fmla   z26.s, z14.s, z5.s[0]\n"            \
  "fmla   z25.s, z15.s, z5.s[1]\n"            \
  "fmla   z24.s, z20.s, z5.s[2]\n"            \
  "fmla   z23.s, z21.s, z5.s[3]\n"

#define SMMLA_STORE_FP32_8x2            \
  "cbz %x[last], 11f\n"                 \
  "mov x0, #0x08\n"                     \
  "b  12f\n"                            \
  "11:  \n"                             \
  "mov x0, #0x0c\n"                     \
  "12:  \n"                             \
  "mov x1, #0x10\n"                     \
  "whilelt p1.b, x0, x1\n"              \
  "st1w {z30.s},  p1, [%x[c_ptr0]]\n"   \
  "st1w {z29.s},  p1, [%x[c_ptr1]]\n"   \
  "st1w {z28.s},  p1, [%x[c_ptr2]]\n"   \
  "st1w {z27.s},  p1, [%x[c_ptr3]]\n"   \
  "st1w {z26.s},  p1, [%x[c_ptr4]]\n"   \
  "st1w {z25.s},  p1, [%x[c_ptr5]]\n"   \
  "st1w {z24.s},  p1, [%x[c_ptr6]]\n"   \
  "st1w {z23.s},  p1, [%x[c_ptr7]]\n"   \
  "add %x[c_ptr0], %x[c_ptr0], #0x08\n" \
  "add %x[c_ptr1], %x[c_ptr1], #0x08\n" \
  "add %x[c_ptr2], %x[c_ptr2], #0x08\n" \
  "add %x[c_ptr3], %x[c_ptr3], #0x08\n" \
  "add %x[c_ptr4], %x[c_ptr4], #0x08\n" \
  "add %x[c_ptr5], %x[c_ptr5], #0x08\n" \
  "add %x[c_ptr6], %x[c_ptr6], #0x08\n" \
  "add %x[c_ptr7], %x[c_ptr7], #0x08\n"

#define INIT_SMMLA_8x4                              \
  "ptrue p0.b \n"                                   \
  "prfm   pldl1keep, [%[a_ptr]]\n"                  \
  "prfm   pldl1keep, [%[b_ptr]]\n"                  \
  "mov    z8.s,   #0x0\n"                           \
  "mov    z14.s,  #0x0\n"                           \
  "mov    z20.s,  #0x0\n"                           \
  "mov    z26.s,  #0x0\n"                           \
  "ld1rqb {z0.b}, p0/Z, [%x[a_ptr]]\n"              \
  "prfm   pldl1keep, [%[a_ptr], #64]\n"             \
  "prfm   pldl1keep, [%[b_ptr], #64]\n"             \
  "ld1b   {z4.b}, p0/Z, [%x[b_ptr]]\n"              \
  "mov    z9.s,   #0x0\n"                           \
  "mov    z15.s,  #0x0\n"                           \
  "mov    z21.s,  #0x0\n"                           \
  "mov    z27.s,  #0x0\n"                           \
  "ld1rqb {z1.b}, p0/Z, [%x[a_ptr], #0x10]\n"       \
  "ld1b   {z5.b}, p0/Z, [%x[b_ptr],  #1, MUL VL]\n" \
  "prfm   pldl1keep, [%[a_ptr], #128]\n"            \
  "prfm   pldl1keep, [%[b_ptr], #128]\n"            \
  "ld1rqb {z2.b}, p0/Z, [%x[a_ptr], #0x20]\n"       \
  "prfm   pldl1keep, [%[a_ptr], #192]\n"            \
  "prfm   pldl1keep, [%[b_ptr], #192]\n"            \
  "prfm   pldl1keep, [%[a_ptr], #256]\n"            \
  "cbz    %x[k],  1f\n"

#define COMPUTE_SMMLA_8x4_0                         \
  "0: \n"                                           \
  "ld1rqb {z3.b}, p0/Z, [%x[a_ptr], #0x30]\n"       \
  "ld1b   {z6.b}, p0/Z, [%x[b_ptr],  #2, MUL VL]\n" \
  "smmla  z8.s,  z0.b, z4.b\n"                      \
  "smmla  z14.s, z1.b, z4.b\n"                      \
  "ld1b   {z7.b}, p0/Z, [%x[b_ptr],  #3, MUL VL]\n" \
  "smmla  z20.s, z2.b, z4.b\n"                      \
  "smmla  z26.s, z3.b, z4.b\n"                      \
  "addvl %x[b_ptr], %x[b_ptr], #4\n"                \
  "smmla  z9.s,  z0.b, z5.b\n"                      \
  "smmla  z15.s, z1.b, z5.b\n"                      \
  "smmla  z21.s, z2.b, z5.b\n"                      \
  "smmla  z27.s, z3.b, z5.b\n"                      \
  "ld1rqb {z0.b}, p0/Z, [%x[a_ptr], #0x40]\n"       \
  "ld1rqb {z1.b}, p0/Z, [%x[a_ptr], #0x50]\n"       \
  "ld1rqb {z2.b}, p0/Z, [%x[a_ptr], #0x60]\n"       \
  "ld1rqb {z3.b}, p0/Z, [%x[a_ptr], #0x70]\n"

#define COMPUTE_SMMLA_8x4_1                         \
  "ld1b   {z4.b}, p0/Z, [%x[b_ptr],  #0, MUL VL]\n" \
  "smmla  z8.s,  z0.b, z6.b\n"                      \
  "smmla  z14.s, z1.b, z6.b\n"                      \
  "smmla  z20.s, z2.b, z6.b\n"                      \
  "smmla  z26.s, z3.b, z6.b\n"                      \
  "ld1b   {z5.b}, p0/Z, [%x[b_ptr],  #1, MUL VL]\n" \
  "smmla  z9.s,  z0.b, z7.b\n"                      \
  "smmla  z15.s, z1.b, z7.b\n"                      \
  "smmla  z21.s, z2.b, z7.b\n"                      \
  "smmla  z27.s, z3.b, z7.b\n"                      \
  "add    %x[a_ptr], %x[a_ptr], #0x80\n"            \
  "subs   %x[k], %x[k], #1\n"                       \
  "ld1rqb {z0.b}, p0/Z, [%x[a_ptr]]\n"              \
  "ld1rqb {z1.b}, p0/Z, [%x[a_ptr], #0x10]\n"       \
  "ld1rqb {z2.b}, p0/Z, [%x[a_ptr], #0x20]\n"       \
  "bne 0b\n"

#define COMPUTE_SMMLA_8x4_REMAIN              \
  "1: \n"                                     \
  "cmp %x[rem_cnt], #2\n"                     \
  "beq 2f\n"                                  \
  "ld1rqb {z3.b}, p0/Z, [%x[a_ptr], #0x30]\n" \
  "smmla  z8.s,  z0.b, z4.b\n"                \
  "smmla  z14.s, z1.b, z4.b\n"                \
  "smmla  z20.s, z2.b, z4.b\n"                \
  "smmla  z26.s, z3.b, z4.b\n"                \
  "addvl %x[b_ptr], %x[b_ptr], #2\n"          \
  "smmla  z9.s,  z0.b, z5.b\n"                \
  "smmla  z15.s, z1.b, z5.b\n"                \
  "smmla  z21.s, z2.b, z5.b\n"                \
  "smmla  z27.s, z3.b, z5.b\n"                \
  "add    %x[a_ptr], %x[a_ptr], #0x40\n"      \
  "b 3f\n"                                    \
  "2: \n" COMPUTE_SMMLA_8x4_0                 \
  "smmla  z8.s,  z0.b, z6.b\n"                \
  "smmla  z14.s, z1.b, z6.b\n"                \
  "smmla  z20.s, z2.b, z6.b\n"                \
  "smmla  z26.s, z3.b, z6.b\n"                \
  "smmla  z9.s,  z0.b, z7.b\n"                \
  "smmla  z15.s, z1.b, z7.b\n"                \
  "smmla  z21.s, z2.b, z7.b\n"                \
  "smmla  z27.s, z3.b, z7.b\n"                \
  "add    %x[a_ptr], %x[a_ptr], #0x80\n"

#define CVT_SMMLA_INT32_TO_FP32_8x4           \
  "3: \n"                                     \
  "ld1rqw {z0.s}, p0/Z, [%x[bias]]\n"         \
  "ld1rqw {z1.s}, p0/Z, [%x[bias], #0x10]\n"  \
  "trn1 z2.d,  z8.d,  z9.d\n"                 \
  "trn2 z3.d,  z8.d,  z9.d\n"                 \
  "trn1 z8.d,  z14.d, z15.d\n"                \
  "trn2 z9.d,  z14.d, z15.d\n"                \
  "trn1 z14.d, z20.d, z21.d\n"                \
  "trn2 z15.d, z20.d, z21.d\n"                \
  "trn1 z20.d, z26.d, z27.d\n"                \
  "trn2 z21.d, z26.d, z27.d\n"                \
  "ld1rqw {z4.s}, p0/Z, [%x[scale]]\n"        \
  "dup    z30.s,  z0.s[0]\n"                  \
  "dup    z29.s,  z0.s[1]\n"                  \
  "ld1rqw {z5.s}, p0/Z, [%x[scale], #0x10]\n" \
  "scvtf  z2.s,  p0/m, z2.s\n"                \
  "scvtf  z3.s,  p0/m, z3.s\n"                \
  "dup    z28.s,  z0.s[2]\n"                  \
  "dup    z27.s,  z0.s[3]\n"                  \
  "scvtf  z8.s,  p0/m, z8.s\n"                \
  "scvtf  z9.s,  p0/m, z9.s\n"                \
  "dup    z26.s,  z1.s[0]\n"                  \
  "dup    z25.s,  z1.s[1]\n"                  \
  "scvtf  z14.s, p0/m, z14.s\n"               \
  "scvtf  z15.s, p0/m, z15.s\n"               \
  "dup    z24.s,  z1.s[2]\n"                  \
  "dup    z23.s,  z1.s[3]\n"                  \
  "scvtf  z20.s, p0/m, z20.s\n"               \
  "scvtf  z21.s, p0/m, z21.s\n"               \
  "fmla   z30.s, z2.s, z4.s[0]\n"             \
  "fmla   z29.s, z3.s, z4.s[1]\n"             \
  "fmla   z28.s, z8.s, z4.s[2]\n"             \
  "fmla   z27.s, z9.s, z4.s[3]\n"             \
  "fmla   z26.s, z14.s, z5.s[0]\n"            \
  "fmla   z25.s, z15.s, z5.s[1]\n"            \
  "fmla   z24.s, z20.s, z5.s[2]\n"            \
  "fmla   z23.s, z21.s, z5.s[3]\n"

#define SMMLA_RELU_8x4(inst) \
  #inst " z30.s,  p0/m, z30.s, z0.s\n" \
  #inst " z29.s,  p0/m, z29.s, z0.s\n" \
  #inst " z28.s,  p0/m, z28.s, z0.s\n" \
  #inst " z27.s,  p0/m, z27.s, z0.s\n" \
  #inst " z26.s,  p0/m, z26.s, z0.s\n" \
  #inst " z25.s,  p0/m, z25.s, z0.s\n" \
  #inst " z24.s,  p0/m, z24.s, z0.s\n" \
  #inst " z23.s,  p0/m, z23.s, z0.s\n"

#define SMMLA_LEAKYRELU_8x4                           \
  "mov z0.s, #0x0    \n"                              \
  "ld1rqw {z1.s}, p0/Z, [%x[alpha]]\n"                \
  "movprfx z4, z30\n  fmin z4.s,  p0/m, z4.s, z0.s\n" \
  "movprfx z5, z29\n  fmin z5.s,  p0/m, z5.s, z0.s\n" \
  "movprfx z6, z28\n  fmin z6.s,  p0/m, z6.s, z0.s\n" \
  "movprfx z7, z27\n  fmin z7.s,  p0/m, z7.s, z0.s\n" \
  "fmax z30.s, p0/m, z30.s,  z0.s\n"                  \
  "fmax z29.s, p0/m, z29.s,  z0.s\n"                  \
  "fmax z28.s, p0/m, z28.s,  z0.s\n"                  \
  "fmax z27.s, p0/m, z27.s,  z0.s\n"                  \
  "fmul z4.s,  p0/m, z4.s,   z1.s\n"                  \
  "fmul z5.s,  p0/m, z5.s,   z1.s\n"                  \
  "fmul z6.s,  p0/m, z6.s,   z1.s\n"                  \
  "fmul z7.s,  p0/m, z7.s,   z1.s\n"                  \
  "fadd z30.s, p0/m, z30.s,  z4.s\n"                  \
  "fadd z29.s, p0/m, z29.s,  z4.s\n"                  \
  "fadd z28.s, p0/m, z28.s,  z6.s\n"                  \
  "fadd z27.s, p0/m, z27.s,  z7.s\n"                  \
  "movprfx z4, z26\n  fmin z4.s,  p0/m, z4.s, z0.s\n" \
  "movprfx z5, z25\n  fmin z5.s,  p0/m, z5.s, z0.s\n" \
  "movprfx z6, z24\n  fmin z6.s,  p0/m, z6.s, z0.s\n" \
  "movprfx z7, z23\n  fmin z7.s,  p0/m, z7.s, z0.s\n" \
  "fmax z26.s, p0/m, z26.s,  z0.s\n"                  \
  "fmax z25.s, p0/m, z25.s,  z0.s\n"                  \
  "fmax z24.s, p0/m, z24.s,  z0.s\n"                  \
  "fmax z23.s, p0/m, z23.s,  z0.s\n"                  \
  "fmul z4.s,  p0/m, z4.s,   z1.s\n"                  \
  "fmul z5.s,  p0/m, z5.s,   z1.s\n"                  \
  "fmul z6.s,  p0/m, z6.s,   z1.s\n"                  \
  "fmul z7.s,  p0/m, z7.s,   z1.s\n"                  \
  "fadd z26.s, p0/m, z26.s,  z4.s\n"                  \
  "fadd z25.s, p0/m, z25.s,  z4.s\n"                  \
  "fadd z24.s, p0/m, z24.s,  z6.s\n"                  \
  "fadd z23.s, p0/m, z23.s,  z7.s\n"

#define SMMLA_HARDSWIDH_8x4                              \
  "mov z0.s, #0x0    \n"                                 \
  "ld1rqw {z1.s}, p0/Z, [%x[alpha]]\n"                   \
  "ld1rqw {z2.s}, p0/Z, [%x[alpha], #0x10]\n"            \
  "ld1rqw {z3.s}, p0/Z, [%x[alpha], #0x20]\n"            \
  "movprfx z4, z30\n  fadd z4.s,   p0/m,  z4.s,  z1.s\n" \
  "movprfx z5, z29\n  fadd z5.s,   p0/m,  z5.s,  z1.s\n" \
  "movprfx z6, z28\n  fadd z6.s,   p0/m,  z6.s,  z1.s\n" \
  "movprfx z7, z27\n  fadd z7.s,   p0/m,  z7.s,  z1.s\n" \
  "fmul z30.s,  p0/m, z30.s,   z2.s\n"                   \
  "fmul z29.s,  p0/m, z29.s,   z2.s\n"                   \
  "fmul z28.s,  p0/m, z28.s,   z2.s\n"                   \
  "fmul z27.s,  p0/m, z27.s,   z2.s\n"                   \
  "fmax z4.s,   p0/m, z4.s,    z0.s\n"                   \
  "fmax z5.s,   p0/m, z5.s,    z0.s\n"                   \
  "fmax z6.s,   p0/m, z6.s,    z0.s\n"                   \
  "fmax z7.s,   p0/m, z7.s,    z0.s\n"                   \
  "fmin z4.s,   p0/m, z4.s,    z3.s\n"                   \
  "fmin z5.s,   p0/m, z5.s,    z3.s\n"                   \
  "fmin z6.s,   p0/m, z6.s,    z3.s\n"                   \
  "fmin z7.s,   p0/m, z7.s,    z3.s\n"                   \
  "fmul z30.s,  p0/m, z30.s,   z4.s\n"                   \
  "fmul z29.s,  p0/m, z29.s,   z5.s\n"                   \
  "fmul z28.s,  p0/m, z28.s,   z6.s\n"                   \
  "fmul z27.s,  p0/m, z27.s,   z7.s\n"                   \
  "movprfx z4, z26\n  fadd z4.s,   p0/m,  z4.s,  z1.s\n" \
  "movprfx z5, z25\n  fadd z5.s,   p0/m,  z5.s,  z1.s\n" \
  "movprfx z6, z24\n  fadd z6.s,   p0/m,  z6.s,  z1.s\n" \
  "movprfx z7, z23\n  fadd z7.s,   p0/m,  z7.s,  z1.s\n" \
  "fmul z26.s,  p0/m, z26.s,   z2.s\n"                   \
  "fmul z25.s,  p0/m, z25.s,   z2.s\n"                   \
  "fmul z24.s,  p0/m, z24.s,   z2.s\n"                   \
  "fmul z23.s,  p0/m, z23.s,   z2.s\n"                   \
  "fmax z4.s,   p0/m, z4.s,    z0.s\n"                   \
  "fmax z5.s,   p0/m, z5.s,    z0.s\n"                   \
  "fmax z6.s,   p0/m, z6.s,    z0.s\n"                   \
  "fmax z7.s,   p0/m, z7.s,    z0.s\n"                   \
  "fmin z4.s,   p0/m, z4.s,    z3.s\n"                   \
  "fmin z5.s,   p0/m, z5.s,    z3.s\n"                   \
  "fmin z6.s,   p0/m, z6.s,    z3.s\n"                   \
  "fmin z7.s,   p0/m, z7.s,    z3.s\n"                   \
  "fmul z26.s,  p0/m, z26.s,   z4.s\n"                   \
  "fmul z25.s,  p0/m, z25.s,   z5.s\n"                   \
  "fmul z24.s,  p0/m, z24.s,   z6.s\n"                   \
  "fmul z23.s,  p0/m, z23.s,   z7.s\n"

#define SMMLA_ACT_PROCESS_8x4 \
  "cmp %x[flag_act], #1\n"     \
  "beq 3f\n"     \
  "cmp %x[flag_act], #0\n"     \
  "beq 10f\n"     \
  "cmp %x[flag_act], #2\n"     \
  "beq 4f\n"     \
  "cmp %x[flag_act], #3\n"     \
  "beq 5f\n"     \
  SMMLA_HARDSWIDH_8x4 \
  "3: \n"        \
  "mov z0.s, #0x0    \n" \
  SMMLA_RELU_8x4(fmax) \
  "b 10f\n"      \
  "4: \n"        \
  "mov z0.s, #0x0    \n" \
  SMMLA_RELU_8x4(fmax) \
  "ld1rqw {z0.s}, p0/Z, [%x[alpha]]\n"                   \
  SMMLA_RELU_8x4(fmin) \
  "b 10f\n"     \
  "5: \n"        \
  SMMLA_LEAKYRELU_8x4 \
  "b 10f\n"     \
  "10: \n"

#define SMMLA_STORE_FP32_8x4            \
  "st1w {z30.s},  p0, [%x[c_ptr0]]\n"   \
  "st1w {z29.s},  p0, [%x[c_ptr1]]\n"   \
  "st1w {z28.s},  p0, [%x[c_ptr2]]\n"   \
  "st1w {z27.s},  p0, [%x[c_ptr3]]\n"   \
  "st1w {z26.s},  p0, [%x[c_ptr4]]\n"   \
  "st1w {z25.s},  p0, [%x[c_ptr5]]\n"   \
  "st1w {z24.s},  p0, [%x[c_ptr6]]\n"   \
  "st1w {z23.s},  p0, [%x[c_ptr7]]\n"   \
  "add %x[c_ptr0], %x[c_ptr0], #0x10\n" \
  "add %x[c_ptr1], %x[c_ptr1], #0x10\n" \
  "add %x[c_ptr2], %x[c_ptr2], #0x10\n" \
  "add %x[c_ptr3], %x[c_ptr3], #0x10\n" \
  "add %x[c_ptr4], %x[c_ptr4], #0x10\n" \
  "add %x[c_ptr5], %x[c_ptr5], #0x10\n" \
  "add %x[c_ptr6], %x[c_ptr6], #0x10\n" \
  "add %x[c_ptr7], %x[c_ptr7], #0x10\n"

#define SMMLA_STORE_INT8_8x4 \
  "ptrue p0.b\n" \
  "ld1rqw {z2.s}, p0/Z, [%x[vmax]]\n" \
  "fcmge v22.4s,  v30.4s,  v2.4s       \n" \
  "fcmge v21.4s,  v29.4s,  v2.4s       \n" \
  "fcmge v20.4s,  v28.4s,  v2.4s       \n" \
  "fcmge v19.4s,  v27.4s,  v2.4s       \n" \
  "fcmge v18.4s,  v26.4s,  v2.4s       \n" \
  "fcmge v17.4s,  v25.4s,  v2.4s       \n" \
  "fcmge v16.4s,  v24.4s,  v2.4s       \n" \
  "fcmge v15.4s,  v23.4s,  v2.4s       \n" \
  "bif   v30.16b, v2.16b, v22.16b      \n" \
  "bif   v29.16b, v2.16b, v21.16b      \n" \
  "bif   v28.16b, v2.16b, v20.16b      \n" \
  "bif   v27.16b, v2.16b, v19.16b      \n" \
  "bif   v26.16b, v2.16b, v18.16b      \n" \
  "bif   v25.16b, v2.16b, v17.16b      \n" \
  "bif   v24.16b, v2.16b, v16.16b      \n" \
  "bif   v23.16b, v2.16b, v15.16b      \n" \
  /* fp32 -> int32 */                      \
  "fcvtas  v22.4s,  v30.4s              \n" \
  "fcvtas  v21.4s,  v29.4s              \n" \
  "fcvtas  v20.4s,  v28.4s              \n" \
  "fcvtas  v19.4s,  v27.4s              \n" \
  "fcvtas  v18.4s,  v26.4s              \n" \
  "fcvtas  v17.4s,  v25.4s              \n" \
  "fcvtas  v16.4s,  v24.4s              \n" \
  "fcvtas  v15.4s,  v23.4s              \n" \
  /* int32 -> int16 */                     \
  "sqxtn   v30.4h,  v22.4s              \n" \
  "sqxtn   v29.4h,  v21.4s              \n" \
  "sqxtn   v28.4h,  v20.4s              \n" \
  "sqxtn   v27.4h,  v19.4s              \n" \
  "sqxtn   v26.4h,  v18.4s              \n" \
  "sqxtn   v25.4h,  v17.4s              \n" \
  "sqxtn   v24.4h,  v16.4s              \n" \
  "sqxtn   v23.4h,  v15.4s              \n" \
  /* int16 -> int8  */                     \
  "sqxtn   v22.8b,  v30.8h           \n" \
  "sqxtn   v21.8b,  v29.8h           \n" \
  "sqxtn   v20.8b,  v28.8h           \n" \
  "sqxtn   v19.8b,  v27.8h           \n" \
  "sqxtn   v18.8b,  v26.8h           \n" \
  "sqxtn   v17.8b,  v25.8h           \n" \
  "sqxtn   v16.8b,  v24.8h           \n" \
  "sqxtn   v15.8b,  v23.8h           \n" \
  "str s22, [%[c_ptr0]], #4\n"           \
  "str s21, [%[c_ptr1]], #4\n"           \
  "str s20, [%[c_ptr2]], #4\n"           \
  "str s19, [%[c_ptr3]], #4\n"           \
  "str s18, [%[c_ptr4]], #4\n"           \
  "str s17, [%[c_ptr5]], #4\n"           \
  "str s16, [%[c_ptr6]], #4\n"           \
  "str s15, [%[c_ptr7]], #4\n"

#define SMMLA_STORE_INT8_8x1               \
  "ptrue p0.b\n"                      \
  "ld1rqw {z2.s}, p0/Z, [%x[vmax]]\n" \
  "fcmge v22.4s,  v30.4s,  v2.4s       \n" \
  "fcmge v21.4s,  v29.4s,  v2.4s       \n" \
  "fcmge v20.4s,  v28.4s,  v2.4s       \n" \
  "fcmge v19.4s,  v27.4s,  v2.4s       \n" \
  "fcmge v18.4s,  v26.4s,  v2.4s       \n" \
  "fcmge v17.4s,  v25.4s,  v2.4s       \n" \
  "fcmge v16.4s,  v24.4s,  v2.4s       \n" \
  "fcmge v15.4s,  v23.4s,  v2.4s       \n" \
  "bif   v30.16b, v2.16b, v22.16b      \n" \
  "bif   v29.16b, v2.16b, v21.16b      \n" \
  "bif   v28.16b, v2.16b, v20.16b      \n" \
  "bif   v27.16b, v2.16b, v19.16b      \n" \
  "bif   v26.16b, v2.16b, v18.16b      \n" \
  "bif   v25.16b, v2.16b, v17.16b      \n" \
  "bif   v24.16b, v2.16b, v16.16b      \n" \
  "bif   v23.16b, v2.16b, v15.16b      \n" \
  /* fp32 -> int32 */                      \
  "fcvtas  v22.4s,  v30.4s              \n" \
  "fcvtas  v21.4s,  v29.4s              \n" \
  "fcvtas  v20.4s,  v28.4s              \n" \
  "fcvtas  v19.4s,  v27.4s              \n" \
  "fcvtas  v18.4s,  v26.4s              \n" \
  "fcvtas  v17.4s,  v25.4s              \n" \
  "fcvtas  v16.4s,  v24.4s              \n" \
  "fcvtas  v15.4s,  v23.4s              \n" \
  /* int32 -> int16 */                     \
  "sqxtn   v30.4h,  v22.4s              \n" \
  "sqxtn   v29.4h,  v21.4s              \n" \
  "sqxtn   v28.4h,  v20.4s              \n" \
  "sqxtn   v27.4h,  v19.4s              \n" \
  "sqxtn   v26.4h,  v18.4s              \n" \
  "sqxtn   v25.4h,  v17.4s              \n" \
  "sqxtn   v24.4h,  v16.4s              \n" \
  "sqxtn   v23.4h,  v15.4s              \n" \
  /* int16 -> int8  */                     \
  "sqxtn   v22.8b,  v30.8h           \n" \
  "sqxtn   v21.8b,  v29.8h           \n" \
  "sqxtn   v20.8b,  v28.8h           \n" \
  "sqxtn   v19.8b,  v27.8h           \n" \
  "sqxtn   v18.8b,  v26.8h           \n" \
  "sqxtn   v17.8b,  v25.8h           \n" \
  "sqxtn   v16.8b,  v24.8h           \n" \
  "sqxtn   v15.8b,  v23.8h           \n" \
  "str s22, [%[c_ptr0]]\n"           \
  "str s21, [%[c_ptr1]]\n"           \
  "str s20, [%[c_ptr2]]\n"           \
  "str s19, [%[c_ptr3]]\n"           \
  "str s18, [%[c_ptr4]]\n"           \
  "str s17, [%[c_ptr5]]\n"           \
  "str s16, [%[c_ptr6]]\n"           \
  "str s15, [%[c_ptr7]]\n"

#define SMMLA_STORE_INT8_8x4_1          \
  "mov z0.s, #0x0\n"                  \
  "ld1rqw {z1.s}, p0/Z, [%x[alpha], #0x30]\n" \
  "mov z2.s, #127\n"            \
  "movprfx z15, z30\n fmin z15.s, p0/m, z15.s, z0.s\n" \
  "movprfx z14, z29\n fmin z14.s, p0/m, z14.s, z0.s\n" \
  "movprfx z13, z28\n fmin z13.s, p0/m, z13.s, z0.s\n" \
  "movprfx z12, z27\n fmin z12.s, p0/m, z12.s, z0.s\n" \
  "fmax z30.s, p0/m, z30.s, z0.s\n" \
  "fmax z29.s, p0/m, z29.s, z0.s\n" \
  "fmax z28.s, p0/m, z28.s, z0.s\n" \
  "fmax z27.s, p0/m, z27.s, z0.s\n" \
  /* a + 0.5 */ \
  "fadd z30.s, p0/m, z30.s, z1.s\n" \
  "fadd z29.s, p0/m, z29.s, z1.s\n" \
  "fadd z28.s, p0/m, z28.s, z1.s\n" \
  "fadd z27.s, p0/m, z27.s, z1.s\n" \
  /* a - 0.5 */ \
  "fsub z15.s, p0/m, z15.s, z1.s\n" \
  "fsub z14.s, p0/m, z14.s, z1.s\n" \
  "fsub z13.s, p0/m, z13.s, z1.s\n" \
  "fsub z12.s, p0/m, z12.s, z1.s\n" \
  /* fp32->int32 */ \
  "fcvtzs z30.s, p0/m, z30.s\n" \
  "fcvtzs z29.s, p0/m, z29.s\n" \
  "fcvtzs z28.s, p0/m, z28.s\n" \
  "fcvtzs z27.s, p0/m, z27.s\n" \
  "fcvtzs z15.s, p0/m, z15.s\n" \
  "fcvtzs z14.s, p0/m, z14.s\n" \
  "fcvtzs z13.s, p0/m, z13.s\n" \
  "fcvtzs z12.s, p0/m, z12.s\n" \
  "add z30.s, p0/m, z30.s, z15.s\n" \
  "add z29.s, p0/m, z29.s, z14.s\n" \
  "add z28.s, p0/m, z28.s, z13.s\n" \
  "add z27.s, p0/m, z27.s, z12.s\n" \
  /* a + 127 */ \
  "add z30.s, p0/m, z30.s, z2.s\n" \
  "add z29.s, p0/m, z29.s, z2.s\n" \
  "add z28.s, p0/m, z28.s, z2.s\n" \
  "add z27.s, p0/m, z27.s, z2.s\n" \
  /* a  + 127 > 0 */ \
  "smax z30.s, p0/m, z30.s, z0.s\n"\
  "smax z29.s, p0/m, z29.s, z0.s\n"\
  "smax z28.s, p0/m, z28.s, z0.s\n"\
  "smax z27.s, p0/m, z27.s, z0.s\n"\
  /* a - 127 */ \
  "sub z30.s, p0/m, z30.s, z2.s\n" \
  "sub z29.s, p0/m, z29.s, z2.s\n" \
  "sub z28.s, p0/m, z28.s, z2.s\n" \
  "sub z27.s, p0/m, z27.s, z2.s\n" \
  "mov x0, #12\n" \
  "mov x1, #16\n" \
  "whilelt p1.b, x0, x1\n" \
  /* int32-> int8*/ \
  "uzp1 z15.h, z30.h, z30.h\n" \
  "uzp1 z14.h, z29.h, z29.h\n" \
  "uzp1 z13.h, z28.h, z28.h\n" \
  "uzp1 z12.h, z27.h, z27.h\n" \
  "uzp1 z30.b, z15.b, z15.b\n" \
  "uzp1 z29.b, z14.b, z14.b\n" \
  "uzp1 z28.b, z13.b, z13.b\n" \
  "uzp1 z27.b, z12.b, z12.b\n" \
  "movprfx z15, z26\n fmin z15.s, p0/m, z15.s, z0.s\n" \
  "movprfx z14, z25\n fmin z14.s, p0/m, z14.s, z0.s\n" \
  "movprfx z13, z24\n fmin z13.s, p0/m, z13.s, z0.s\n" \
  "movprfx z12, z23\n fmin z12.s, p0/m, z12.s, z0.s\n" \
  "fmax z26.s, p0/m, z26.s, z0.s\n" \
  "fmax z25.s, p0/m, z25.s, z0.s\n" \
  "fmax z24.s, p0/m, z24.s, z0.s\n" \
  "fmax z23.s, p0/m, z23.s, z0.s\n" \
  /* a + 0.5 */ \
  "fadd z26.s, p0/m, z26.s, z1.s\n" \
  "fadd z25.s, p0/m, z25.s, z1.s\n" \
  "fadd z24.s, p0/m, z24.s, z1.s\n" \
  "fadd z23.s, p0/m, z23.s, z1.s\n" \
  /* a - 0.5 */ \
  "fsub z15.s, p0/m, z15.s, z1.s\n" \
  "fsub z14.s, p0/m, z14.s, z1.s\n" \
  "fsub z13.s, p0/m, z13.s, z1.s\n" \
  "fsub z12.s, p0/m, z12.s, z1.s\n" \
  /* fp32->int32 */ \
  "fcvtzs z26.s, p0/m, z26.s\n" \
  "fcvtzs z25.s, p0/m, z25.s\n" \
  "fcvtzs z24.s, p0/m, z24.s\n" \
  "fcvtzs z23.s, p0/m, z23.s\n" \
  "fcvtzs z15.s, p0/m, z15.s\n" \
  "fcvtzs z14.s, p0/m, z14.s\n" \
  "fcvtzs z13.s, p0/m, z13.s\n" \
  "fcvtzs z12.s, p0/m, z12.s\n" \
  "add z26.s, p0/m, z26.s, z15.s\n" \
  "add z25.s, p0/m, z25.s, z14.s\n" \
  "add z24.s, p0/m, z24.s, z13.s\n" \
  "add z23.s, p0/m, z23.s, z12.s\n" \
  /* a + 127 */ \
  "add z26.s, p0/m, z26.s, z2.s\n" \
  "add z25.s, p0/m, z25.s, z2.s\n" \
  "add z24.s, p0/m, z24.s, z2.s\n" \
  "add z23.s, p0/m, z23.s, z2.s\n" \
  /* a  + 127 > 0 */ \
  "smax z26.s, p0/m, z26.s, z0.s\n"\
  "smax z25.s, p0/m, z25.s, z0.s\n"\
  "smax z24.s, p0/m, z24.s, z0.s\n"\
  "smax z23.s, p0/m, z23.s, z0.s\n"\
  /* a - 127 */ \
  "sub z26.s, p0/m, z26.s, z2.s\n" \
  "sub z25.s, p0/m, z25.s, z2.s\n" \
  "sub z24.s, p0/m, z24.s, z2.s\n" \
  "sub z23.s, p0/m, z23.s, z2.s\n" \
  /* int32-> int8*/ \
  "uzp1 z15.h, z30.h, z30.h\n" \
  "uzp1 z14.h, z29.h, z29.h\n" \
  "uzp1 z13.h, z28.h, z28.h\n" \
  "uzp1 z12.h, z27.h, z27.h\n" \
  "uzp1 z30.b, z15.b, z15.b\n" \
  "uzp1 z29.b, z14.b, z14.b\n" \
  "uzp1 z28.b, z13.b, z13.b\n" \
  "uzp1 z27.b, z12.b, z12.b\n" \
  "st1b {z30.b},  p1, [%x[c_ptr0]]\n" \
  "st1b {z29.b},  p1, [%x[c_ptr1]]\n" \
  "st1b {z28.b},  p1, [%x[c_ptr2]]\n" \
  "st1b {z27.b},  p1, [%x[c_ptr3]]\n" \
  "st1b {z26.b},  p1, [%x[c_ptr4]]\n" \
  "st1b {z25.b},  p1, [%x[c_ptr5]]\n" \
  "st1b {z24.b},  p1, [%x[c_ptr6]]\n" \
  "st1b {z23.b},  p1, [%x[c_ptr7]]\n" \
  "add %[c_ptr0], %[c_ptr0], #0x4\n"  \
  "add %[c_ptr1], %[c_ptr1], #0x4\n"  \
  "add %[c_ptr2], %[c_ptr2], #0x4\n"  \
  "add %[c_ptr3], %[c_ptr3], #0x4\n"  \
  "add %[c_ptr4], %[c_ptr4], #0x4\n"  \
  "add %[c_ptr5], %[c_ptr5], #0x4\n"  \
  "add %[c_ptr6], %[c_ptr6], #0x4\n"  \
  "add %[c_ptr7], %[c_ptr7], #0x4\n"

#define INIT_SMMLA_8x12                             \
  "ptrue p0.b \n"                                   \
  "prfm   pldl1keep, [%[a_ptr]]\n"                  \
  "prfm   pldl1keep, [%[b_ptr]]\n"                  \
  "mov    z8.s,   #0x0\n"                           \
  "mov    z14.s,  #0x0\n"                           \
  "mov    z20.s,  #0x0\n"                           \
  "mov    z26.s,  #0x0\n"                           \
  "ld1rqb {z0.b}, p0/Z, [%x[a_ptr]]\n"              \
  "prfm   pldl1keep, [%[a_ptr], #64]\n"             \
  "prfm   pldl1keep, [%[b_ptr], #64]\n"             \
  "ld1b   {z4.b}, p0/Z, [%x[b_ptr]]\n"              \
  "mov    z9.s,   #0x0\n"                           \
  "mov    z15.s,  #0x0\n"                           \
  "mov    z21.s,  #0x0\n"                           \
  "mov    z27.s,  #0x0\n"                           \
  "ld1rqb {z1.b}, p0/Z, [%x[a_ptr], #0x10]\n"       \
  "ld1b   {z5.b}, p0/Z, [%x[b_ptr],  #1, MUL VL]\n" \
  "prfm   pldl1keep, [%[a_ptr], #128]\n"            \
  "prfm   pldl1keep, [%[b_ptr], #128]\n"            \
  "mov    z10.s,  #0x0\n"                           \
  "mov    z16.s,  #0x0\n"                           \
  "mov    z22.s,  #0x0\n"                           \
  "mov    z28.s,  #0x0\n"                           \
  "ld1rqb {z2.b}, p0/Z, [%x[a_ptr], #0x20]\n"       \
  "ld1b   {z6.b}, p0/Z, [%x[b_ptr],  #2, MUL VL]\n" \
  "prfm   pldl1keep, [%[a_ptr], #192]\n"            \
  "prfm   pldl1keep, [%[b_ptr], #192]\n"            \
  "mov    z11.s,  #0x0\n"                           \
  "mov    z17.s,  #0x0\n"                           \
  "mov    z23.s,  #0x0\n"                           \
  "mov    z29.s,  #0x0\n"                           \
  "prfm   pldl1keep, [%[a_ptr], #256]\n"            \
  "prfm   pldl1keep, [%[b_ptr], #256]\n"            \
  "mov    z12.s,  #0x0\n"                           \
  "mov    z18.s,  #0x0\n"                           \
  "mov    z24.s,  #0x0\n"                           \
  "mov    z30.s,  #0x0\n"                           \
  "prfm   pldl1keep, [%[b_ptr], #320]\n"            \
  "mov    z13.s,  #0x0\n"                           \
  "mov    z19.s,  #0x0\n"                           \
  "mov    z25.s,  #0x0\n"                           \
  "mov    z31.s,  #0x0\n"                           \
  "cbz    %x[k],  1f\n"

#define COMPUTE_SMMLA_8x12_0                         \
  "0: \n"                                            \
  "ld1rqb {z3.b}, p0/Z, [%x[a_ptr], #0x30]\n"        \
  "ld1b   {z7.b}, p0/Z, [%x[b_ptr],  #3, MUL VL]\n"  \
  "smmla  z8.s,  z0.b, z4.b\n"                       \
  "smmla  z14.s, z1.b, z4.b\n"                       \
  "smmla  z20.s, z2.b, z4.b\n"                       \
  "smmla  z26.s, z3.b, z4.b\n"                       \
  "ld1b   {z4.b}, p0/Z, [%x[b_ptr],  #4, MUL VL]\n"  \
  "smmla  z9.s,  z0.b, z5.b\n"                       \
  "smmla  z15.s, z1.b, z5.b\n"                       \
  "smmla  z21.s, z2.b, z5.b\n"                       \
  "smmla  z27.s, z3.b, z5.b\n"                       \
  "ld1b   {z5.b}, p0/Z, [%x[b_ptr],  #5, MUL VL]\n"  \
  "addvl %x[b_ptr], %x[b_ptr], #12\n"                \
  "smmla  z10.s, z0.b, z6.b\n"                       \
  "smmla  z16.s, z1.b, z6.b\n"                       \
  "smmla  z22.s, z2.b, z6.b\n"                       \
  "smmla  z28.s, z3.b, z6.b\n"                       \
  "ld1b   {z6.b}, p0/Z, [%x[b_ptr],  #-6, MUL VL]\n" \
  "smmla  z11.s, z0.b, z7.b\n"                       \
  "smmla  z17.s, z1.b, z7.b\n"                       \
  "smmla  z23.s, z2.b, z7.b\n"                       \
  "smmla  z29.s, z3.b, z7.b\n"                       \
  "ld1b   {z7.b}, p0/Z, [%x[b_ptr],  #-5, MUL VL]\n" \
  "smmla  z12.s, z0.b, z4.b\n"                       \
  "smmla  z13.s, z0.b, z5.b\n"                       \
  "ld1rqb {z0.b}, p0/Z, [%x[a_ptr], #0x40]\n"        \
  "smmla  z18.s, z1.b, z4.b\n"                       \
  "smmla  z19.s, z1.b, z5.b\n"                       \
  "ld1rqb {z1.b}, p0/Z, [%x[a_ptr], #0x50]\n"        \
  "smmla  z24.s, z2.b, z4.b\n"                       \
  "smmla  z25.s, z2.b, z5.b\n"                       \
  "ld1rqb {z2.b}, p0/Z, [%x[a_ptr], #0x60]\n"        \
  "smmla  z30.s, z3.b, z4.b\n"                       \
  "smmla  z31.s, z3.b, z5.b\n"                       \
  "ld1rqb {z3.b}, p0/Z, [%x[a_ptr], #0x70]\n"        \
  "ld1b   {z4.b}, p0/Z, [%x[b_ptr],  #-4, MUL VL]\n"

#define COMPUTE_SMMLA_8x12_1                         \
  "smmla  z8.s,  z0.b, z6.b\n"                       \
  "smmla  z14.s, z1.b, z6.b\n"                       \
  "ld1b   {z5.b}, p0/Z, [%x[b_ptr],  #-3, MUL VL]\n" \
  "smmla  z20.s, z2.b, z6.b\n"                       \
  "smmla  z26.s, z3.b, z6.b\n"                       \
  "ld1b   {z6.b}, p0/Z, [%x[b_ptr],  #-2, MUL VL]\n" \
  "smmla  z9.s,  z0.b, z7.b\n"                       \
  "smmla  z15.s, z1.b, z7.b\n"                       \
  "smmla  z21.s, z2.b, z7.b\n"                       \
  "smmla  z27.s, z3.b, z7.b\n"                       \
  "ld1b   {z7.b}, p0/Z, [%x[b_ptr],  #-1, MUL VL]\n" \
  "smmla  z10.s, z0.b, z4.b\n"                       \
  "smmla  z16.s, z1.b, z4.b\n"                       \
  "smmla  z22.s, z2.b, z4.b\n"                       \
  "smmla  z28.s, z3.b, z4.b\n"                       \
  "add    %x[a_ptr], %x[a_ptr], #0x80\n"             \
  "smmla  z11.s, z0.b, z5.b\n"                       \
  "smmla  z17.s, z1.b, z5.b\n"                       \
  "smmla  z23.s, z2.b, z5.b\n"                       \
  "smmla  z29.s, z3.b, z5.b\n"                       \
  "subs   %x[k], %x[k], #1\n"                        \
  "ld1b   {z4.b}, p0/Z, [%x[b_ptr]]\n"               \
  "smmla  z12.s, z0.b, z6.b\n"                       \
  "smmla  z13.s, z0.b, z7.b\n"                       \
  "ld1b   {z5.b}, p0/Z, [%x[b_ptr],  #1, MUL VL]\n"  \
  "smmla  z18.s, z1.b, z6.b\n"                       \
  "smmla  z19.s, z1.b, z7.b\n"                       \
  "ld1rqb {z0.b}, p0/Z, [%x[a_ptr]]\n"               \
  "smmla  z24.s, z2.b, z6.b\n"                       \
  "smmla  z25.s, z2.b, z7.b\n"                       \
  "ld1rqb {z1.b}, p0/Z, [%x[a_ptr], #0x10]\n"        \
  "smmla  z30.s, z3.b, z6.b\n"                       \
  "smmla  z31.s, z3.b, z7.b\n"                       \
  "ld1rqb {z2.b}, p0/Z, [%x[a_ptr], #0x20]\n"        \
  "ld1b   {z6.b}, p0/Z, [%x[b_ptr],  #2, MUL VL]\n"  \
  "bne 0b\n"

#define COMPUTE_SMMLA_8x12_REMAIN                    \
  "1: \n"                                            \
  "cmp %x[rem_cnt], #2\n"                            \
  "beq 2f\n"                                         \
  "ld1rqb {z3.b}, p0/Z, [%x[a_ptr], #0x30]\n"        \
  "ld1b   {z7.b}, p0/Z, [%x[b_ptr],  #3, MUL VL]\n"  \
  "smmla  z8.s,  z0.b, z4.b\n"                       \
  "smmla  z14.s, z1.b, z4.b\n"                       \
  "smmla  z20.s, z2.b, z4.b\n"                       \
  "smmla  z26.s, z3.b, z4.b\n"                       \
  "ld1b   {z4.b}, p0/Z, [%x[b_ptr],  #4, MUL VL]\n"  \
  "smmla  z9.s,  z0.b, z5.b\n"                       \
  "smmla  z15.s, z1.b, z5.b\n"                       \
  "smmla  z21.s, z2.b, z5.b\n"                       \
  "smmla  z27.s, z3.b, z5.b\n"                       \
  "ld1b   {z5.b}, p0/Z, [%x[b_ptr],  #5, MUL VL]\n"  \
  "addvl %x[b_ptr], %x[b_ptr], #6\n"                 \
  "smmla  z10.s, z0.b, z6.b\n"                       \
  "smmla  z16.s, z1.b, z6.b\n"                       \
  "smmla  z22.s, z2.b, z6.b\n"                       \
  "smmla  z28.s, z3.b, z6.b\n"                       \
  "add    %x[a_ptr], %x[a_ptr], #0x40\n"             \
  "smmla  z11.s, z0.b, z7.b\n"                       \
  "smmla  z17.s, z1.b, z7.b\n"                       \
  "smmla  z23.s, z2.b, z7.b\n"                       \
  "smmla  z29.s, z3.b, z7.b\n"                       \
  "smmla  z12.s, z0.b, z4.b\n"                       \
  "smmla  z18.s, z1.b, z4.b\n"                       \
  "smmla  z24.s, z2.b, z4.b\n"                       \
  "smmla  z30.s, z3.b, z4.b\n"                       \
  "smmla  z13.s, z0.b, z5.b\n"                       \
  "smmla  z19.s, z1.b, z5.b\n"                       \
  "smmla  z25.s, z2.b, z5.b\n"                       \
  "smmla  z31.s, z3.b, z5.b\n"                       \
  "b 3f\n"                                           \
  "2: \n" COMPUTE_SMMLA_8x12_0                       \
  "smmla  z8.s,  z0.b, z6.b\n"                       \
  "smmla  z14.s, z1.b, z6.b\n"                       \
  "ld1b   {z5.b}, p0/Z, [%x[b_ptr],  #-3, MUL VL]\n" \
  "smmla  z20.s, z2.b, z6.b\n"                       \
  "smmla  z26.s, z3.b, z6.b\n"                       \
  "ld1b   {z6.b}, p0/Z, [%x[b_ptr],  #-2, MUL VL]\n" \
  "smmla  z9.s,  z0.b, z7.b\n"                       \
  "smmla  z15.s, z1.b, z7.b\n"                       \
  "smmla  z21.s, z2.b, z7.b\n"                       \
  "smmla  z27.s, z3.b, z7.b\n"                       \
  "ld1b   {z7.b}, p0/Z, [%x[b_ptr],  #-1, MUL VL]\n" \
  "smmla  z10.s, z0.b, z4.b\n"                       \
  "smmla  z16.s, z1.b, z4.b\n"                       \
  "smmla  z22.s, z2.b, z4.b\n"                       \
  "smmla  z28.s, z3.b, z4.b\n"                       \
  "add    %x[a_ptr], %x[a_ptr], #0x80\n"             \
  "smmla  z11.s, z0.b, z5.b\n"                       \
  "smmla  z17.s, z1.b, z5.b\n"                       \
  "smmla  z23.s, z2.b, z5.b\n"                       \
  "smmla  z29.s, z3.b, z5.b\n"                       \
  "smmla  z12.s, z0.b, z6.b\n"                       \
  "smmla  z18.s, z1.b, z6.b\n"                       \
  "smmla  z24.s, z2.b, z6.b\n"                       \
  "smmla  z30.s, z3.b, z6.b\n"                       \
  "smmla  z13.s, z0.b, z7.b\n"                       \
  "smmla  z19.s, z1.b, z7.b\n"                       \
  "smmla  z25.s, z2.b, z7.b\n"                       \
  "smmla  z31.s, z3.b, z7.b\n"

#define CVT_SMMLA_INT32_TO_FP32_8x12                    \
  "3: \n"                                               \
  "ld1rqw {z0.s}, p0/Z, [%x[bias]]\n"                   \
  "ld1rqw {z1.s}, p0/Z, [%x[bias], #0x10]\n"            \
  "trn1 z2.d,  z8.d,  z9.d\n"                           \
  "trn2 z3.d,  z8.d,  z9.d\n"                           \
  "trn1 z4.d,  z10.d, z11.d\n"                          \
  "trn2 z5.d,  z10.d, z11.d\n"                          \
  "trn1 z6.d,  z12.d, z13.d\n"                          \
  "trn2 z7.d,  z12.d, z13.d\n"                          \
  "trn1 z8.d,  z14.d, z15.d\n"                          \
  "trn2 z9.d,  z14.d, z15.d\n"                          \
  "trn1 z10.d, z16.d, z17.d\n"                          \
  "trn2 z11.d, z16.d, z17.d\n"                          \
  "trn1 z12.d, z18.d, z19.d\n"                          \
  "trn2 z13.d, z18.d, z19.d\n"                          \
  "trn1 z14.d, z20.d, z21.d\n"                          \
  "trn2 z15.d, z20.d, z21.d\n"                          \
  "trn1 z16.d, z22.d, z23.d\n"                          \
  "trn2 z17.d, z22.d, z23.d\n"                          \
  "trn1 z18.d, z24.d, z25.d\n"                          \
  "trn2 z19.d, z24.d, z25.d\n"                          \
  "trn1 z20.d, z26.d, z27.d\n"                          \
  "trn2 z21.d, z26.d, z27.d\n"                          \
  "trn1 z22.d, z28.d, z29.d\n"                          \
  "trn2 z23.d, z28.d, z29.d\n"                          \
  "trn1 z24.d, z30.d, z31.d\n"                          \
  "trn2 z25.d, z30.d, z31.d\n"                          \
  "mov z28.s, #0x0\n"                                   \
  "movprfx z26, z2\n fadd z26.s, p0/m, z26.s, z28.s\n"  \
  "ld1rqw {z2.s}, p0/Z, [%x[scale]]\n"                  \
  "movprfx z27, z3\n fadd z27.s, p0/m, z27.s, z28.s\n"  \
  "dup    z29.s,  z0.s[0]\n"                            \
  "dup    z30.s,  z0.s[0]\n"                            \
  "dup    z31.s,  z0.s[0]\n"                            \
  "ld1rqw {z3.s}, p0/Z, [%x[scale], #0x10]\n"           \
  "scvtf  z26.s, p0/m, z26.s\n"                         \
  "scvtf  z4.s,  p0/m, z4.s\n"                          \
  "scvtf  z6.s,  p0/m, z6.s\n"                          \
  "scvtf  z27.s, p0/m, z27.s\n"                         \
  "scvtf  z5.s,  p0/m, z5.s\n"                          \
  "scvtf  z7.s,  p0/m, z7.s\n"                          \
  "scvtf  z8.s,  p0/m, z8.s\n"                          \
  "scvtf  z10.s, p0/m, z10.s\n"                         \
  "scvtf  z12.s, p0/m, z12.s\n" /* c0-12 z29 z30 z31*/  \
  "fmla   z29.s, z26.s, z2.s[0]\n"                      \
  "dup    z26.s, z0.s[1]\n"                             \
  "fmla   z30.s, z4.s,  z2.s[0]\n"                      \
  "dup    z4.s,  z0.s[1]\n"                             \
  "fmla   z31.s, z6.s,  z2.s[0]\n"                      \
  "dup    z6.s,  z0.s[1]\n"                             \
  "scvtf  z9.s,  p0/m, z9.s\n"                          \
  "scvtf  z11.s, p0/m, z11.s\n"                         \
  "scvtf  z13.s, p0/m, z13.s\n" /* c1-12 z26 z4 z6*/    \
  "fmla   z26.s, z27.s, z2.s[1]\n"                      \
  "dup    z27.s, z0.s[2]\n"                             \
  "fmla   z4.s,  z5.s,  z2.s[1]\n"                      \
  "dup    z5.s,  z0.s[2]\n"                             \
  "fmla   z6.s,  z7.s,  z2.s[1]\n"                      \
  "dup    z7.s,  z0.s[2]\n"                             \
  "scvtf  z14.s, p0/m, z14.s\n"                         \
  "scvtf  z16.s, p0/m, z16.s\n"                         \
  "scvtf  z18.s, p0/m, z18.s\n" /* c2-12 z27 z5 z7*/    \
  "fmla   z27.s, z8.s, z2.s[2]\n"                       \
  "dup    z8.s,  z0.s[3]\n"                             \
  "fmla   z5.s,  z10.s,  z2.s[2]\n"                     \
  "dup    z10.s, z0.s[3]\n"                             \
  "fmla   z7.s,  z12.s,  z2.s[2]\n"                     \
  "dup    z12.s, z0.s[3]\n"                             \
  "scvtf  z15.s, p0/m, z15.s\n"                         \
  "scvtf  z17.s, p0/m, z17.s\n"                         \
  "scvtf  z19.s, p0/m, z19.s\n" /* c3-12 z8 z10 z12*/   \
  "fmla   z8.s,   z9.s, z2.s[3]\n"                      \
  "dup    z9.s,   z1.s[0]\n"                            \
  "fmla   z10.s,  z11.s,  z2.s[3]\n"                    \
  "dup    z11.s,  z1.s[0]\n"                            \
  "fmla   z12.s,  z13.s,  z2.s[3]\n"                    \
  "dup    z13.s,  z1.s[0]\n"                            \
  "scvtf  z20.s,  p0/m, z20.s\n"                        \
  "scvtf  z22.s,  p0/m, z22.s\n"                        \
  "scvtf  z24.s,  p0/m, z24.s\n" /* c4-12 z9 z11 z13*/  \
  "fmla   z9.s,   z14.s, z3.s[0]\n"                     \
  "dup    z14.s,  z1.s[1]\n"                            \
  "fmla   z11.s,  z16.s,  z3.s[0]\n"                    \
  "dup    z16.s,  z1.s[1]\n"                            \
  "fmla   z13.s,  z18.s,  z3.s[0]\n"                    \
  "dup    z18.s,  z1.s[1]\n"                            \
  "scvtf  z21.s,  p0/m, z21.s\n"                        \
  "scvtf  z23.s,  p0/m, z23.s\n"                        \
  "scvtf  z25.s,  p0/m, z25.s\n" /* c5-12 z14 z16 z18*/ \
  "fmla   z14.s,  z15.s, z3.s[1]\n"                     \
  "dup    z15.s,  z1.s[2]\n"                            \
  "fmla   z16.s,  z17.s,  z3.s[1]\n"                    \
  "dup    z17.s,  z1.s[2]\n"                            \
  "fmla   z18.s,  z19.s,  z3.s[1]\n"                    \
  "dup    z19.s,  z1.s[2]\n" /* c6-12 z15 z17 z19*/     \
  "fmla   z15.s,  z20.s, z3.s[2]\n"                     \
  "dup    z20.s,  z1.s[3]\n"                            \
  "fmla   z17.s,  z22.s,  z3.s[2]\n"                    \
  "dup    z22.s,  z1.s[3]\n"                            \
  "fmla   z19.s,  z24.s,  z3.s[2]\n"                    \
  "dup    z24.s,  z1.s[3]\n" /* c7-12 z20 z22 z24*/     \
  "fmla   z20.s,  z21.s,  z3.s[3]\n"                    \
  "fmla   z22.s,  z23.s,  z3.s[3]\n"                    \
  "fmla   z24.s,  z25.s,  z3.s[3]\n"


#define SMMLA_RELU_8x12(inst) \
  #inst " z29.s,  p0/m, z29.s, z0.s\n" \
  #inst " z30.s,  p0/m, z30.s, z0.s\n" \
  #inst " z31.s,  p0/m, z31.s, z0.s\n" \
  #inst " z26.s,  p0/m, z26.s, z0.s\n" \
  #inst " z4.s,   p0/m, z4.s,  z0.s\n" \
  #inst " z6.s,   p0/m, z6.s,  z0.s\n" \
  #inst " z27.s,  p0/m, z27.s, z0.s\n" \
  #inst " z5.s,   p0/m, z5.s,  z0.s\n" \
  #inst " z7.s,   p0/m, z7.s,  z0.s\n" \
  #inst " z8.s,   p0/m, z8.s,  z0.s\n" \
  #inst " z10.s,  p0/m, z10.s, z0.s\n" \
  #inst " z12.s,  p0/m, z12.s, z0.s\n" \
  #inst " z9.s,   p0/m, z9.s,  z0.s\n" \
  #inst " z11.s,  p0/m, z11.s, z0.s\n" \
  #inst " z13.s,  p0/m, z13.s, z0.s\n" \
  #inst " z14.s,  p0/m, z14.s, z0.s\n" \
  #inst " z16.s,  p0/m, z16.s, z0.s\n" \
  #inst " z18.s,  p0/m, z18.s, z0.s\n" \
  #inst " z15.s,  p0/m, z15.s, z0.s\n" \
  #inst " z17.s,  p0/m, z17.s, z0.s\n" \
  #inst " z19.s,  p0/m, z19.s, z0.s\n" \
  #inst " z20.s,  p0/m, z20.s, z0.s\n" \
  #inst " z22.s,  p0/m, z22.s, z0.s\n" \
  #inst " z24.s,  p0/m, z24.s, z0.s\n"

#define SMMLA_LEAKYRELU_8x12                             \
  "mov z0.s, #0x0    \n"                                 \
  "ld1rqw {z1.s}, p0/Z, [%x[alpha]]\n"                   \
  "movprfx z21, z29\n  fmin z21.s,  p0/m, z21.s, z0.s\n" \
  "movprfx z23, z30\n  fmin z23.s,  p0/m, z23.s, z0.s\n" \
  "movprfx z25, z31\n  fmin z25.s,  p0/m, z25.s, z0.s\n" \
  "fmax z29.s, p0/m, z29.s,  z0.s\n"                     \
  "fmax z30.s, p0/m, z30.s,  z0.s\n"                     \
  "fmax z31.s, p0/m, z31.s,  z0.s\n"                     \
  "fmul z21.s, p0/m, z21.s,  z1.s\n"                     \
  "fmul z23.s, p0/m, z23.s,  z1.s\n"                     \
  "fmul z25.s, p0/m, z25.s,  z1.s\n"                     \
  "fadd z29.s, p0/m, z29.s,  z21.s\n"                    \
  "fadd z30.s, p0/m, z30.s,  z23.s\n"                    \
  "fadd z31.s, p0/m, z31.s,  z25.s\n"                    \
  "movprfx z21, z26\n  fmin z21.s,  p0/m, z21.s, z0.s\n" \
  "movprfx z23, z4\n   fmin z23.s,  p0/m, z23.s, z0.s\n" \
  "movprfx z25, z6\n   fmin z25.s,  p0/m, z25.s, z0.s\n" \
  "fmax z26.s, p0/m, z26.s,  z0.s\n"                     \
  "fmax z4.s,  p0/m, z4.s,   z0.s\n"                     \
  "fmax z6.s,  p0/m, z6.s,   z0.s\n"                     \
  "fmul z21.s, p0/m, z21.s,  z1.s\n"                     \
  "fmul z23.s, p0/m, z23.s,  z1.s\n"                     \
  "fmul z25.s, p0/m, z25.s,  z1.s\n"                     \
  "fadd z26.s, p0/m, z26.s,  z21.s\n"                    \
  "fadd z4.s,  p0/m, z4.s,   z23.s\n"                    \
  "fadd z6.s,  p0/m, z6.s,   z25.s\n"                    \
  "movprfx z21, z27\n  fmin z21.s,  p0/m, z21.s, z0.s\n" \
  "movprfx z23, z5\n   fmin z23.s,  p0/m, z23.s, z0.s\n" \
  "movprfx z25, z7\n   fmin z25.s,  p0/m, z25.s, z0.s\n" \
  "fmax z27.s, p0/m, z27.s,  z0.s\n"                     \
  "fmax z5.s,  p0/m, z5.s,   z0.s\n"                     \
  "fmax z7.s,  p0/m, z7.s,   z0.s\n"                     \
  "fmul z21.s, p0/m, z21.s,  z1.s\n"                     \
  "fmul z23.s, p0/m, z23.s,  z1.s\n"                     \
  "fmul z25.s, p0/m, z25.s,  z1.s\n"                     \
  "fadd z27.s, p0/m, z27.s,  z21.s\n"                    \
  "fadd z5.s,  p0/m, z5.s,   z23.s\n"                    \
  "fadd z7.s,  p0/m, z7.s,   z25.s\n"                    \
  "movprfx z21, z8\n   fmin z21.s,  p0/m, z21.s, z0.s\n" \
  "movprfx z23, z10\n  fmin z23.s,  p0/m, z23.s, z0.s\n" \
  "movprfx z25, z12\n  fmin z25.s,  p0/m, z25.s, z0.s\n" \
  "fmax z8.s,  p0/m, z8.s,   z0.s\n"                     \
  "fmax z10.s, p0/m, z10.s,  z0.s\n"                     \
  "fmax z12.s, p0/m, z12.s,  z0.s\n"                     \
  "fmul z21.s, p0/m, z21.s,  z1.s\n"                     \
  "fmul z23.s, p0/m, z23.s,  z1.s\n"                     \
  "fmul z25.s, p0/m, z25.s,  z1.s\n"                     \
  "fadd z8.s,  p0/m, z8.s,   z21.s\n"                    \
  "fadd z10.s, p0/m, z10.s,  z23.s\n"                    \
  "fadd z12.s, p0/m, z12.s,  z25.s\n"                    \
  "movprfx z21, z9\n   fmin z21.s,  p0/m, z21.s, z0.s\n" \
  "movprfx z23, z11\n  fmin z23.s,  p0/m, z23.s, z0.s\n" \
  "movprfx z25, z13\n  fmin z25.s,  p0/m, z25.s, z0.s\n" \
  "fmax z9.s,  p0/m, z9.s,   z0.s\n"                     \
  "fmax z11.s, p0/m, z11.s,  z0.s\n"                     \
  "fmax z13.s, p0/m, z13.s,  z0.s\n"                     \
  "fmul z21.s, p0/m, z21.s,  z1.s\n"                     \
  "fmul z23.s, p0/m, z23.s,  z1.s\n"                     \
  "fmul z25.s, p0/m, z25.s,  z1.s\n"                     \
  "fadd z9.s,  p0/m, z9.s,   z21.s\n"                    \
  "fadd z11.s, p0/m, z11.s,  z23.s\n"                    \
  "fadd z13.s, p0/m, z13.s,  z25.s\n"                    \
  "movprfx z21, z14\n  fmin z21.s,  p0/m, z21.s, z0.s\n" \
  "movprfx z23, z16\n  fmin z23.s,  p0/m, z23.s, z0.s\n" \
  "movprfx z25, z18\n  fmin z25.s,  p0/m, z25.s, z0.s\n" \
  "fmax z14.s, p0/m, z14.s,  z0.s\n"                     \
  "fmax z16.s, p0/m, z16.s,  z0.s\n"                     \
  "fmax z18.s, p0/m, z18.s,  z0.s\n"                     \
  "fmul z21.s, p0/m, z21.s,  z1.s\n"                     \
  "fmul z23.s, p0/m, z23.s,  z1.s\n"                     \
  "fmul z25.s, p0/m, z25.s,  z1.s\n"                     \
  "fadd z14.s, p0/m, z14.s,  z21.s\n"                    \
  "fadd z16.s, p0/m, z16.s,  z23.s\n"                    \
  "fadd z18.s, p0/m, z18.s,  z25.s\n"                    \
  "movprfx z21, z15\n  fmin z21.s,  p0/m, z21.s, z0.s\n" \
  "movprfx z23, z17\n  fmin z23.s,  p0/m, z23.s, z0.s\n" \
  "movprfx z25, z19\n  fmin z25.s,  p0/m, z25.s, z0.s\n" \
  "fmax z15.s, p0/m, z15.s,  z0.s\n"                     \
  "fmax z17.s, p0/m, z17.s,  z0.s\n"                     \
  "fmax z19.s, p0/m, z19.s,  z0.s\n"                     \
  "fmul z21.s, p0/m, z21.s,  z1.s\n"                     \
  "fmul z23.s, p0/m, z23.s,  z1.s\n"                     \
  "fmul z25.s, p0/m, z25.s,  z1.s\n"                     \
  "fadd z15.s, p0/m, z15.s,  z21.s\n"                    \
  "fadd z17.s, p0/m, z17.s,  z23.s\n"                    \
  "fadd z19.s, p0/m, z19.s,  z25.s\n"                    \
  "movprfx z21, z20\n  fmin z21.s,  p0/m, z21.s, z0.s\n" \
  "movprfx z23, z22\n  fmin z23.s,  p0/m, z23.s, z0.s\n" \
  "movprfx z25, z24\n  fmin z25.s,  p0/m, z25.s, z0.s\n" \
  "fmax z20.s, p0/m, z20.s,  z0.s\n"                     \
  "fmax z22.s, p0/m, z22.s,  z0.s\n"                     \
  "fmax z24.s, p0/m, z24.s,  z0.s\n"                     \
  "fmul z21.s, p0/m, z21.s,  z1.s\n"                     \
  "fmul z23.s, p0/m, z23.s,  z1.s\n"                     \
  "fmul z25.s, p0/m, z25.s,  z1.s\n"                     \
  "fadd z20.s, p0/m, z20.s,  z21.s\n"                    \
  "fadd z22.s, p0/m, z22.s,  z23.s\n"                    \
  "fadd z24.s, p0/m, z24.s,  z25.s\n"

#define SMMLA_HARDSWIDH_8x12                             \
  "mov z0.s, #0x0    \n"                                 \
  "ld1rqw {z1.s}, p0/Z, [%x[alpha]]\n"                   \
  "ld1rqw {z2.s}, p0/Z, [%x[alpha], #0x10]\n"            \
  "ld1rqw {z3.s}, p0/Z, [%x[alpha], #0x20]\n"            \
  "movprfx z21, z29\n  fadd z21.s,  p0/m, z21.s, z1.s\n" \
  "movprfx z23, z30\n  fadd z23.s,  p0/m, z23.s, z1.s\n" \
  "movprfx z25, z31\n  fadd z25.s,  p0/m, z25.s, z1.s\n" \
  "fmul z29.s,  p0/m, z29.s,   z2.s\n"                   \
  "fmul z30.s,  p0/m, z30.s,   z2.s\n"                   \
  "fmul z31.s,  p0/m, z31.s,   z2.s\n"                   \
  "fmax z21.s,  p0/m, z21.s,   z0.s\n"                   \
  "fmax z23.s,  p0/m, z23.s,   z0.s\n"                   \
  "fmax z25.s,  p0/m, z25.s,   z0.s\n"                   \
  "fmin z21.s,  p0/m, z21.s,   z3.s\n"                   \
  "fmin z23.s,  p0/m, z23.s,   z3.s\n"                   \
  "fmin z25.s,  p0/m, z25.s,   z3.s\n"                   \
  "fmul z29.s,  p0/m, z29.s,   z21.s\n"                  \
  "fmul z30.s,  p0/m, z30.s,   z23.s\n"                  \
  "fmul z31.s,  p0/m, z31.s,   z25.s\n"                  \
  "movprfx z21, z26\n  fadd z21.s,  p0/m, z21.s, z1.s\n" \
  "movprfx z23, z4\n   fadd z23.s,  p0/m, z23.s, z1.s\n" \
  "movprfx z25, z6\n   fadd z25.s,  p0/m, z25.s, z1.s\n" \
  "fmul z26.s,  p0/m, z26.s,   z2.s\n"                   \
  "fmul z4.s,   p0/m, z4.s,    z2.s\n"                   \
  "fmul z6.s,   p0/m, z6.s,    z2.s\n"                   \
  "fmax z21.s,  p0/m, z21.s,   z0.s\n"                   \
  "fmax z23.s,  p0/m, z23.s,   z0.s\n"                   \
  "fmax z25.s,  p0/m, z25.s,   z0.s\n"                   \
  "fmin z21.s,  p0/m, z21.s,   z3.s\n"                   \
  "fmin z23.s,  p0/m, z23.s,   z3.s\n"                   \
  "fmin z25.s,  p0/m, z25.s,   z3.s\n"                   \
  "fmul z26.s,  p0/m, z26.s,   z21.s\n"                  \
  "fmul z4.s,   p0/m, z4.s,    z23.s\n"                  \
  "fmul z6.s,   p0/m, z6.s,    z25.s\n"                  \
  "movprfx z21, z27\n  fadd z21.s,  p0/m, z21.s, z1.s\n" \
  "movprfx z23, z5\n   fadd z23.s,  p0/m, z23.s, z1.s\n" \
  "movprfx z25, z7\n   fadd z25.s,  p0/m, z25.s, z1.s\n" \
  "fmul z27.s,  p0/m, z27.s,   z2.s\n"                   \
  "fmul z5.s,   p0/m, z5.s,    z2.s\n"                   \
  "fmul z7.s,   p0/m, z7.s,    z2.s\n"                   \
  "fmax z21.s,  p0/m, z21.s,   z0.s\n"                   \
  "fmax z23.s,  p0/m, z23.s,   z0.s\n"                   \
  "fmax z25.s,  p0/m, z25.s,   z0.s\n"                   \
  "fmin z21.s,  p0/m, z21.s,   z3.s\n"                   \
  "fmin z23.s,  p0/m, z23.s,   z3.s\n"                   \
  "fmin z25.s,  p0/m, z25.s,   z3.s\n"                   \
  "fmul z27.s,  p0/m, z27.s,   z21.s\n"                  \
  "fmul z5.s,   p0/m, z5.s,    z23.s\n"                  \
  "fmul z7.s,   p0/m, z7.s,    z25.s\n"                  \
  "movprfx z21, z8\n   fadd z21.s,  p0/m, z21.s, z1.s\n" \
  "movprfx z23, z10\n  fadd z23.s,  p0/m, z23.s, z1.s\n" \
  "movprfx z25, z12\n  fadd z25.s,  p0/m, z25.s, z1.s\n" \
  "fmul z8.s,   p0/m, z8.s,    z2.s\n"                   \
  "fmul z10.s,  p0/m, z10.s,   z2.s\n"                   \
  "fmul z12.s,  p0/m, z12.s,   z2.s\n"                   \
  "fmax z21.s,  p0/m, z21.s,   z0.s\n"                   \
  "fmax z23.s,  p0/m, z23.s,   z0.s\n"                   \
  "fmax z25.s,  p0/m, z25.s,   z0.s\n"                   \
  "fmin z21.s,  p0/m, z21.s,   z3.s\n"                   \
  "fmin z23.s,  p0/m, z23.s,   z3.s\n"                   \
  "fmin z25.s,  p0/m, z25.s,   z3.s\n"                   \
  "fmul z8.s,   p0/m, z8.s,    z21.s\n"                  \
  "fmul z10.s,  p0/m, z10.s,   z23.s\n"                  \
  "fmul z12.s,  p0/m, z12.s,   z25.s\n"                  \
  "movprfx z21, z9\n   fadd z21.s,  p0/m, z21.s, z1.s\n" \
  "movprfx z23, z11\n  fadd z23.s,  p0/m, z23.s, z1.s\n" \
  "movprfx z25, z13\n  fadd z25.s,  p0/m, z25.s, z1.s\n" \
  "fmul z9.s,   p0/m, z9.s,    z2.s\n"                   \
  "fmul z11.s,  p0/m, z11.s,   z2.s\n"                   \
  "fmul z13.s,  p0/m, z13.s,   z2.s\n"                   \
  "fmax z21.s,  p0/m, z21.s,   z0.s\n"                   \
  "fmax z23.s,  p0/m, z23.s,   z0.s\n"                   \
  "fmax z25.s,  p0/m, z25.s,   z0.s\n"                   \
  "fmin z21.s,  p0/m, z21.s,   z3.s\n"                   \
  "fmin z23.s,  p0/m, z23.s,   z3.s\n"                   \
  "fmin z25.s,  p0/m, z25.s,   z3.s\n"                   \
  "fmul z9.s,   p0/m, z9.s,    z21.s\n"                  \
  "fmul z11.s,  p0/m, z11.s,   z23.s\n"                  \
  "fmul z13.s,  p0/m, z13.s,   z25.s\n"                  \
  "movprfx z21, z14\n  fadd z21.s,  p0/m, z21.s, z1.s\n" \
  "movprfx z23, z16\n  fadd z23.s,  p0/m, z23.s, z1.s\n" \
  "movprfx z25, z18\n  fadd z25.s,  p0/m, z25.s, z1.s\n" \
  "fmul z14.s,  p0/m, z14.s,   z2.s\n"                   \
  "fmul z16.s,  p0/m, z16.s,   z2.s\n"                   \
  "fmul z18.s,  p0/m, z18.s,   z2.s\n"                   \
  "fmax z21.s,  p0/m, z21.s,   z0.s\n"                   \
  "fmax z23.s,  p0/m, z23.s,   z0.s\n"                   \
  "fmax z25.s,  p0/m, z25.s,   z0.s\n"                   \
  "fmin z21.s,  p0/m, z21.s,   z3.s\n"                   \
  "fmin z23.s,  p0/m, z23.s,   z3.s\n"                   \
  "fmin z25.s,  p0/m, z25.s,   z3.s\n"                   \
  "fmul z14.s,  p0/m, z14.s,   z21.s\n"                  \
  "fmul z16.s,  p0/m, z16.s,   z23.s\n"                  \
  "fmul z18.s,  p0/m, z18.s,   z25.s\n"                  \
  "movprfx z21, z15\n  fadd z21.s,  p0/m, z21.s, z1.s\n" \
  "movprfx z23, z17\n  fadd z23.s,  p0/m, z23.s, z1.s\n" \
  "movprfx z25, z19\n  fadd z25.s,  p0/m, z25.s, z1.s\n" \
  "fmul z15.s,  p0/m, z15.s,   z2.s\n"                   \
  "fmul z17.s,  p0/m, z17.s,   z2.s\n"                   \
  "fmul z19.s,  p0/m, z19.s,   z2.s\n"                   \
  "fmax z21.s,  p0/m, z21.s,   z0.s\n"                   \
  "fmax z23.s,  p0/m, z23.s,   z0.s\n"                   \
  "fmax z25.s,  p0/m, z25.s,   z0.s\n"                   \
  "fmin z21.s,  p0/m, z21.s,   z3.s\n"                   \
  "fmin z23.s,  p0/m, z23.s,   z3.s\n"                   \
  "fmin z25.s,  p0/m, z25.s,   z3.s\n"                   \
  "fmul z15.s,  p0/m, z15.s,   z21.s\n"                  \
  "fmul z17.s,  p0/m, z17.s,   z23.s\n"                  \
  "fmul z19.s,  p0/m, z19.s,   z25.s\n"                  \
  "movprfx z21, z20\n  fadd z21.s,  p0/m, z21.s, z1.s\n" \
  "movprfx z23, z22\n  fadd z23.s,  p0/m, z23.s, z1.s\n" \
  "movprfx z25, z24\n  fadd z25.s,  p0/m, z25.s, z1.s\n" \
  "fmul z20.s,  p0/m, z20.s,   z2.s\n"                   \
  "fmul z22.s,  p0/m, z22.s,   z2.s\n"                   \
  "fmul z24.s,  p0/m, z24.s,   z2.s\n"                   \
  "fmax z21.s,  p0/m, z21.s,   z0.s\n"                   \
  "fmax z23.s,  p0/m, z23.s,   z0.s\n"                   \
  "fmax z25.s,  p0/m, z25.s,   z0.s\n"                   \
  "fmin z21.s,  p0/m, z21.s,   z3.s\n"                   \
  "fmin z23.s,  p0/m, z23.s,   z3.s\n"                   \
  "fmin z25.s,  p0/m, z25.s,   z3.s\n"                   \
  "fmul z20.s,  p0/m, z20.s,   z21.s\n"                  \
  "fmul z22.s,  p0/m, z22.s,   z23.s\n"                  \
  "fmul z24.s,  p0/m, z24.s,   z25.s\n"

#define SMMLA_ACT_PROCESS_8x12 \
  "cmp %x[flag_act], #1\n"     \
  "beq 3f\n"     \
  "cmp %x[flag_act], #0\n"     \
  "beq 10f\n"     \
  "cmp %x[flag_act], #2\n"     \
  "beq 4f\n"     \
  "cmp %x[flag_act], #3\n"     \
  "beq 5f\n"     \
  SMMLA_HARDSWIDH_8x12 \
  "3: \n"        \
  "mov z0.s, #0x0    \n" \
  SMMLA_RELU_8x12(fmax) \
  "b 10f\n"      \
  "4: \n"        \
  "mov z0.s, #0x0    \n" \
  SMMLA_RELU_8x12(fmax) \
  "ld1rqw {z0.s}, p0/Z, [%x[alpha]]\n"                   \
  SMMLA_RELU_8x12(fmin) \
  "b 10f\n"     \
  "5: \n"        \
  SMMLA_LEAKYRELU_8x12 \
  "b 10f\n"     \
  "10: \n"

#define SMMLA_STORE_FP32_8x12                     \
  "st1w {z29.s},  p0, [%x[c_ptr0]]\n"             \
  "st1w {z26.s},  p0, [%x[c_ptr1]]\n"             \
  "st1w {z27.s},  p0, [%x[c_ptr2]]\n"             \
  "st1w {z8.s},   p0, [%x[c_ptr3]]\n"             \
  "st1w {z9.s},   p0, [%x[c_ptr4]]\n"             \
  "st1w {z14.s},  p0, [%x[c_ptr5]]\n"             \
  "st1w {z15.s},  p0, [%x[c_ptr6]]\n"             \
  "st1w {z20.s},  p0, [%x[c_ptr7]]\n"             \
  "st1w {z30.s},  p0, [%x[c_ptr0], #1, MUL VL]\n" \
  "st1w {z4.s},   p0, [%x[c_ptr1], #1, MUL VL]\n" \
  "st1w {z5.s},   p0, [%x[c_ptr2], #1, MUL VL]\n" \
  "st1w {z10.s},  p0, [%x[c_ptr3], #1, MUL VL]\n" \
  "st1w {z11.s},  p0, [%x[c_ptr4], #1, MUL VL]\n" \
  "st1w {z16.s},  p0, [%x[c_ptr5], #1, MUL VL]\n" \
  "st1w {z17.s},  p0, [%x[c_ptr6], #1, MUL VL]\n" \
  "st1w {z22.s},  p0, [%x[c_ptr7], #1, MUL VL]\n" \
  "st1w {z31.s},  p0, [%x[c_ptr0], #2, MUL VL]\n" \
  "st1w {z6.s},   p0, [%x[c_ptr1], #2, MUL VL]\n" \
  "st1w {z7.s},   p0, [%x[c_ptr2], #2, MUL VL]\n" \
  "st1w {z12.s},  p0, [%x[c_ptr3], #2, MUL VL]\n" \
  "st1w {z13.s},  p0, [%x[c_ptr4], #2, MUL VL]\n" \
  "st1w {z18.s},  p0, [%x[c_ptr5], #2, MUL VL]\n" \
  "st1w {z19.s},  p0, [%x[c_ptr6], #2, MUL VL]\n" \
  "st1w {z24.s},  p0, [%x[c_ptr7], #2, MUL VL]\n" \
  "addvl %x[c_ptr0], %x[c_ptr0], #3\n"            \
  "addvl %x[c_ptr1], %x[c_ptr1], #3\n"            \
  "addvl %x[c_ptr2], %x[c_ptr2], #3\n"            \
  "addvl %x[c_ptr3], %x[c_ptr3], #3\n"            \
  "addvl %x[c_ptr4], %x[c_ptr4], #3\n"            \
  "addvl %x[c_ptr5], %x[c_ptr5], #3\n"            \
  "addvl %x[c_ptr6], %x[c_ptr6], #3\n"            \
  "addvl %x[c_ptr7], %x[c_ptr7], #3\n"

#define SMMLA_STORE_INT8_ASM(a, b, c)                           \
  "fcmge v21.4s,  v" #a ".4s,  v2.4s       \n" \
  "fcmge v23.4s,  v" #b ".4s,  v2.4s       \n" \
  "fcmge v25.4s,  v" #c ".4s,  v2.4s       \n" \
  "bif   v" #a ".16b, v2.16b, v21.16b      \n" \
  "bif   v" #b ".16b, v2.16b, v23.16b      \n" \
  "bif   v" #c ".16b, v2.16b, v25.16b      \n" \
  /* fp32 -> int32 */                          \
  "fcvtas  v21.4s,  v" #a ".4s             \n" \
  "fcvtas  v23.4s,  v" #b ".4s             \n" \
  "fcvtas  v25.4s,  v" #c ".4s             \n" \
  /* int32 -> int16 */                        \
  "sqxtn   v0.4h,  v21.4s                  \n" \
  "sqxtn2  v0.8h,  v23.4s                  \n" \
  "sqxtn   v1.4h,  v25.4s                  \n" \
  /* int16 -> int8  */                         \
  "sqxtn   v" #c ".8b,  v0.8h              \n" \
  "sqxtn   v" #b ".8b,  v1.8h           \n"
// clang-format on

#define SMMLA_STORE_INT8_8x12 \
  "ptrue p0.b\n" \
  "ld1rqw {z2.s}, p0/Z, [%x[vmax]]\n" \
  SMMLA_STORE_INT8_ASM(29, 30, 31) \
  SMMLA_STORE_INT8_ASM(26, 4,  6) \
  SMMLA_STORE_INT8_ASM(27, 5,  7) \
  SMMLA_STORE_INT8_ASM(8,  10, 12) \
  SMMLA_STORE_INT8_ASM(9,  11, 13) \
  SMMLA_STORE_INT8_ASM(14, 16, 18) \
  SMMLA_STORE_INT8_ASM(15, 17, 19) \
  SMMLA_STORE_INT8_ASM(20, 22, 24) \
  "str d31, [%[c_ptr0]], #0x08\n"  \
  "str d6,  [%[c_ptr1]], #0x08\n"  \
  "str d7,  [%[c_ptr2]], #0x08\n"  \
  "str d12, [%[c_ptr3]], #0x08\n"  \
  "str d13, [%[c_ptr4]], #0x08\n"  \
  "str d18, [%[c_ptr5]], #0x08\n"  \
  "str d19, [%[c_ptr6]], #0x08\n"  \
  "str d24, [%[c_ptr7]], #0x08\n"  \
  "str s30, [%[c_ptr0]], #4\n"           \
  "str s4,  [%[c_ptr1]], #4\n"           \
  "str s5,  [%[c_ptr2]], #4\n"           \
  "str s10, [%[c_ptr3]], #4\n"           \
  "str s11, [%[c_ptr4]], #4\n"           \
  "str s16, [%[c_ptr5]], #4\n"           \
  "str s17, [%[c_ptr6]], #4\n"           \
  "str s22, [%[c_ptr7]], #4\n"

#define SMMLA_STORE_INT8_8x12_1                         \
  "mov z0.s, #0x0\n"                                    \
  "ld1rqw {z1.s}, p0/Z, [%x[alpha], #0x30]\n"           \
  "mov z2.s, #127\n"                                    \
  "movprfx z21, z29\n fmin z21.s, p0/m, z21.s, z0.s\n"  \
  "movprfx z23, z30\n fmin z23.s, p0/m, z23.s, z0.s\n"  \
  "movprfx z25, z31\n fmin z25.s, p0/m, z25.s, z0.s\n"  \
  "fmax z29.s, p0/m, z29.s, z0.s\n"                     \
  "fmax z30.s, p0/m, z30.s, z0.s\n"                     \
  "fmax z31.s, p0/m, z31.s, z0.s\n" /* a + 0.5 */       \
  "fadd z29.s, p0/m, z29.s, z1.s\n"                     \
  "fadd z30.s, p0/m, z30.s, z1.s\n"                     \
  "fadd z31.s, p0/m, z31.s, z1.s\n" /* a - 0.5 */       \
  "fsub z21.s, p0/m, z21.s, z1.s\n"                     \
  "fsub z23.s, p0/m, z23.s, z1.s\n"                     \
  "fsub z25.s, p0/m, z25.s, z1.s\n" /* fp32->int32 */   \
  "fcvtzs z29.s, p0/m, z29.s\n"                         \
  "fcvtzs z30.s, p0/m, z30.s\n"                         \
  "fcvtzs z31.s, p0/m, z31.s\n"                         \
  "fcvtzs z21.s, p0/m, z21.s\n"                         \
  "fcvtzs z23.s, p0/m, z23.s\n"                         \
  "fcvtzs z25.s, p0/m, z25.s\n"                         \
  "add z29.s, p0/m, z29.s, z21.s\n"                     \
  "add z30.s, p0/m, z30.s, z23.s\n"                     \
  "add z31.s, p0/m, z31.s, z25.s\n" /* a + 127 */       \
  "add z29.s, p0/m, z29.s, z2.s\n"                      \
  "add z30.s, p0/m, z30.s, z2.s\n"                      \
  "add z31.s, p0/m, z31.s, z2.s\n" /* a  + 127 > 0 */   \
  "smax z29.s, p0/m, z29.s, z0.s\n"                     \
  "smax z30.s, p0/m, z30.s, z0.s\n"                     \
  "smax z31.s, p0/m, z31.s, z0.s\n" /* a - 127 */       \
  "sub z29.s, p0/m, z29.s, z2.s\n"                      \
  "sub z30.s, p0/m, z30.s, z2.s\n"                      \
  "sub z31.s, p0/m, z31.s, z2.s\n"                      \
  "mov x0, #14\n"                                       \
  "mov x1, #16\n"                                       \
  "whilelt p1.b, x0, x1\n" /* int32-> int8*/            \
  "uzp1 z21.h, z29.h, z30.h\n"                          \
  "uzp1 z23.h, z31.h, z31.h\n"                          \
  "uzp1 z29.b, z21.b, z21.b\n"                          \
  "uzp1 z30.b, z23.b, z23.b\n"                          \
  "movprfx z21, z26\n fmin z21.s, p0/m, z21.s, z0.s\n"  \
  "movprfx z23, z4\n  fmin z23.s, p0/m, z23.s, z0.s\n"  \
  "movprfx z25, z6\n  fmin z25.s, p0/m, z25.s, z0.s\n"  \
  "uzp1 z31.d, z29.d, z30.d\n"                          \
  "fmax z26.s, p0/m, z26.s, z0.s\n"                     \
  "fmax z4.s,  p0/m, z4.s,  z0.s\n"                     \
  "fmax z6.s,  p0/m, z6.s,  z0.s\n" /* a + 0.5 */       \
  "fadd z26.s, p0/m, z26.s, z1.s\n"                     \
  "fadd z4.s,  p0/m, z4.s,  z1.s\n"                     \
  "fadd z6.s,  p0/m, z6.s,  z1.s\n" /* a - 0.5 */       \
  "fsub z21.s, p0/m, z21.s, z1.s\n"                     \
  "fsub z23.s, p0/m, z23.s, z1.s\n"                     \
  "fsub z25.s, p0/m, z25.s, z1.s\n" /* fp32->int32 */   \
  "fcvtzs z26.s, p0/m, z26.s\n"                         \
  "fcvtzs z4.s,  p0/m, z4.s\n"                          \
  "fcvtzs z6.s,  p0/m, z6.s\n"                          \
  "fcvtzs z21.s, p0/m, z21.s\n"                         \
  "fcvtzs z23.s, p0/m, z23.s\n"                         \
  "fcvtzs z25.s, p0/m, z25.s\n"                         \
  "add z26.s, p0/m, z26.s, z21.s\n"                     \
  "add z4.s,  p0/m, z4.s,  z23.s\n"                     \
  "add z6.s,  p0/m, z6.s,  z25.s\n" /* a + 127 */       \
  "add z26.s, p0/m, z26.s, z2.s\n"                      \
  "add z4.s,  p0/m, z4.s,  z2.s\n"                      \
  "add z6.s,  p0/m, z6.s,  z2.s\n" /* a  + 127 > 0 */   \
  "smax z26.s, p0/m, z26.s, z0.s\n"                     \
  "smax z4.s,  p0/m, z4.s,  z0.s\n"                     \
  "smax z6.s,  p0/m, z6.s,  z0.s\n" /* a - 127 */       \
  "sub z26.s, p0/m, z26.s, z2.s\n"                      \
  "sub z4.s,  p0/m, z4.s,  z2.s\n"                      \
  "sub z6.s,  p0/m, z6.s,  z2.s\n" /* int32-> int8*/    \
  "uzp1 z21.h, z26.h, z4.h\n"                           \
  "uzp1 z23.h, z6.h,  z6.h\n"                           \
  "uzp1 z26.b, z21.b, z21.b\n"                          \
  "uzp1 z4.b,  z23.b, z23.b\n"                          \
  "movprfx z21, z27\n fmin z21.s, p0/m, z21.s, z0.s\n"  \
  "movprfx z23, z5\n  fmin z23.s, p0/m, z23.s, z0.s\n"  \
  "movprfx z25, z7\n  fmin z25.s, p0/m, z25.s, z0.s\n"  \
  "uzp1 z6.d,  z26.d, z4.d\n"                           \
  "fmax z27.s, p0/m, z27.s, z0.s\n"                     \
  "fmax z5.s,  p0/m, z5.s,  z0.s\n"                     \
  "fmax z7.s,  p0/m, z7.s,  z0.s\n" /* a + 0.5 */       \
  "fadd z27.s, p0/m, z27.s, z1.s\n"                     \
  "fadd z5.s,  p0/m, z5.s,  z1.s\n"                     \
  "fadd z7.s,  p0/m, z7.s,  z1.s\n" /* a - 0.5 */       \
  "fsub z21.s, p0/m, z21.s, z1.s\n"                     \
  "fsub z23.s, p0/m, z23.s, z1.s\n"                     \
  "fsub z25.s, p0/m, z25.s, z1.s\n" /* fp32->int32 */   \
  "fcvtzs z27.s, p0/m, z27.s\n"                         \
  "fcvtzs z5.s,  p0/m, z5.s\n"                          \
  "fcvtzs z7.s,  p0/m, z7.s\n"                          \
  "fcvtzs z21.s, p0/m, z21.s\n"                         \
  "fcvtzs z23.s, p0/m, z23.s\n"                         \
  "fcvtzs z25.s, p0/m, z25.s\n"                         \
  "add z27.s, p0/m, z27.s, z21.s\n"                     \
  "add z5.s,  p0/m, z5.s,  z23.s\n"                     \
  "add z7.s,  p0/m, z7.s,  z25.s\n" /* a + 127 */       \
  "add z27.s, p0/m, z27.s, z2.s\n"                      \
  "add z5.s,  p0/m, z5.s,  z2.s\n"                      \
  "add z7.s,  p0/m, z7.s,  z2.s\n" /* a  + 127 > 0 */   \
  "smax z27.s, p0/m, z27.s, z0.s\n"                     \
  "smax z5.s,  p0/m, z5.s,  z0.s\n"                     \
  "smax z7.s,  p0/m, z7.s,  z0.s\n" /* a - 127 */       \
  "sub z27.s, p0/m, z27.s, z2.s\n"                      \
  "sub z5.s,  p0/m, z5.s,  z2.s\n"                      \
  "sub z7.s,  p0/m, z7.s,  z2.s\n" /* int32-> int8*/    \
  "uzp1 z21.h, z27.h, z5.h\n"                           \
  "uzp1 z23.h, z7.h,  z7.h\n"                           \
  "uzp1 z27.b, z21.b, z21.b\n"                          \
  "uzp1 z5.b,  z23.b, z23.b\n"                          \
  "movprfx z21, z8\n   fmin z21.s, p0/m, z21.s, z0.s\n" \
  "movprfx z23, z10\n  fmin z23.s, p0/m, z23.s, z0.s\n" \
  "movprfx z25, z12\n  fmin z25.s, p0/m, z25.s, z0.s\n" \
  "uzp1 z7.d,  z27.d, z5.d\n"                           \
  "fmax z8.s,  p0/m, z8.s,  z0.s\n"                     \
  "fmax z10.s, p0/m, z10.s, z0.s\n"                     \
  "fmax z12.s, p0/m, z12.s, z0.s\n" /* a + 0.5 */       \
  "fadd z8.s,  p0/m, z8.s,  z1.s\n"                     \
  "fadd z10.s, p0/m, z10.s, z1.s\n"                     \
  "fadd z12.s, p0/m, z12.s, z1.s\n" /* a - 0.5 */       \
  "fsub z21.s, p0/m, z21.s, z1.s\n"                     \
  "fsub z23.s, p0/m, z23.s, z1.s\n"                     \
  "fsub z25.s, p0/m, z25.s, z1.s\n" /* fp32->int32 */   \
  "fcvtzs z8.s,  p0/m, z8.s\n"                          \
  "fcvtzs z10.s, p0/m, z10.s\n"                         \
  "fcvtzs z12.s, p0/m, z12.s\n"                         \
  "fcvtzs z21.s, p0/m, z21.s\n"                         \
  "fcvtzs z23.s, p0/m, z23.s\n"                         \
  "fcvtzs z25.s, p0/m, z25.s\n"                         \
  "add z8.s,  p0/m,  z8.s, z21.s\n"                     \
  "add z10.s, p0/m, z10.s,  z23.s\n"                    \
  "add z12.s, p0/m, z12.s,  z25.s\n" /* a + 127 */      \
  "add z8.s,  p0/m, z8.s,  z2.s\n"                      \
  "add z10.s, p0/m, z10.s, z2.s\n"                      \
  "add z12.s, p0/m, z12.s, z2.s\n" /* a  + 127 > 0 */   \
  "smax z8.s,  p0/m, z8.s,  z0.s\n"                     \
  "smax z10.s, p0/m, z10.s, z0.s\n"                     \
  "smax z12.s, p0/m, z12.s, z0.s\n" /* a - 127 */       \
  "sub z8.s,  p0/m, z8.s,  z2.s\n"                      \
  "sub z10.s, p0/m, z10.s, z2.s\n"                      \
  "sub z12.s, p0/m, z12.s, z2.s\n" /* int32-> int8*/    \
  "uzp1 z21.h, z8.h,  z10.h\n"                          \
  "uzp1 z23.h, z12.h, z12.h\n"                          \
  "uzp1 z8.b,  z21.b, z21.b\n"                          \
  "uzp1 z10.b, z23.b, z23.b\n"                          \
  "movprfx z21, z9\n   fmin z21.s, p0/m, z21.s, z0.s\n" \
  "movprfx z23, z11\n  fmin z23.s, p0/m, z23.s, z0.s\n" \
  "movprfx z25, z13\n  fmin z25.s, p0/m, z25.s, z0.s\n" \
  "uzp1 z12.d, z8.d,  z10.d\n"                          \
  "fmax z9.s,  p0/m, z9.s,  z0.s\n"                     \
  "fmax z11.s, p0/m, z11.s, z0.s\n"                     \
  "fmax z13.s, p0/m, z13.s, z0.s\n" /* a + 0.5 */       \
  "fadd z9.s,  p0/m, z9.s,  z1.s\n"                     \
  "fadd z11.s, p0/m, z11.s, z1.s\n"                     \
  "fadd z13.s, p0/m, z13.s, z1.s\n" /* a - 0.5 */       \
  "fsub z21.s, p0/m, z21.s, z1.s\n"                     \
  "fsub z23.s, p0/m, z23.s, z1.s\n"                     \
  "fsub z25.s, p0/m, z25.s, z1.s\n" /* fp32->int32 */   \
  "fcvtzs z9.s,  p0/m, z9.s\n"                          \
  "fcvtzs z11.s, p0/m, z11.s\n"                         \
  "fcvtzs z13.s, p0/m, z13.s\n"                         \
  "fcvtzs z21.s, p0/m, z21.s\n"                         \
  "fcvtzs z23.s, p0/m, z23.s\n"                         \
  "fcvtzs z25.s, p0/m, z25.s\n"                         \
  "add z9.s,  p0/m, z9.s,  z21.s\n"                     \
  "add z11.s, p0/m, z11.s, z23.s\n"                     \
  "add z13.s, p0/m, z13.s, z25.s\n" /* a + 127 */       \
  "add z9.s,  p0/m, z9.s,  z2.s\n"                      \
  "add z11.s, p0/m, z11.s, z2.s\n"                      \
  "add z13.s, p0/m, z13.s, z2.s\n" /* a  + 127 > 0 */   \
  "smax z9.s,  p0/m, z9.s,  z0.s\n"                     \
  "smax z11.s, p0/m, z11.s, z0.s\n"                     \
  "smax z13.s, p0/m, z13.s, z0.s\n" /* a - 127 */       \
  "sub z9.s,  p0/m, z9.s,  z2.s\n"                      \
  "sub z11.s, p0/m, z11.s, z2.s\n"                      \
  "sub z13.s, p0/m, z13.s, z2.s\n" /* int32-> int8*/    \
  "uzp1 z21.h, z9.h,  z11.h\n"                          \
  "uzp1 z23.h, z13.h, z13.h\n"                          \
  "uzp1 z9.b,  z21.b, z21.b\n"                          \
  "uzp1 z11.b, z23.b, z23.b\n"                          \
  "movprfx z21, z14\n  fmin z21.s, p0/m, z21.s, z0.s\n" \
  "movprfx z23, z16\n  fmin z23.s, p0/m, z23.s, z0.s\n" \
  "movprfx z25, z18\n  fmin z25.s, p0/m, z25.s, z0.s\n" \
  "uzp1 z13.d, z9.d,  z11.d\n"                          \
  "fmax z14.s, p0/m, z14.s, z0.s\n"                     \
  "fmax z16.s, p0/m, z16.s, z0.s\n"                     \
  "fmax z18.s, p0/m, z18.s, z0.s\n" /* a + 0.5 */       \
  "fadd z14.s, p0/m, z14.s,  z1.s\n"                    \
  "fadd z16.s, p0/m, z16.s, z1.s\n"                     \
  "fadd z18.s, p0/m, z18.s, z1.s\n" /* a - 0.5 */       \
  "fsub z21.s, p0/m, z21.s, z1.s\n"                     \
  "fsub z23.s, p0/m, z23.s, z1.s\n"                     \
  "fsub z25.s, p0/m, z25.s, z1.s\n" /* fp32->int32 */   \
  "fcvtzs z14.s, p0/m, z14.s\n"                         \
  "fcvtzs z16.s, p0/m, z16.s\n"                         \
  "fcvtzs z18.s, p0/m, z18.s\n"                         \
  "fcvtzs z21.s, p0/m, z21.s\n"                         \
  "fcvtzs z23.s, p0/m, z23.s\n"                         \
  "fcvtzs z25.s, p0/m, z25.s\n"                         \
  "add z14.s, p0/m, z14.s, z21.s\n"                     \
  "add z16.s, p0/m, z16.s, z23.s\n"                     \
  "add z18.s, p0/m, z18.s, z25.s\n" /* a + 127 */       \
  "add z14.s, p0/m, z14.s, z2.s\n"                      \
  "add z16.s, p0/m, z16.s, z2.s\n"                      \
  "add z18.s, p0/m, z18.s, z2.s\n" /* a  + 127 > 0 */   \
  "smax z14.s, p0/m, z14.s, z0.s\n"                     \
  "smax z16.s, p0/m, z16.s, z0.s\n"                     \
  "smax z18.s, p0/m, z18.s, z0.s\n" /* a - 127 */       \
  "sub z14.s, p0/m, z14.s, z2.s\n"                      \
  "sub z16.s, p0/m, z16.s, z2.s\n"                      \
  "sub z18.s, p0/m, z18.s, z2.s\n" /* int32-> int8*/    \
  "uzp1 z21.h, z14.h, z16.h\n"                          \
  "uzp1 z23.h, z18.h, z18.h\n"                          \
  "uzp1 z14.b, z21.b, z21.b\n"                          \
  "uzp1 z16.b, z23.b, z23.b\n"                          \
  "movprfx z21, z15\n  fmin z21.s, p0/m, z21.s, z0.s\n" \
  "movprfx z23, z17\n  fmin z23.s, p0/m, z23.s, z0.s\n" \
  "movprfx z25, z19\n  fmin z25.s, p0/m, z25.s, z0.s\n" \
  "uzp1 z18.d, z14.d,  z16.d\n"                         \
  "fmax z15.s, p0/m, z15.s, z0.s\n"                     \
  "fmax z17.s, p0/m, z17.s, z0.s\n"                     \
  "fmax z19.s, p0/m, z19.s, z0.s\n" /* a + 0.5 */       \
  "fadd z15.s, p0/m, z15.s,  z1.s\n"                    \
  "fadd z17.s, p0/m, z17.s, z1.s\n"                     \
  "fadd z19.s, p0/m, z19.s, z1.s\n" /* a - 0.5 */       \
  "fsub z21.s, p0/m, z21.s, z1.s\n"                     \
  "fsub z23.s, p0/m, z23.s, z1.s\n"                     \
  "fsub z25.s, p0/m, z25.s, z1.s\n" /* fp32->int32 */   \
  "fcvtzs z15.s, p0/m, z15.s\n"                         \
  "fcvtzs z17.s, p0/m, z17.s\n"                         \
  "fcvtzs z19.s, p0/m, z19.s\n"                         \
  "fcvtzs z21.s, p0/m, z21.s\n"                         \
  "fcvtzs z23.s, p0/m, z23.s\n"                         \
  "fcvtzs z25.s, p0/m, z25.s\n"                         \
  "add z15.s, p0/m, z15.s, z21.s\n"                     \
  "add z17.s, p0/m, z17.s, z23.s\n"                     \
  "add z19.s, p0/m, z19.s, z25.s\n" /* a + 127 */       \
  "add z15.s, p0/m, z15.s, z2.s\n"                      \
  "add z17.s, p0/m, z17.s, z2.s\n"                      \
  "add z19.s, p0/m, z19.s, z2.s\n" /* a  + 127 > 0 */   \
  "smax z15.s, p0/m, z15.s, z0.s\n"                     \
  "smax z17.s, p0/m, z17.s, z0.s\n"                     \
  "smax z19.s, p0/m, z19.s, z0.s\n" /* a - 127 */       \
  "sub z15.s, p0/m, z15.s, z2.s\n"                      \
  "sub z17.s, p0/m, z17.s, z2.s\n"                      \
  "sub z19.s, p0/m, z19.s, z2.s\n" /* int32-> int8*/    \
  "uzp1 z21.h, z15.h, z17.h\n"                          \
  "uzp1 z23.h, z19.h, z19.h\n"                          \
  "uzp1 z15.b, z21.b, z21.b\n"                          \
  "uzp1 z17.b, z23.b, z23.b\n"                          \
  "movprfx z21, z20\n  fmin z21.s, p0/m, z21.s, z0.s\n" \
  "movprfx z23, z22\n  fmin z23.s, p0/m, z23.s, z0.s\n" \
  "movprfx z25, z24\n  fmin z25.s, p0/m, z25.s, z0.s\n" \
  "uzp1 z19.d, z15.d,  z17.d\n"                         \
  "fmax z20.s, p0/m, z20.s, z0.s\n"                     \
  "fmax z22.s, p0/m, z22.s, z0.s\n"                     \
  "fmax z24.s, p0/m, z24.s, z0.s\n" /* a + 0.5 */       \
  "fadd z20.s, p0/m, z20.s,  z1.s\n"                    \
  "fadd z22.s, p0/m, z22.s, z1.s\n"                     \
  "fadd z24.s, p0/m, z24.s, z1.s\n" /* a - 0.5 */       \
  "fsub z21.s, p0/m, z21.s, z1.s\n"                     \
  "fsub z23.s, p0/m, z23.s, z1.s\n"                     \
  "fsub z25.s, p0/m, z25.s, z1.s\n" /* fp32->int32 */   \
  "fcvtzs z20.s, p0/m, z20.s\n"                         \
  "fcvtzs z22.s, p0/m, z22.s\n"                         \
  "fcvtzs z24.s, p0/m, z24.s\n"                         \
  "fcvtzs z21.s, p0/m, z21.s\n"                         \
  "fcvtzs z23.s, p0/m, z23.s\n"                         \
  "fcvtzs z25.s, p0/m, z25.s\n"                         \
  "add z20.s, p0/m, z20.s, z21.s\n"                     \
  "add z22.s, p0/m, z22.s, z23.s\n"                     \
  "add z24.s, p0/m, z24.s, z25.s\n" /* a + 127 */       \
  "add z20.s, p0/m, z20.s, z2.s\n"                      \
  "add z22.s, p0/m, z22.s, z2.s\n"                      \
  "add z24.s, p0/m, z24.s, z2.s\n" /* a  + 127 > 0 */   \
  "smax z20.s, p0/m, z20.s, z0.s\n"                     \
  "smax z22.s, p0/m, z22.s, z0.s\n"                     \
  "smax z24.s, p0/m, z24.s, z0.s\n" /* a - 127 */       \
  "sub z20.s, p0/m, z20.s, z2.s\n"                      \
  "sub z22.s, p0/m, z22.s, z2.s\n"                      \
  "sub z24.s, p0/m, z24.s, z2.s\n" /* int32-> int8*/    \
  "uzp1 z21.h, z20.h, z22.h\n"                          \
  "uzp1 z23.h, z24.h, z24.h\n"                          \
  "uzp1 z20.b, z21.b, z21.b\n"                          \
  "uzp1 z22.b, z23.b, z23.b\n"                          \
  "uzp1 z24.d, z20.d, z22.d\n"                          \
  "st1b {z31.b},  p1, [%x[c_ptr0]]\n"                   \
  "st1b {z6.b},   p1, [%x[c_ptr1]]\n"                   \
  "st1b {z7.b},   p1, [%x[c_ptr2]]\n"                   \
  "st1b {z12.b},  p1, [%x[c_ptr3]]\n"                   \
  "st1b {z13.b},  p1, [%x[c_ptr4]]\n"                   \
  "st1b {z18.b},  p1, [%x[c_ptr5]]\n"                   \
  "st1b {z19.b},  p1, [%x[c_ptr6]]\n"                   \
  "st1b {z24.b},  p1, [%x[c_ptr7]]\n"                   \
  "add %x[c_ptr0], %x[c_ptr0], #0x0c\n"                 \
  "add %x[c_ptr1], %x[c_ptr1], #0x0c\n"                 \
  "add %x[c_ptr2], %x[c_ptr2], #0x0c\n"                 \
  "add %x[c_ptr3], %x[c_ptr3], #0x0c\n"                 \
  "add %x[c_ptr4], %x[c_ptr4], #0x0c\n"                 \
  "add %x[c_ptr5], %x[c_ptr5], #0x0c\n"                 \
  "add %x[c_ptr6], %x[c_ptr6], #0x0c\n"                 \
  "add %x[c_ptr7], %x[c_ptr7], #0x0c\n"

#define ASM_PARAMS_FP32                                                   \
  : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [k] "+r"(k), \
    [c_ptr0] "+r"(c_ptr0), [c_ptr1] "+r"(c_ptr1), [c_ptr2] "+r"(c_ptr2), \
    [c_ptr3] "+r"(c_ptr3), [c_ptr4] "+r"(c_ptr4), [c_ptr5] "+r"(c_ptr5), \
    [c_ptr6] "+r"(c_ptr6), [c_ptr7] "+r"(c_ptr7) \
  : [bias] "r"(bias), [alpha] "r"(alpha), [scale] "r"(scale), \
    [flag_act] "r"(flag_act), [rem_cnt] "r"(tail), [last] "r"(last) \
  : "cc", "memory", "x0", "x1", "p0", "p1", "z0", "z1", "z2", "z3", "z4", \
    "z5", "z6", "z7", \
    "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", \
    "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", \
    "z27", "z28", "z29", "z30", "z31"

#define ASM_PARAMS_FP32_INT8                                              \
  : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [k] "+r"(k), \
    [c_ptr0] "+r"(c_ptr0), [c_ptr1] "+r"(c_ptr1), [c_ptr2] "+r"(c_ptr2), \
    [c_ptr3] "+r"(c_ptr3), [c_ptr4] "+r"(c_ptr4), [c_ptr5] "+r"(c_ptr5), \
    [c_ptr6] "+r"(c_ptr6), [c_ptr7] "+r"(c_ptr7) \
  : [bias] "r"(bias), [alpha] "r"(alpha), [scale] "r"(scale), \
    [flag_act] "r"(flag_act), [rem_cnt] "r"(tail), [last] "r"(last), \
    [vmax] "r"(vmax) \
  : "cc", "memory", "x0", "x1", "p0", "p1", "z0", "z1", "z2", "z3", "z4", \
    "z5", "z6", "z7", \
    "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", \
    "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", \
    "z27", "z28", "z29", "z30", "z31"

#define ASM_PARAMS_INT8                                                   \
  : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [k] "+r"(k)  \
  : [c_ptr0] "r"(out0), [c_ptr1] "r"(out1), [c_ptr2] "r"(out2), \
    [c_ptr3] "r"(out3), [c_ptr4] "r"(out4), [c_ptr5] "r"(out5), \
    [c_ptr6] "r"(out6), [c_ptr7] "r"(out7), \
    [bias] "r"(bias), [alpha] "r"(alpha), [vmax] "r"(vmax), \
    [flag_act] "r"(flag_act), [rem_cnt] "r"(tail), [last] "r"(last), \
    [scale] "r"(scale) \
  : "cc", "memory", "x0", "x1", "p0", "p1", "z0", "z1", "z2", "z3", "z4", \
    "z5", "z6", "z7", \
    "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", \
    "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", \
    "z27", "z28", "z29", "z30", "z31"

template <>
inline void gemm_smmla_int8_kernel_8x1(SMMLA_PARAMS(float)) {
  asm volatile(INIT_SMMLA_8x2 COMPUTE_SMMLA_8x2 COMPUTE_SMMLA_8x2_REMAIN
                   CVT_SMMLA_INT32_TO_FP32_8x2 SMMLA_ACT_PROCESS_8x4
                       SMMLA_STORE_FP32_8x2 ASM_PARAMS_FP32);
}

template <>
inline void gemm_smmla_int8_kernel_8x1(SMMLA_PARAMS(int8_t)) {
  int8_t out0[4] = {0};
  int8_t out1[4] = {0};
  int8_t out2[4] = {0};
  int8_t out3[4] = {0};
  int8_t out4[4] = {0};
  int8_t out5[4] = {0};
  int8_t out6[4] = {0};
  int8_t out7[4] = {0};
  float vmax[4] = {-127.f, -127.f, -127.f, -127.f};

  asm volatile(INIT_SMMLA_8x2 COMPUTE_SMMLA_8x2 COMPUTE_SMMLA_8x2_REMAIN
                   CVT_SMMLA_INT32_TO_FP32_8x2 SMMLA_ACT_PROCESS_8x4
                       SMMLA_STORE_INT8_8x1 ASM_PARAMS_INT8);

  int cnt = 2;
  if (last == 0) cnt = 1;
  for (int i = 0; i < cnt; i++) {
    c_ptr0[i] = out0[i];
    c_ptr1[i] = out1[i];
    c_ptr2[i] = out2[i];
    c_ptr3[i] = out3[i];
    c_ptr4[i] = out4[i];
    c_ptr5[i] = out5[i];
    c_ptr6[i] = out6[i];
    c_ptr7[i] = out7[i];
  }
  c_ptr0 += cnt;
  c_ptr1 += cnt;
  c_ptr2 += cnt;
  c_ptr3 += cnt;
  c_ptr4 += cnt;
  c_ptr5 += cnt;
  c_ptr6 += cnt;
  c_ptr7 += cnt;
}

template <>
inline void gemm_smmla_int8_kernel_8x4(SMMLA_PARAMS(float)) {
  asm volatile(
      INIT_SMMLA_8x4 COMPUTE_SMMLA_8x4_0 COMPUTE_SMMLA_8x4_1
          COMPUTE_SMMLA_8x4_REMAIN CVT_SMMLA_INT32_TO_FP32_8x4
              SMMLA_ACT_PROCESS_8x4 SMMLA_STORE_FP32_8x4 ASM_PARAMS_FP32);
}

template <>
inline void gemm_smmla_int8_kernel_8x4(SMMLA_PARAMS(int8_t)) {
  float vmax[4] = {-127.f, -127.f, -127.f, -127.f};
  asm volatile(
      INIT_SMMLA_8x4 COMPUTE_SMMLA_8x4_0 COMPUTE_SMMLA_8x4_1
          COMPUTE_SMMLA_8x4_REMAIN CVT_SMMLA_INT32_TO_FP32_8x4
              SMMLA_ACT_PROCESS_8x4 SMMLA_STORE_INT8_8x4 ASM_PARAMS_FP32_INT8);
}

template <>
inline void gemm_smmla_int8_kernel_8x12(SMMLA_PARAMS(float)) {
  asm volatile(
      INIT_SMMLA_8x12 COMPUTE_SMMLA_8x12_0 COMPUTE_SMMLA_8x12_1
          COMPUTE_SMMLA_8x12_REMAIN CVT_SMMLA_INT32_TO_FP32_8x12
              SMMLA_ACT_PROCESS_8x12 SMMLA_STORE_FP32_8x12 ASM_PARAMS_FP32);
}

template <>
inline void gemm_smmla_int8_kernel_8x12(SMMLA_PARAMS(int8_t)) {
  float vmax[4] = {-127.f, -127.f, -127.f, -127.f};
  asm volatile(INIT_SMMLA_8x12 COMPUTE_SMMLA_8x12_0 COMPUTE_SMMLA_8x12_1
                   COMPUTE_SMMLA_8x12_REMAIN CVT_SMMLA_INT32_TO_FP32_8x12
                       SMMLA_ACT_PROCESS_8x12 SMMLA_STORE_INT8_8x12
                           ASM_PARAMS_FP32_INT8);
}
// clang-format on
/// a: m*k  b: k*n  c: m*n
// A: m/8 * (8 * 8 * k_8), A0 = a0_0-7 + a1_0-7 A1 = a2_0-7 + a3_0-7
// B: (k_8 * (8 * 12)) * n_12, B0 = b0-7_0 + b0-7_1 B1 = b0-7_2 + b0-7_3
// C: (m/8 * ((8 * 12)) * n_12
// smmla = A(2x8) * B(8x2) = C(2x2) = c0_0 + c0_1 + c1_0 + c1_1
template <typename dtype>
void gemm_prepack_int8_sve(const int8_t* A_packed,
                           const int8_t* B,
                           const float* bias,
                           dtype* C,
                           int M,
                           int N,
                           int K,
                           bool is_bias,
                           bool is_transB,
                           const float* scale,
                           const operators::ActivationParam act_param,
                           ARMContext* ctx) {
  // l2_size compute
  size_t llc_size = ctx->llc_size() / 4;
  auto workspace = ctx->workspace_data<int8_t>();
  int kup = ROUNDUP_SVE(K, 8);
  //! MBLOCK_INT8_SVE * x (result) + MBLOCK_INT8_SVE * k (A) + x * k (B) = l2
  int x_block = (llc_size - (MBLOCK_INT8_SVE * kup)) /
                (sizeof(int8_t) * (kup + MBLOCK_INT8_SVE));
  x_block /= NBLOCK_INT8_SVE;
  x_block *= NBLOCK_INT8_SVE;

  int x_num = (N + (x_block - 1)) / x_block;
  x_block = (N + x_num - 1) / x_num;
  x_block = (x_block + NBLOCK_INT8_SVE - 1) / NBLOCK_INT8_SVE;
  x_block *= NBLOCK_INT8_SVE;
  x_block = x_block < NBLOCK_INT8_SVE ? NBLOCK_INT8_SVE : x_block;

  int k_cnt = kup >> 3;
  float local_alpha = 0.f;
  float offset = 0.f;
  float threshold = 6.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  float alpha[16] = {0.f,
                     0.f,
                     0.f,
                     0.f,
                     0.f,
                     0.f,
                     0.f,
                     0.f,
                     0.f,
                     0.f,
                     0.f,
                     0.f,
                     0.5f,
                     0.5f,
                     0.5f,
                     0.5f};
  if (act_param.has_active) {
    switch (act_param.active_type) {
      case lite_api::ActivationType::kRelu:
        flag_act = 0x01;
        break;
      case lite_api::ActivationType::kRelu6:
        flag_act = 0x02;
        local_alpha = act_param.Relu_clipped_coef;
        for (int i = 0; i < 4; i++) {
          alpha[i] = local_alpha;
        }
        break;
      case lite_api::ActivationType::kLeakyRelu:
        flag_act = 0x03;
        local_alpha = act_param.Leaky_relu_alpha;
        for (int i = 0; i < 4; i++) {
          alpha[i] = local_alpha;
        }
        break;
      case lite_api::ActivationType::kHardSwish:
        flag_act = 0x04;
        local_alpha = 1.0 / act_param.hard_swish_scale;
        offset = act_param.hard_swish_offset;
        threshold = act_param.hard_swish_threshold;
        for (int i = 0; i < 4; i++) {
          alpha[i] = offset;
          alpha[i + 4] = local_alpha;
          alpha[i + 8] = threshold;
        }
        break;
      default:
        break;
    }
  }

  // unroll 2 loop
  int tail_pre = (k_cnt & (KBLOCK_INT8_SVE - 1));
  int k_pre = ((k_cnt + KBLOCK_INT8_SVE - 1) / KBLOCK_INT8_SVE) - 1;
  bool flag_p_remain = false;
  int remain = 0;
  if (tail_pre == 0) {
    tail_pre = KBLOCK_INT8_SVE;
  }
  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    xmax = (xmax > N) ? N : xmax;
    int bblocks = (xmax - x0 + NBLOCK_INT8_SVE - 1) / NBLOCK_INT8_SVE;
    remain = xmax - x0 - (bblocks - 1) * NBLOCK_INT8_SVE;

    if (remain > 0 && remain != 12) {
      flag_p_remain = true;
    }
    //! load bpanel
    auto b_pannel = static_cast<int8_t*>(workspace);
    if (!is_transB) {
      // K * N
      loadb_k8n12_int8_sve(b_pannel, B, N, 0, K, x0, xmax);
    } else {
      // N X K
      loadb_k8n12_trans_int8_sve(b_pannel, B, K, 0, K, x0, xmax);
    }

    LITE_PARALLEL_COMMON_BEGIN(y, tid, M, 0, MBLOCK_INT8_SVE) {
      unsigned int ymax = y + MBLOCK_INT8_SVE;
      ymax = (ymax > M) ? M : ymax;
      float32_t bias_local[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      if (is_bias) {
        int j = 0;
        for (int i = y; i < ymax && j < 8; i++, j++) {
          bias_local[j] = bias[i];
        }
      }
      float32_t scale_local[8] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
      if (scale) {
        int j = 0;
        for (int i = y; i < ymax && j < 8; i++, j++) {
          scale_local[j] = scale[i];
        }
      }
      dtype cout0[NBLOCK_INT8_SVE] = {0};
      dtype cout1[NBLOCK_INT8_SVE] = {0};
      dtype cout2[NBLOCK_INT8_SVE] = {0};
      dtype cout3[NBLOCK_INT8_SVE] = {0};
      dtype cout4[NBLOCK_INT8_SVE] = {0};
      dtype cout5[NBLOCK_INT8_SVE] = {0};
      dtype cout6[NBLOCK_INT8_SVE] = {0};
      dtype cout7[NBLOCK_INT8_SVE + 16] = {0};

      dtype* c_ptr0 = C + y * N + x0;
      dtype* c_ptr1 = c_ptr0 + N;
      dtype* c_ptr2 = c_ptr1 + N;
      dtype* c_ptr3 = c_ptr2 + N;
      dtype* c_ptr4 = c_ptr3 + N;
      dtype* c_ptr5 = c_ptr4 + N;
      dtype* c_ptr6 = c_ptr5 + N;
      dtype* c_ptr7 = c_ptr6 + N;

      dtype* pout0 = cout0;
      dtype* pout1 = cout1;
      dtype* pout2 = cout2;
      dtype* pout3 = cout3;
      dtype* pout4 = cout4;
      dtype* pout5 = cout5;
      dtype* pout6 = cout6;
      dtype* pout7 = cout7;
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

      const int8_t* a_ptr_l = A_packed + y * kup;
      const int8_t* b_ptr = b_pannel;
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
            const int8_t* a_ptr = a_ptr_l;
            // 8x4
            gemm_smmla_int8_kernel_8x4<dtype>(a_ptr,
                                              b_ptr,
                                              bias_local,
                                              c_ptr0,
                                              c_ptr1,
                                              c_ptr2,
                                              c_ptr3,
                                              c_ptr4,
                                              c_ptr5,
                                              c_ptr6,
                                              c_ptr7,
                                              scale_local,
                                              alpha,
                                              k_pre,
                                              tail_pre,
                                              flag_act,
                                              1);
          }
          for (int i = 0; i < rem_rem; i += 2) {
            const int8_t* a_ptr = a_ptr_l;
            int last = (i + 1 == rem_rem) ? 0 : 1;
            // 8x1
            gemm_smmla_int8_kernel_8x1<dtype>(a_ptr,
                                              b_ptr,
                                              bias_local,
                                              c_ptr0,
                                              c_ptr1,
                                              c_ptr2,
                                              c_ptr3,
                                              c_ptr4,
                                              c_ptr5,
                                              c_ptr6,
                                              c_ptr7,
                                              scale_local,
                                              alpha,
                                              k_pre,
                                              tail_pre,
                                              flag_act,
                                              last);
          }
        } else {
          const int8_t* a_ptr = a_ptr_l;
          gemm_smmla_int8_kernel_8x12<dtype>(a_ptr,
                                             b_ptr,
                                             bias_local,
                                             c_ptr0,
                                             c_ptr1,
                                             c_ptr2,
                                             c_ptr3,
                                             c_ptr4,
                                             c_ptr5,
                                             c_ptr6,
                                             c_ptr7,
                                             scale_local,
                                             alpha,
                                             k_pre,
                                             tail_pre,
                                             flag_act,
                                             1);
        }
      }
    }
    LITE_PARALLEL_COMMON_END();
  }
}
#define GEMM_PREPACK_INT8_SVE(dtype)              \
  template void gemm_prepack_int8_sve<dtype>(     \
      const int8_t* A_packed,                     \
      const int8_t* B,                            \
      const float* bias,                          \
      dtype* C,                                   \
      int M,                                      \
      int N,                                      \
      int K,                                      \
      bool is_bias,                               \
      bool is_transB,                             \
      const float* scale,                         \
      const operators::ActivationParam act_param, \
      ARMContext* ctx);
GEMM_PREPACK_INT8_SVE(int8_t);
GEMM_PREPACK_INT8_SVE(float);

#undef ZIP8x16_16x8
#undef LOAD_DATA_SVE
#undef STORE_DATA_SVE
#undef SMMLA_PARAMS
#undef INIT_SMMLA_8x12
#undef COMPUTE_SMMLA_8x12_0
#undef COMPUTE_SMMLA_8x12_1
#undef COMPUTE_SMMLA_REMAIN
#undef CVT_SMMLA_INT32_TO_FP32_8x12
#undef SMMLA_RELU_8x12
#undef SMMLA_LEAKYRELU_8x12
#undef SMMLA_HARDSWIDH_8x12
#undef SMMLA_ACT_PROCESS_8x12
#undef SMMLA_STORE_FP32_8x12
#undef SMMLA_STORE_FP32_8x4
#undef SMMLA_STORE_FP32_8x2
#undef SMMLA_STORE_INT8
#undef ASM_PARAMS_FP32
#undef ASM_PARAMS_INT8
#undef GEMM_PREPACK_INT8_SVE
#undef SMMLA_STORE_INT8_8x4
#undef SMMLA_STORE_INT8_8x4_1
#undef SMMLA_STORE_INT8_8x12
#undef SMMLA_STORE_INT8_8x12_1
#undef SMMLA_STORE_INT8_8x1
#undef ASM_PARAMS_FP32_INT8
}  // namespace sve
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
