// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/backends/arm/math/packed_sgemm.h"
#include <arm_neon.h>
#include "lite/backends/arm/math/conv_block_utils.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

#ifdef __aarch64__
void prepackA_8x12(float *out,
                   const float *in,
                   float alpha,
                   int ldin,
                   int m0,
                   int mmax,
                   int k0,
                   int kmax);

void prepackA_trans_8x12(float *out,
                         const float *in,
                         float alpha,
                         int ldin,
                         int m0,
                         int mmax,
                         int k0,
                         int kmax);

void sgemm_prepacked_8x12(bool is_transB,
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

void sgemm_prepacked_8x12_a53(bool is_transB,
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

void pack_m4(float *out,
             const float *in,
             float alpha,
             int ldin,
             int m0,
             int mmax,
             int k0,
             int kmax);

void pack_trans_m4(float *out,
                   const float *in,
                   float alpha,
                   int ldin,
                   int m0,
                   int mmax,
                   int k0,
                   int kmax);

void sgemm_prepacked_4x4(bool is_transB,
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
#else
void prepackA_6x8(float *out,
                  const float *in,
                  float alpha,
                  int ldin,
                  int m0,
                  int mmax,
                  int k0,
                  int kmax);

void prepackA_trans_6x8(float *out,
                        const float *in,
                        float alpha,
                        int ldin,
                        int m0,
                        int mmax,
                        int k0,
                        int kmax);

// v7: for kA72, 6x8
void sgemm_prepacked_6x8(bool is_transB,
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
// for kA53
void sgemm_prepacked_6x8_a53(bool is_transB,
                             int M,
                             int N,
                             int K,
                             const float *A_packed,
                             const float *B,
                             int ldb,
                             float *C,
                             int ldc,
                             const float *bias,
                             bool has_bias,
                             int is_relu,
                             ARMContext *ctx);
#endif  // __aarch64__

void prepackA_4x8(float *out,
                  const float *in,
                  float alpha,
                  int ldin,
                  int m0,
                  int mmax,
                  int k0,
                  int kmax);

void prepackA_trans_4x8(float *out,
                        const float *in,
                        float alpha,
                        int ldin,
                        int m0,
                        int mmax,
                        int k0,
                        int kmax);

// v7: for kA73, 4x8; M <= 4
void sgemm_prepacked_4x8(bool is_transB,
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

/**
 * \brief input data is not transpose
 * for arm-v7a, transform data to block x k x 6 layout
 * for arm-v8a, transform data to block x k x 8 layout
 */
void prepackA(float *out,
              const float *in,
              float alpha,
              int ldin,
              int m0,
              int mmax,
              int k0,
              int kmax,
              bool is_trans,
              ARMContext *ctx) {
#ifdef __aarch64__
  if (mmax <= 4) {
    if (is_trans) {
      prepackA_trans_4x8(out, in, alpha, ldin, m0, mmax, k0, kmax);
    } else {
      prepackA_4x8(out, in, alpha, ldin, m0, mmax, k0, kmax);
    }
  } else {
    if (is_trans) {
      prepackA_trans_8x12(out, in, alpha, ldin, m0, mmax, k0, kmax);
    } else {
      prepackA_8x12(out, in, alpha, ldin, m0, mmax, k0, kmax);
    }
  }
#else
  if (ctx->arch() == kA73 || mmax <= 4) {
    if (is_trans) {
      prepackA_trans_4x8(out, in, alpha, ldin, m0, mmax, k0, kmax);
    } else {
      prepackA_4x8(out, in, alpha, ldin, m0, mmax, k0, kmax);
    }
  } else {
    if (is_trans) {
      prepackA_trans_6x8(out, in, alpha, ldin, m0, mmax, k0, kmax);
    } else {
      prepackA_6x8(out, in, alpha, ldin, m0, mmax, k0, kmax);
    }
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
  int hblock = 4;
#ifdef __aarch64__
  if (m <= 4) {
    hblock = 4;
  } else {
    hblock = 8;
  }
#else
  if (ctx->arch() == kA73 || m <= 4) {
    hblock = 4;
  } else {
    hblock = 6;
  }
#endif
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

/// a: m*k  b: k*n  c: m*n
void sgemm_prepack(bool is_transB,
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
#ifdef __aarch64__
  if (M <= 4) {
    sgemm_prepacked_4x8(is_transB,
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
  } else if (ctx->arch() == kA53) {
    sgemm_prepacked_8x12_a53(is_transB,
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
  } else {
    sgemm_prepacked_8x12(is_transB,
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
  }
#else   // armv7
  if (ctx->arch() == kA73 || M <= 4) {
    sgemm_prepacked_4x8(is_transB,
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
  } else if (ctx->arch() == kA53) {
    auto act_type = act_param.active_type;
    bool has_act = act_param.has_active;
    bool act_flag =
        (has_act == false) ||
        (has_act == true && act_type == lite_api::ActivationType::kRelu);
    bool has_beta = fabsf(beta) > 1e-8f ? true : false;
    bool a53_sgemm = act_flag && !has_beta;
    if (a53_sgemm) {
      sgemm_prepacked_6x8_a53(is_transB,
                              M,
                              N,
                              K,
                              A_packed,
                              B,
                              ldb,
                              C,
                              ldc,
                              bias,
                              has_bias,
                              static_cast<int>(has_act),
                              ctx);
    } else {
      sgemm_prepacked_6x8(is_transB,
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
    }
  } else {
    sgemm_prepacked_6x8(is_transB,
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
  }
#endif  // arm64
}

#ifdef __aarch64__
/*
 * The following function prepackA_8x12 is base on
 * https://github.com/ARM-software/ComputeLibrary/
 *
 * Copyright (c) 2017-2019 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
void prepackA_8x12(float *dout,
                   const float *inptr,
                   float alpha,
                   int ldin,
                   int m0,
                   int mmax,
                   int k0,
                   int kmax) {
  int x_len = kmax - k0;
  int stride = x_len * 8;
  float zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(float) * x_len);
  bool has_alpha = fabsf(alpha - 1.f) > 1e-8f;

#pragma omp parallel for
  for (int y = m0; y < mmax; y += 8) {
    float *outptr = dout + stride * (y - m0) / 8;

    const float *inptr0 = inptr + y * ldin + k0;
    const float *inptr1 = inptr0 + ldin;
    const float *inptr2 = inptr1 + ldin;
    const float *inptr3 = inptr2 + ldin;
    const float *inptr4 = inptr3 + ldin;
    const float *inptr5 = inptr4 + ldin;
    const float *inptr6 = inptr5 + ldin;
    const float *inptr7 = inptr6 + ldin;

    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]        \n"
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
        :
        : [ptr0] "r"(inptr0),
          [ptr1] "r"(inptr1),
          [ptr2] "r"(inptr2),
          [ptr3] "r"(inptr3),
          [ptr4] "r"(inptr4),
          [ptr5] "r"(inptr5),
          [ptr6] "r"(inptr6),
          [ptr7] "r"(inptr7)
        : "memory");

    int x = x_len;
    //! cope with row index exceed real size, set to zero buffer
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
    for (; x > 7; x -= 8) {
      asm volatile(
          "cbz    %w[has_alpha], 0f\n"            /* check alpha == 1.f? */
          "dup    v31.4s, %w[alpha]\n"            /* alpha to vector */
          "ldp    q0, q1,     [%[inptr0]], #32\n" /* load r0, a0~a7 */
          "ldp    q2, q3,     [%[inptr1]], #32\n" /* load r1, b0~b7 */
          "fmul   v0.4s,  v31.4s, v0.4s\n"        /* mul alpha */
          "fmul   v1.4s,  v31.4s, v1.4s\n"        /* mul alpha */
          "ldp    q4, q5,     [%[inptr2]], #32\n" /* load r2, c0~c7 */
          "fmul   v2.4s,  v31.4s, v2.4s\n"        /* mul alpha */
          "fmul   v3.4s,  v31.4s, v3.4s\n"        /* mul alpha */
          "ldp    q6, q7,     [%[inptr3]], #32\n" /* load r3, d0~d7 */
          "fmul   v4.4s,  v31.4s, v4.4s\n"        /* mul alpha */
          "fmul   v5.4s,  v31.4s, v5.4s\n"        /* mul alpha */
          "ldp    q8, q9,     [%[inptr4]], #32\n" /* load r4, e0~e7 */
          "fmul   v6.4s,  v31.4s, v6.4s\n"        /* mul alpha */
          "fmul   v7.4s,  v31.4s, v7.4s\n"        /* mul alpha */
          "ldp    q10, q11,   [%[inptr5]], #32\n" /* load r5, f0~f7 */
          "fmul   v8.4s,  v31.4s, v8.4s\n"        /* mul alpha */
          "fmul   v9.4s,  v31.4s, v9.4s\n"        /* mul alpha */
          "ldp    q12, q13,   [%[inptr6]], #32\n" /* load r6, g0~g7 */
          "fmul   v10.4s,  v31.4s, v10.4s\n"      /* mul alpha */
          "fmul   v11.4s,  v31.4s, v11.4s\n"      /* mul alpha */
          "ldp    q14, q15,   [%[inptr7]], #32\n" /* load r7, h0~h7 */
          "fmul   v12.4s,  v31.4s, v12.4s\n"      /* mul alpha */
          "fmul   v13.4s,  v31.4s, v13.4s\n"      /* mul alpha */
          "fmul   v14.4s,  v31.4s, v14.4s\n"      /* mul alpha */
          "fmul   v15.4s,  v31.4s, v15.4s\n"      /* mul alpha */
          "b 1f\n"                                /* to main process */
          "0: \n"                                 /* alpha == 1 */
          "ldp    q0, q1,     [%[inptr0]], #32\n" /* load r0, a0~a7 */
          "ldp    q2, q3,     [%[inptr1]], #32\n" /* load r1, b0~b7 */
          "ldp    q4, q5,     [%[inptr2]], #32\n" /* load r2, c0~c7 */
          "ldp    q6, q7,     [%[inptr3]], #32\n" /* load r3, d0~d7 */
          "ldp    q8, q9,     [%[inptr4]], #32\n" /* load r4, e0~e7 */
          "ldp    q10, q11,   [%[inptr5]], #32\n" /* load r5, f0~f7 */
          "ldp    q12, q13,   [%[inptr6]], #32\n" /* load r6, g0~g7 */
          "ldp    q14, q15,   [%[inptr7]], #32\n" /* load r7, h0~h7 */
          "1: \n"                                 /* main process */
          "trn1   v16.4s, v0.4s, v2.4s\n"         /* a0b0a2b2*/
          "trn2   v17.4s, v0.4s, v2.4s\n"         /* a1b1a3b3*/
          "trn1   v18.4s, v1.4s, v3.4s\n"         /* a4b4a6b6*/
          "trn2   v19.4s, v1.4s, v3.4s\n"         /* a5b5a7b7*/

          "trn1   v20.4s, v4.4s, v6.4s\n" /* c0d0c2d2*/
          "trn2   v21.4s, v4.4s, v6.4s\n" /* c1d1c3d3*/
          "trn1   v22.4s, v5.4s, v7.4s\n" /* c4d4c6d6*/
          "trn2   v23.4s, v5.4s, v7.4s\n" /* c5d5c7d7*/

          "trn1   v24.4s, v8.4s, v10.4s\n" /* e0f0e2f2*/
          "trn2   v25.4s, v8.4s, v10.4s\n" /* e1f1e3f3*/
          "trn1   v26.4s, v9.4s, v11.4s\n" /* e4f4e6f6*/
          "trn2   v27.4s, v9.4s, v11.4s\n" /* e5f5e7f7*/

          "trn1   v28.4s, v12.4s, v14.4s\n" /* g0h0g2h2*/
          "trn2   v29.4s, v12.4s, v14.4s\n" /* g1h1g3h3*/
          "trn1   v30.4s, v13.4s, v15.4s\n" /* g4h4g6h6*/
          "trn2   v31.4s, v13.4s, v15.4s\n" /* g5h5g7h7*/

          "trn1   v0.2d, v16.2d, v20.2d\n" /* a0b0c0d0 */
          "trn1   v1.2d, v24.2d, v28.2d\n" /* e0f0g0h0 */
          "trn1   v2.2d, v17.2d, v21.2d\n" /* a1b1c1d1 */
          "trn1   v3.2d, v25.2d, v29.2d\n" /* e1b1c1d1 */

          "trn2   v4.2d, v16.2d, v20.2d\n"    /* a2b2c2d2 */
          "trn2   v5.2d, v24.2d, v28.2d\n"    /* e2f2g2h2 */
          "stp    q0, q1, [%[outptr]], #32\n" /* save q0, q1, a0~h0*/
          "trn2   v6.2d, v17.2d, v21.2d\n"    /* a3b3c3d3 */
          "trn2   v7.2d, v25.2d, v29.2d\n"    /* e3f3g3h3 */
          "stp    q2, q3, [%[outptr]], #32\n" /* save q2, q3, a1~h1*/

          "trn1   v8.2d, v18.2d, v22.2d\n"    /* a4b4c4d4 */
          "trn1   v9.2d, v26.2d, v30.2d\n"    /* e4f4g4h4 */
          "stp    q4, q5, [%[outptr]], #32\n" /* save q4, q5, a2~h2*/
          "trn1   v10.2d, v19.2d, v23.2d\n"   /* a5b5c5d5 */
          "trn1   v11.2d, v27.2d, v31.2d\n"   /* e5f5g5h5 */
          "stp    q6, q7, [%[outptr]], #32\n" /* save q6, q7, a3~h3*/

          "trn2   v12.2d, v18.2d, v22.2d\n"     /* a6b6c6d6 */
          "trn2   v13.2d, v26.2d, v30.2d\n"     /* e6f6g6h6 */
          "stp    q8, q9, [%[outptr]], #32\n"   /* save q8, q9, a4~h4*/
          "trn2   v14.2d, v19.2d, v23.2d\n"     /* a7b7c7d7 */
          "trn2   v15.2d, v27.2d, v31.2d\n"     /* e7f7g7h7 */
          "stp    q10, q11, [%[outptr]], #32\n" /* save q10, q11, a5~h5*/

          "stp    q12, q13, [%[outptr]], #32\n" /* save q12, q13, a6~h6*/
          "stp    q14, q15, [%[outptr]], #32\n" /* save q14, q15, a7~h7*/
          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [inptr4] "+r"(inptr4),
            [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6),
            [inptr7] "+r"(inptr7),
            [outptr] "+r"(outptr)
          : [alpha] "r"(alpha), [has_alpha] "r"(has_alpha)
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
            "v16",
            "v17",
            "v18",
            "v19",
            "v20",
            "v21",
            "v22",
            "v23",
            "v24",
            "v25",
            "v26",
            "v27",
            "v28",
            "v29",
            "v30",
            "v31",
            "cc",
            "memory");
    }

    for (; x > 0; x--) {
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

void prepackA_4x8(float *outptr,
                  const float *inptr,
                  float alpha,
                  int ldin,
                  int m0,
                  int mmax,
                  int k0,
                  int kmax) {
  int x_len = kmax - k0;
  float zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(float) * x_len);

  bool has_alpha = fabsf(alpha - 1.f) > 1e-8f;
  float32x4_t valpha = vdupq_n_f32(alpha);
#pragma omp parallel for
  for (int y = m0; y < mmax; y += 4) {
    const float *inptr0 = inptr + y * ldin + k0;
    const float *inptr1 = inptr0 + ldin;
    const float *inptr2 = inptr1 + ldin;
    const float *inptr3 = inptr2 + ldin;

    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]        \n"
        "prfm   pldl1keep, [%[ptr0], #64]   \n"
        "prfm   pldl1keep, [%[ptr1]]        \n"
        "prfm   pldl1keep, [%[ptr1], #64]   \n"
        "prfm   pldl1keep, [%[ptr2]]        \n"
        "prfm   pldl1keep, [%[ptr2], #64]   \n"
        "prfm   pldl1keep, [%[ptr3]]        \n"
        "prfm   pldl1keep, [%[ptr3], #64]   \n"
        :
        : [ptr0] "r"(inptr0),
          [ptr1] "r"(inptr1),
          [ptr2] "r"(inptr2),
          [ptr3] "r"(inptr3)
        : "memory");
    int x = x_len;
    if ((y + 3) >= mmax) {
      switch ((y + 3) - mmax) {
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

    for (; x > 7; x -= 8) {
      // clang-format off
      asm volatile(
          "cbz %w[has_alpha], 0f\n"
          "ldp q0, q1, [%[inptr0]], #32\n"    // load r0, a0~a7
          "ldp q2, q3, [%[inptr1]], #32\n"    // load r1, b0~b7
          "fmul v0.4s, v0.4s, %[alpha].4s\n"
          "fmul v1.4s, v1.4s, %[alpha].4s\n"
          "ldp  q4, q5, [%[inptr2]], #32\n"   // load r2, c0~c7
          "fmul v2.4s, v2.4s, %[alpha].4s\n"
          "fmul v3.4s, v3.4s, %[alpha].4s\n"
          "ldp  q6, q7, [%[inptr3]], #32\n"   // load r3, d0~d7
          "fmul v4.4s, v4.4s, %[alpha].4s\n"
          "fmul v5.4s, v5.4s, %[alpha].4s\n"
          "fmul v6.4s, v6.4s, %[alpha].4s\n"
          "fmul v7.4s, v7.4s, %[alpha].4s\n"
          "b 1f\n"                            // to main process
          "0: \n"                             // alpha == 1
          "ldp  q0, q1, [%[inptr0]], #32\n"   // load r0, a0~a7
          "ldp  q2, q3, [%[inptr1]], #32\n"   // load r1, b0~b7
          "ldp  q4, q5, [%[inptr2]], #32\n"   // load r2, c0~c7
          "ldp  q6, q7, [%[inptr3]], #32\n"   // load r3, d0~d7
          "1: \n"
          "trn1 v8.4s, v0.4s, v2.4s\n"        // a0b0a2b2
          "trn2 v9.4s, v0.4s, v2.4s\n"        // a1b1a3b3
          "trn1 v10.4s, v1.4s, v3.4s\n"       // a4b4a6b6
          "trn2 v11.4s, v1.4s, v3.4s\n"       // a5b5a7b7
          "trn1 v12.4s, v4.4s, v6.4s\n"       // c0d0c2d2
          "trn2 v13.4s, v4.4s, v6.4s\n"       // c1d1c3d3
          "trn1 v14.4s, v5.4s, v7.4s\n"       // c4d4c6d6
          "trn2 v15.4s, v5.4s, v7.4s\n"       // c5d5c7d7

          "trn1 v0.2d, v8.2d, v12.2d\n"       // a0b0c0d0
          "trn1 v1.2d, v9.2d, v13.2d\n"       // a1b1c1d1
          "trn2 v2.2d, v8.2d, v12.2d\n"       // a2b2c2d2
          "trn2 v3.2d, v9.2d, v13.2d\n"       // a3b3c3d3
          "st1 {v0.4s}, [%[outptr]], #16\n"
          "trn1 v4.2d, v10.2d, v14.2d\n"      // a4b4c4d4
          "st1 {v1.4s}, [%[outptr]], #16\n"
          "trn1 v5.2d, v11.2d, v15.2d\n"      // a5b5c5d5
          "st1 {v2.4s}, [%[outptr]], #16\n"
          "trn2 v6.2d, v10.2d, v14.2d\n"      // a6b6c6d6
          "st1 {v3.4s}, [%[outptr]], #16\n"
          "trn2 v7.2d, v11.2d, v15.2d\n"      // a7b7c7d7
          "stp q4, q5, [%[outptr]], #32\n"
          "stp q6, q7, [%[outptr]], #32\n"
          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outptr] "+r"(outptr)
          : [has_alpha] "r"(has_alpha), [alpha] "w"(valpha)
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
    }

    for (; x > 0; x--) {
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

void pack_m4(float *dout,
             const float *inptr,
             float alpha,
             int ldin,
             int m0,
             int mmax,
             int k0,
             int kmax) {
  int x_len = kmax - k0;
  int stride = x_len * 4;
  float zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(float) * x_len);
  bool has_alpha = fabsf(alpha - 1.f) > 1e-8f;

#pragma omp parallel for
  for (int y = m0; y < mmax; y += 4) {
    float *outptr = dout + stride * (y - m0) / 4;

    const float *inptr0 = inptr + y * ldin + k0;
    const float *inptr1 = inptr0 + ldin;
    const float *inptr2 = inptr1 + ldin;
    const float *inptr3 = inptr2 + ldin;

    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]        \n"
        "prfm   pldl1keep, [%[ptr0], #64]   \n"
        "prfm   pldl1keep, [%[ptr1]]        \n"
        "prfm   pldl1keep, [%[ptr1], #64]   \n"
        "prfm   pldl1keep, [%[ptr2]]        \n"
        "prfm   pldl1keep, [%[ptr2], #64]   \n"
        "prfm   pldl1keep, [%[ptr3]]        \n"
        "prfm   pldl1keep, [%[ptr3], #64]   \n"
        :
        : [ptr0] "r"(inptr0),
          [ptr1] "r"(inptr1),
          [ptr2] "r"(inptr2),
          [ptr3] "r"(inptr3)
        : "memory");

    int x = x_len;
    //! cope with row index exceed real size, set to zero buffer
    if ((y + 3) >= mmax) {
      switch ((y + 3) - mmax) {
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
    for (; x > 7; x -= 8) {
      asm volatile(
          "cbz    %w[has_alpha], 0f\n"            /* check alpha == 1.f? */
          "dup    v31.4s, %w[alpha]\n"            /* alpha to vector */
          "ldp    q0, q1,     [%[inptr0]], #32\n" /* load r0, a0~a7 */
          "ldp    q2, q3,     [%[inptr1]], #32\n" /* load r1, b0~b7 */
          "fmul   v0.4s,  v31.4s, v0.4s\n"        /* mul alpha */
          "fmul   v1.4s,  v31.4s, v1.4s\n"        /* mul alpha */
          "ldp    q4, q5,     [%[inptr2]], #32\n" /* load r2, c0~c7 */
          "fmul   v2.4s,  v31.4s, v2.4s\n"        /* mul alpha */
          "fmul   v3.4s,  v31.4s, v3.4s\n"        /* mul alpha */
          "ldp    q6, q7,     [%[inptr3]], #32\n" /* load r3, d0~d7 */
          "fmul   v4.4s,  v31.4s, v4.4s\n"        /* mul alpha */
          "fmul   v5.4s,  v31.4s, v5.4s\n"        /* mul alpha */
          "fmul   v6.4s,  v31.4s, v6.4s\n"        /* mul alpha */
          "fmul   v7.4s,  v31.4s, v7.4s\n"        /* mul alpha */
          "b 1f\n"                                /* to main process */
          "0: \n"                                 /* alpha == 1 */
          "ldp    q0, q1,     [%[inptr0]], #32\n" /* load r0, a0~a7 */
          "ldp    q2, q3,     [%[inptr1]], #32\n" /* load r1, b0~b7 */
          "ldp    q4, q5,     [%[inptr2]], #32\n" /* load r2, c0~c7 */
          "ldp    q6, q7,     [%[inptr3]], #32\n" /* load r3, d0~d7 */
          "1: \n"                                 /* main process */
          "trn1   v8.4s, v0.4s, v2.4s\n"          /* a0b0a2b2*/
          "trn2   v9.4s, v0.4s, v2.4s\n"          /* a1b1a3b3*/
          "trn1   v10.4s, v1.4s, v3.4s\n"         /* a4b4a6b6*/
          "trn2   v11.4s, v1.4s, v3.4s\n"         /* a5b5a7b7*/

          "trn1   v12.4s, v4.4s, v6.4s\n" /* c0d0c2d2*/
          "trn2   v13.4s, v4.4s, v6.4s\n" /* c1d1c3d3*/
          "trn1   v14.4s, v5.4s, v7.4s\n" /* c4d4c6d6*/
          "trn2   v15.4s, v5.4s, v7.4s\n" /* c5d5c7d7*/

          "trn1   v0.2d, v8.2d, v12.2d\n"  /* a0b0c0d0 */
          "trn1   v1.2d, v9.2d, v13.2d\n"  /* a1b1c1d1 */
          "trn1   v2.2d, v10.2d, v14.2d\n" /* a4b4c4d4 */
          "trn1   v3.2d, v11.2d, v15.2d\n" /* a5b5c5d5 */

          "trn2   v4.2d, v8.2d, v12.2d\n"     /* a2b2c2d2 */
          "trn2   v5.2d, v9.2d, v13.2d\n"     /* a3b3c3d3 */
          "stp    q0, q1, [%[outptr]], #32\n" /* save q0, q1, a0~h0*/
          "trn2   v6.2d, v10.2d, v14.2d\n"    /* a6b6c6d6 */
          "trn2   v7.2d, v11.2d, v15.2d\n"    /* a7b7c7d7 */
          "stp    q4, q5, [%[outptr]], #32\n" /* save q2, q3, a1~h1*/
          "stp    q2, q3, [%[outptr]], #32\n" /* save q4, q5, a2~h2*/
          "stp    q6, q7, [%[outptr]], #32\n" /* save q6, q7, a3~h3*/

          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outptr] "+r"(outptr)
          : [alpha] "r"(alpha), [has_alpha] "r"(has_alpha)
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
    }

    for (; x > 0; x--) {
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

void prepackA_trans_8x12(float *outptr,
                         const float *in,
                         float alpha,
                         int ldin,
                         int m0,
                         int mmax,
                         int k0,
                         int kmax) {
  auto inptr = in + k0 * ldin + m0;
  uint32_t mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int x_len = mmax - m0;
  int y_len = kmax - k0;
  int right_remain = x_len - 8 * (x_len / 8);
  int stride_out = 8 * y_len;

  float32x4_t vzero = vdupq_n_f32(0.f);
  uint32x4_t vmask1 =
      vcltq_u32(vld1q_u32(mask_buffer), vdupq_n_u32(right_remain));
  uint32x4_t vmask2 =
      vcltq_u32(vld1q_u32(mask_buffer + 4), vdupq_n_u32(right_remain));

  bool has_alpha = fabsf(alpha - 1.f) > 1e-8f;
  float32x4_t valpha = vdupq_n_f32(alpha);

#pragma omp parallel for
  for (int y = 0; y < y_len - 3; y += 4) {
    const float *ptr0 = inptr + y * ldin;
    const float *ptr1 = ptr0 + ldin;
    const float *ptr2 = ptr1 + ldin;
    const float *ptr3 = ptr2 + ldin;

    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]        \n"
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

    float *outptr_row_col = outptr + y * 8;
    int i = 0;
    for (; i < x_len - 7; i += 8) {
      float32x4_t vr00 = vld1q_f32(ptr0);
      float32x4_t vr01 = vld1q_f32(ptr0 + 4);
      float32x4_t vr10 = vld1q_f32(ptr1);
      float32x4_t vr11 = vld1q_f32(ptr1 + 4);
      float32x4_t vr20 = vld1q_f32(ptr2);
      float32x4_t vr21 = vld1q_f32(ptr2 + 4);
      float32x4_t vr30 = vld1q_f32(ptr3);
      float32x4_t vr31 = vld1q_f32(ptr3 + 4);
      if (has_alpha) {
        vr00 = vmulq_f32(vr00, valpha);
        vr01 = vmulq_f32(vr01, valpha);
        vr10 = vmulq_f32(vr10, valpha);
        vr11 = vmulq_f32(vr11, valpha);
        vr20 = vmulq_f32(vr20, valpha);
        vr21 = vmulq_f32(vr21, valpha);
        vr30 = vmulq_f32(vr30, valpha);
        vr31 = vmulq_f32(vr31, valpha);
      }

      vst1q_f32(outptr_row_col, vr00);
      vst1q_f32(outptr_row_col + 4, vr01);
      vst1q_f32(outptr_row_col + 8, vr10);
      vst1q_f32(outptr_row_col + 12, vr11);
      vst1q_f32(outptr_row_col + 16, vr20);
      vst1q_f32(outptr_row_col + 20, vr21);
      vst1q_f32(outptr_row_col + 24, vr30);
      vst1q_f32(outptr_row_col + 28, vr31);

      ptr0 += 8;
      ptr1 += 8;
      ptr2 += 8;
      ptr3 += 8;

      outptr_row_col += stride_out;
    }
    if (right_remain > 0) {
      float32x4_t vr00 = vld1q_f32(ptr0);
      float32x4_t vr01 = vld1q_f32(ptr0 + 4);
      float32x4_t vr10 = vld1q_f32(ptr1);
      float32x4_t vr11 = vld1q_f32(ptr1 + 4);
      float32x4_t vr20 = vld1q_f32(ptr2);
      float32x4_t vr21 = vld1q_f32(ptr2 + 4);
      float32x4_t vr30 = vld1q_f32(ptr3);
      float32x4_t vr31 = vld1q_f32(ptr3 + 4);

      if (has_alpha) {
        vr00 = vmulq_f32(vr00, valpha);
        vr01 = vmulq_f32(vr01, valpha);
        vr10 = vmulq_f32(vr10, valpha);
        vr11 = vmulq_f32(vr11, valpha);
        vr20 = vmulq_f32(vr20, valpha);
        vr21 = vmulq_f32(vr21, valpha);
        vr30 = vmulq_f32(vr30, valpha);
        vr31 = vmulq_f32(vr31, valpha);
      }

      float32x4_t vr00_1 = vbslq_f32(vmask1, vr00, vzero);
      float32x4_t vr01_1 = vbslq_f32(vmask2, vr01, vzero);
      float32x4_t vr10_1 = vbslq_f32(vmask1, vr10, vzero);
      float32x4_t vr11_1 = vbslq_f32(vmask2, vr11, vzero);
      float32x4_t vr20_1 = vbslq_f32(vmask1, vr20, vzero);
      float32x4_t vr21_1 = vbslq_f32(vmask2, vr21, vzero);
      float32x4_t vr30_1 = vbslq_f32(vmask1, vr30, vzero);
      float32x4_t vr31_1 = vbslq_f32(vmask2, vr31, vzero);

      vst1q_f32(outptr_row_col, vr00_1);
      vst1q_f32(outptr_row_col + 4, vr01_1);
      vst1q_f32(outptr_row_col + 8, vr10_1);
      vst1q_f32(outptr_row_col + 12, vr11_1);
      vst1q_f32(outptr_row_col + 16, vr20_1);
      vst1q_f32(outptr_row_col + 20, vr21_1);
      vst1q_f32(outptr_row_col + 24, vr30_1);
      vst1q_f32(outptr_row_col + 28, vr31_1);
    }
  }

#pragma omp parallel for
  for (int y = 4 * (y_len / 4); y < y_len; ++y) {
    const float *ptr0 = inptr + y * ldin;
    float *outptr_row_col = outptr + y * 8;
    int i = 0;
    for (; i < x_len - 7; i += 8) {
      float32x4_t vr0 = vld1q_f32(ptr0);
      float32x4_t vr1 = vld1q_f32(ptr0 + 4);
      if (has_alpha) {
        vr0 = vmulq_f32(vr0, valpha);
        vr1 = vmulq_f32(vr1, valpha);
      }
      vst1q_f32(outptr_row_col, vr0);
      vst1q_f32(outptr_row_col + 4, vr1);

      ptr0 += 8;

      outptr_row_col += stride_out;
    }
    if (right_remain > 0) {
      float32x4_t vr0 = vld1q_f32(ptr0);
      float32x4_t vr1 = vld1q_f32(ptr0 + 4);

      if (has_alpha) {
        vr0 = vmulq_f32(vr0, valpha);
        vr1 = vmulq_f32(vr1, valpha);
      }

      float32x4_t vr0_1 = vbslq_f32(vmask1, vr0, vzero);
      float32x4_t vr1_1 = vbslq_f32(vmask2, vr1, vzero);

      vst1q_f32(outptr_row_col, vr0_1);
      vst1q_f32(outptr_row_col + 4, vr1_1);
    }
  }
}

void prepackA_trans_4x8(float *outptr,
                        const float *in,
                        float alpha,
                        int ldin,
                        int m0,
                        int mmax,
                        int k0,
                        int kmax) {
  auto inptr = in + k0 * ldin + m0;
  bool has_alpha = fabsf(alpha - 1.f) > 1e-8f;
  float32x4_t valpha = vdupq_n_f32(alpha);

  uint32_t mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int x_len = mmax - m0;
  int y_len = kmax - k0;
  int right_remain = x_len - 4 * (x_len / 4);
  int right_pad = 4 - right_remain;
  if (right_remain == 0) {
    right_pad = 0;
  }

  int stride_out = 4 * y_len;

  float32x4_t vzero = vdupq_n_f32(0.f);
  uint32x4_t vmask1 =
      vcltq_u32(vld1q_u32(mask_buffer), vdupq_n_u32(right_remain));

#pragma omp parallel for
  for (int y = 0; y < y_len - 3; y += 4) {
    const float *ptr0 = inptr + y * ldin;
    const float *ptr1 = ptr0 + ldin;
    const float *ptr2 = ptr1 + ldin;
    const float *ptr3 = ptr2 + ldin;

    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]        \n"
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
    float *outptr_row_col = outptr + y * 4;
    int i = 0;
    for (; i < x_len - 3; i += 4) {
      float *ptr_out = outptr_row_col;
      // clang-format off
      asm volatile(
          "cmp %w[has_alpha], #0\n"
          "ld1 {v0.4s}, [%[ptr0]], #16\n"
          "ld1 {v1.4s}, [%[ptr1]], #16\n"
          "ld1 {v2.4s}, [%[ptr2]], #16\n"
          "ld1 {v3.4s}, [%[ptr3]], #16\n"
          "beq  0f\n"
          "1: \n"
          "fmul v0.4s, v0.4s, %[alpha].4s\n"
          "fmul v1.4s, v1.4s, %[alpha].4s\n"
          "fmul v2.4s, v2.4s, %[alpha].4s\n"
          "fmul v3.4s, v3.4s, %[alpha].4s\n"
          "0: \n"
          "stp q0, q1, [%[outptr]], #32\n"
          "stp q2, q3, [%[outptr]], #32\n"
          : [outptr] "+r"(ptr_out),
            [ptr0] "+r"(ptr0),
            [ptr1] "+r"(ptr1),
            [ptr2] "+r"(ptr2),
            [ptr3] "+r"(ptr3)
          : [has_alpha] "r"(has_alpha), [alpha] "w"(valpha)
          : "v0", "v1", "v2", "v3", "cc", "memory");
      // clang-format on
      outptr_row_col += stride_out;
    }
    if (right_pad > 0) {
      float *ptr_out = outptr_row_col;
      // clang-format off
      asm volatile(
          "cmp %w[has_alpha], #0\n"
          "ld1 {v0.4s}, [%[ptr0]], #16\n"
          "ld1 {v1.4s}, [%[ptr1]], #16\n"
          "ld1 {v2.4s}, [%[ptr2]], #16\n"
          "ld1 {v3.4s}, [%[ptr3]], #16\n"
          "beq  0f\n"
          "1: \n"
          "fmul v0.4s, v0.4s, %[alpha].4s\n"
          "fmul v1.4s, v1.4s, %[alpha].4s\n"
          "fmul v2.4s, v2.4s, %[alpha].4s\n"
          "fmul v3.4s, v3.4s, %[alpha].4s\n"
          "0: \n"
          "bif v0.16b, %[vzero].16b, %[vmask1].16b\n"
          "bif v1.16b, %[vzero].16b, %[vmask1].16b\n"
          "bif v2.16b, %[vzero].16b, %[vmask1].16b\n"
          "bif v3.16b, %[vzero].16b, %[vmask1].16b\n"
          "stp q0, q1, [%[outptr]], #32\n"
          "stp q2, q3, [%[outptr]], #32\n"
          : [outptr] "+r"(ptr_out),
            [ptr0] "+r"(ptr0),
            [ptr1] "+r"(ptr1),
            [ptr2] "+r"(ptr2),
            [ptr3] "+r"(ptr3)
          : [vmask1] "w"(vmask1),
            [vzero] "w"(vzero),
            [has_alpha] "r"(has_alpha),
            [alpha] "w"(valpha)
          : "v0", "v1", "v2", "v3", "cc", "memory");
      // clang-format on
    }
  }
#pragma omp parallel for
  for (int y = 4 * (y_len / 4); y < y_len; ++y) {
    const float *ptr0 = inptr + y * ldin;
    float *outptr_row_col = outptr + y * 4;
    int i = 0;
    for (; i < x_len - 3; i += 4) {
      float *ptr_out = outptr_row_col;
      asm volatile(
          "cmp %w[has_alpha], #0\n"
          "ld1 {v0.4s}, [%[ptr0]], #16\n"
          "beq  0f\n"
          "1: \n"
          "fmul v0.4s, v0.4s, %[alpha].4s\n"
          "0: \n"
          "st1 {v0.4s}, [%[outptr]], #16\n"
          : [ptr0] "+r"(ptr0), [outptr] "+r"(ptr_out)
          : [has_alpha] "r"(has_alpha), [alpha] "w"(valpha)
          : "v0", "v1", "cc", "memory");
      outptr_row_col += stride_out;
    }
    if (right_pad > 0) {
      float *ptr_out = outptr_row_col;
      asm volatile(
          "cmp %w[has_alpha], #0\n"
          "ld1 {v0.4s}, [%[ptr0]], #16\n"
          "beq  0f\n"
          "1: \n"
          "fmul v0.4s, v0.4s, %[alpha].4s\n"
          "0: \n"
          "bif v0.16b, %[vzero].16b, %[vmask1].16b\n"
          "st1 {v0.4s}, [%[outptr]], #16\n"
          : [ptr0] "+r"(ptr0), [outptr] "+r"(ptr_out)
          : [vmask1] "w"(vmask1),
            [vzero] "w"(vzero),
            [has_alpha] "r"(has_alpha),
            [alpha] "w"(valpha)
          : "v0", "v1", "cc", "memory");
    }
  }
}

void pack_trans_m4(float *outptr,
                   const float *in,
                   float alpha,
                   int ldin,
                   int m0,
                   int mmax,
                   int k0,
                   int kmax) {
  auto inptr = in + k0 * ldin + m0;
  uint32_t mask_buffer[4] = {0, 1, 2, 3};
  int x_len = mmax - m0;
  int y_len = kmax - k0;
  int right_remain = x_len - 4 * (x_len / 4);
  int stride_out = 4 * y_len;

  float32x4_t vzero = vdupq_n_f32(0.f);
  uint32x4_t vmask1 =
      vcltq_u32(vld1q_u32(mask_buffer), vdupq_n_u32(right_remain));

  bool has_alpha = fabsf(alpha - 1.f) > 1e-8f;
  float32x4_t valpha = vdupq_n_f32(alpha);

#pragma omp parallel for
  for (int y = 0; y < y_len - 3; y += 4) {
    const float *ptr0 = inptr + y * ldin;
    const float *ptr1 = ptr0 + ldin;
    const float *ptr2 = ptr1 + ldin;
    const float *ptr3 = ptr2 + ldin;

    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]        \n"
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

    float *outptr_row_col = outptr + y * 4;
    int i = 0;
    for (; i < x_len - 3; i += 4) {
      float32x4_t vr00 = vld1q_f32(ptr0);
      float32x4_t vr10 = vld1q_f32(ptr1);
      float32x4_t vr20 = vld1q_f32(ptr2);
      float32x4_t vr30 = vld1q_f32(ptr3);
      if (has_alpha) {
        vr00 = vmulq_f32(vr00, valpha);
        vr10 = vmulq_f32(vr10, valpha);
        vr20 = vmulq_f32(vr20, valpha);
        vr30 = vmulq_f32(vr30, valpha);
      }

      vst1q_f32(outptr_row_col, vr00);
      vst1q_f32(outptr_row_col + 4, vr10);
      vst1q_f32(outptr_row_col + 8, vr20);
      vst1q_f32(outptr_row_col + 12, vr30);

      ptr0 += 4;
      ptr1 += 4;
      ptr2 += 4;
      ptr3 += 4;

      outptr_row_col += stride_out;
    }
    if (right_remain > 0) {
      float32x4_t vr00 = vld1q_f32(ptr0);
      float32x4_t vr10 = vld1q_f32(ptr1);
      float32x4_t vr20 = vld1q_f32(ptr2);
      float32x4_t vr30 = vld1q_f32(ptr3);

      if (has_alpha) {
        vr00 = vmulq_f32(vr00, valpha);
        vr10 = vmulq_f32(vr10, valpha);
        vr20 = vmulq_f32(vr20, valpha);
        vr30 = vmulq_f32(vr30, valpha);
      }

      float32x4_t vr00_1 = vbslq_f32(vmask1, vr00, vzero);
      float32x4_t vr10_1 = vbslq_f32(vmask1, vr10, vzero);
      float32x4_t vr20_1 = vbslq_f32(vmask1, vr20, vzero);
      float32x4_t vr30_1 = vbslq_f32(vmask1, vr30, vzero);

      vst1q_f32(outptr_row_col, vr00_1);
      vst1q_f32(outptr_row_col + 4, vr10_1);
      vst1q_f32(outptr_row_col + 8, vr20_1);
      vst1q_f32(outptr_row_col + 12, vr30_1);
    }
  }

#pragma omp parallel for
  for (int y = 4 * (y_len / 4); y < y_len; ++y) {
    const float *ptr0 = inptr + y * ldin;
    float *outptr_row_col = outptr + y * 4;
    int i = 0;
    for (; i < x_len - 3; i += 4) {
      float32x4_t vr0 = vld1q_f32(ptr0);
      if (has_alpha) {
        vr0 = vmulq_f32(vr0, valpha);
      }
      vst1q_f32(outptr_row_col, vr0);

      ptr0 += 4;

      outptr_row_col += stride_out;
    }
    if (right_remain > 0) {
      float32x4_t vr0 = vld1q_f32(ptr0);

      if (has_alpha) {
        vr0 = vmulq_f32(vr0, valpha);
      }

      float32x4_t vr0_1 = vbslq_f32(vmask1, vr0, vzero);

      vst1q_f32(outptr_row_col, vr0_1);
    }
  }
}

#else  // __aarch64__
void prepackA_6x8(float* outptr,
                  const float* inptr,
                  float alpha,
                  int ldin,
                  int m0,
                  int mmax,
                  int k0,
                  int kmax) {
  int x_len = kmax - k0;
  float zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(float) * x_len);

  bool has_alpha = fabsf(alpha - 1.f) > 1e-8f;
  float32x4_t valpha = vdupq_n_f32(alpha);

  for (int y = m0; y < mmax; y += 6) {
    const float* inptr0 = inptr + y * ldin + k0;
    const float* inptr1 = inptr0 + ldin;
    const float* inptr2 = inptr1 + ldin;
    const float* inptr3 = inptr2 + ldin;
    const float* inptr4 = inptr3 + ldin;
    const float* inptr5 = inptr4 + ldin;

    int x = x_len;
    if ((y + 5) >= mmax) {
      switch ((y + 5) - mmax) {
        case 4:
          inptr1 = zerobuff;
        case 3:
          inptr2 = zerobuff;
        case 2:
          inptr3 = zerobuff;
        case 1:
          inptr4 = zerobuff;
        case 0:
          inptr5 = zerobuff;
        default:
          break;
      }
    }

    for (; x > 7; x -= 8) {
      asm volatile(
          "vld4.32  {d0-d3}, [%[inptr0]]!   @ zip load r0, "
          "q0,q1=r00,r04,r01,r05,r02,r06,r03,r07\n"
          "vld4.32  {d4-d7}, [%[inptr1]]!   @ zip load r1, "
          "q2,q3=r10,r14,r11,r15,r12,r16,r13,r17\n"
          "vld4.32  {d8-d11}, [%[inptr2]]!  @ zip load r2, "
          "q4,q5=r20,r24,r21,r25,r22,r26,r23,r27\n"
          "vld4.32  {d12-d15}, [%[inptr3]]! @ zip load r3, "
          "q6,q7=r30,r34,r31,r35,r32,r36,r33,r37\n"
          "vld4.32  {d16-d19}, [%[inptr4]]! @ zip load r4, "
          "q8,q9=r40,r44,r41,r45,r42,r46,r43,r47\n"
          "vld4.32  {d20-d23}, [%[inptr5]]! @ zip load r5, "
          "q10,q11=r50,r54,r51,r55,r52,r56,r53,r57\n"
          "cmp %[has_alpha], #0\n"
          "beq  0f\n"                        /* check whether alpha == 1? */
          "vmul.f32   q0, q0, %q[alpha]\n"   /* mul alpha */
          "vmul.f32   q1, q1, %q[alpha]\n"   /* mul alpha */
          "vmul.f32   q2, q2, %q[alpha]\n"   /* mul alpha */
          "vmul.f32   q3, q3, %q[alpha]\n"   /* mul alpha */
          "vmul.f32   q4, q4, %q[alpha]\n"   /* mul alpha */
          "vmul.f32   q5, q5, %q[alpha]\n"   /* mul alpha */
          "vmul.f32   q6, q6, %q[alpha]\n"   /* mul alpha */
          "vmul.f32   q7, q7, %q[alpha]\n"   /* mul alpha */
          "vmul.f32   q8, q8, %q[alpha]\n"   /* mul alpha */
          "vmul.f32   q9, q9, %q[alpha]\n"   /* mul alpha */
          "vmul.f32   q10, q10, %q[alpha]\n" /* mul alpha */
          "vmul.f32   q11, q11, %q[alpha]\n" /* mul alpha */
          "0: \n"
          "vtrn.32  q0, q2                  @ trans data: q0=r00,r10,r01,r11; "
          "q2=r04,r14,r05,r15\n"
          "vtrn.32  q4, q6                  @ trans data: q4=r20,r30,r21,r31; "
          "q6=r24,r34,r25,r35\n"
          "vtrn.32  q8, q10                 @ trans data: q8=r40,r50,r41,r51; "
          "q10=r44,r54,r45,r55\n"

          "vswp     d1, d8                  @ swap d1, d8, q0=r00,r10,r20,r30; "
          "q4=r01,r11,r21,r31\n"
          "vst1.32  {d0-d1},  [%[outptr]]!  @ write q0:r00,r10,r20,r30\n"
          "vst1.32  {d16},    [%[outptr]]!  @ write d16(q8,low),r40,r50\n"
          "vst1.32  {d8-d9},  [%[outptr]]!  @ write q4:r01,r11,r21,r31\n"
          "vst1.32  {d17},    [%[outptr]]!  @ write d16(q8,high),r41,r51\n"

          "vtrn.32  q1, q3                  @ trans data: q1=r02,r12,r03,r13; "
          "q3=r06,r16,r07,r17\n"
          "vtrn.32  q5, q7                  @ trans data: q5=r22,r32,r23,r33; "
          "q7=r26,r36,r27,r37\n"
          "vtrn.32  q9, q11                 @ trans data: q9=r42,r52,r43,r53; "
          "q11=r46,r56,r47,r57\n"

          "vswp     d3, d10                 @ swap d3, d10, "
          "q1=r02,r12,r22,r32; q5=r03,r13,r23,r33\n"
          "vst1.32  {d2-d3},  [%[outptr]]!  @ write q1:r02,r12,r22,r32\n"
          "vst1.32  {d18},    [%[outptr]]!  @ write d18(q9,low),r42,r52\n"
          "vst1.32  {d10-d11},[%[outptr]]!  @ write q5:r03,r13,r23,r33\n"
          "vst1.32  {d19},    [%[outptr]]!  @ write d19(q9,high),r43,r53\n"

          "vswp     d5, d12                 @ swap d5, d12,q2=r04,r14,r24,r34; "
          "q6=r05,r15,r25,r35\n"
          "vst1.32  {d4-d5},  [%[outptr]]!  @ write q2:r04,r14,r24,r34\n"
          "vst1.32  {d20},    [%[outptr]]!  @ write d20(q10,low),r44,r54\n"
          "vst1.32  {d12-d13},[%[outptr]]!  @ write q6:r05,r15,r25,r35\n"
          "vst1.32  {d21},    [%[outptr]]!  @ write d21(q10,high),r45,r55\n"

          "vswp     d7, d14                 @ swap d7, d14, "
          "q3=r06,r16,r26,r36; q7=r07,r17,r27,r37\n"
          "vst1.32  {d6-d7},  [%[outptr]]!  @ write q3:r06,r16,r26,r36\n"
          "vst1.32  {d22},    [%[outptr]]!  @ write d22(q11,low),r46,r56\n"
          "vst1.32  {d14-d15},[%[outptr]]!  @ write q7:r07,r17,r27,r37\n"
          "vst1.32  {d23},    [%[outptr]]!  @ write d23(q11,high),r47,r57\n"
          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [inptr4] "+r"(inptr4),
            [inptr5] "+r"(inptr5),
            [outptr] "+r"(outptr)
          : [has_alpha] "r"(has_alpha), [alpha] "w"(valpha)
          : "q0",
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "q7",
            "q8",
            "q9",
            "q10",
            "q11",
            "q15",
            "cc",
            "memory");
    }

    for (; x > 0; x--) {
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

void prepackA_trans_6x8(float* outptr,
                        const float* in,
                        float alpha,
                        int ldin,
                        int m0,
                        int mmax,
                        int k0,
                        int kmax) {
  auto inptr = in + k0 * ldin + m0;

  bool has_alpha = fabsf(alpha - 1.f) > 1e-8f;
  float32x4_t valpha = vdupq_n_f32(alpha);

  uint32_t mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int x_len = mmax - m0;
  int y_len = kmax - k0;
  int right_remain = x_len - 6 * (x_len / 6);
  int right_pad = 6 - right_remain;
  if (right_remain == 0) {
    right_pad = 0;
  }

  float* outptr_row = outptr;
  int stride_out = 6 * y_len;

  float32x4_t vzero = vdupq_n_f32(0.f);
  uint32x4_t vmask1 =
      vcltq_u32(vld1q_u32(mask_buffer), vdupq_n_u32(right_remain));
  uint32x4_t vmask2 =
      vcltq_u32(vld1q_u32(mask_buffer + 4), vdupq_n_u32(right_remain));

#pragma omp parallel for
  for (int y = 0; y < y_len - 3; y += 4) {
    const float* ptr0 = inptr + y * ldin;
    const float* ptr1 = ptr0 + ldin;
    const float* ptr2 = ptr1 + ldin;
    const float* ptr3 = ptr2 + ldin;

    float* outptr_row_col = outptr_row + y * 6;
    int i = 0;
    for (; i < x_len - 5; i += 6) {
      float* ptr_out = outptr_row_col;
      asm volatile(
          "vld1.32 {d0-d2}, [%[ptr0]]!        @ load r0, 6 elements\n"
          "vld1.32 {d4-d6}, [%[ptr1]]!        @ load r1, 6 elements\n"
          "vld1.32 {d8-d10}, [%[ptr2]]!       @ load r2, 6 elements\n"
          "vld1.32 {d12-d14}, [%[ptr3]]!      @ load r3, 6 elements\n"
          "cmp %[has_alpha], #0\n"
          "beq  0f\n"                        /* check whether alpha == 1? */
          "vmul.f32   q0, q0, %q[alpha]\n"   /* mul alpha */
          "vmul.f32   d2, d2, %e[alpha]\n"   /* mul alpha */
          "vmul.f32   q2, q2, %q[alpha]\n"   /* mul alpha */
          "vmul.f32   d6, d6, %e[alpha]\n"   /* mul alpha */
          "vmul.f32   q4, q4, %q[alpha]\n"   /* mul alpha */
          "vmul.f32   d10, d10, %e[alpha]\n" /* mul alpha */
          "vmul.f32   q6, q6, %q[alpha]\n"   /* mul alpha */
          "vmul.f32   d14, d14, %e[alpha]\n" /* mul alpha */
          "0: \n"
          "vst1.32 {d0-d2}, [%[outptr]]!      @ write to output ptr\n"
          "vst1.32 {d4-d6}, [%[outptr]]!      @ write to output ptr\n"
          "vst1.32 {d8-d10}, [%[outptr]]!     @ write to output ptr\n"
          "vst1.32 {d12-d14}, [%[outptr]]!    @ write to output ptr\n"
          : [outptr] "+r"(ptr_out),
            [ptr0] "+r"(ptr0),
            [ptr1] "+r"(ptr1),
            [ptr2] "+r"(ptr2),
            [ptr3] "+r"(ptr3)
          : [has_alpha] "r"(has_alpha), [alpha] "w"(valpha)
          : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "cc", "memory");
      outptr_row_col += stride_out;
    }
    if (right_pad > 0) {
      float* ptr_out = outptr_row_col;
      asm volatile(
          "vld1.32 {d0-d2}, [%[ptr0]]!        @ load r0, 6 elements\n"
          "vld1.32 {d4-d6}, [%[ptr1]]!        @ load r1, 6 elements\n"
          "vld1.32 {d8-d10}, [%[ptr2]]!       @ load r2, 8 elements\n"
          "vld1.32 {d12-d14}, [%[ptr3]]!      @ load r3, 8 elements\n"
          "cmp %[has_alpha], #0\n"
          "beq  0f\n"                        /* check whether alpha == 1? */
          "vmul.f32   q0, q0, %q[alpha]\n"   /* mul alpha */
          "vmul.f32   d2, d2, %e[alpha]\n"   /* mul alpha */
          "vmul.f32   q2, q2, %q[alpha]\n"   /* mul alpha */
          "vmul.f32   d6, d6, %e[alpha]\n"   /* mul alpha */
          "vmul.f32   q4, q4, %q[alpha]\n"   /* mul alpha */
          "vmul.f32   d10, d10, %e[alpha]\n" /* mul alpha */
          "vmul.f32   q6, q6, %q[alpha]\n"   /* mul alpha */
          "vmul.f32   d14, d14, %e[alpha]\n" /* mul alpha */
          "0: \n"
          "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
          "vbif   d2, %e[vzero], %e[vmask2]   @ bit select, pad zero\n"
          "vbif   q2, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
          "vbif   d6, %e[vzero], %e[vmask2]   @ bit select, pad zero\n"
          "vst1.32 {d0-d2}, [%[outptr]]!      @ write to output ptr\n"
          "vst1.32 {d4-d6}, [%[outptr]]!      @ write to output ptr\n"
          "vbif   q4, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
          "vbif   d10, %e[vzero], %e[vmask2]  @ bit select, pad zero\n"
          "vbif   q6, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
          "vbif   d14, %e[vzero], %e[vmask2]  @ bit select, pad zero\n"
          "vst1.32 {d8-d10}, [%[outptr]]!     @ write to output ptr\n"
          "vst1.32 {d12-d14}, [%[outptr]]!    @ write to output ptr\n"
          : [outptr] "+r"(ptr_out),
            [ptr0] "+r"(ptr0),
            [ptr1] "+r"(ptr1),
            [ptr2] "+r"(ptr2),
            [ptr3] "+r"(ptr3)
          : [vmask1] "w"(vmask1),
            [vmask2] "w"(vmask2),
            [vzero] "w"(vzero),
            [has_alpha] "r"(has_alpha),
            [alpha] "w"(valpha)
          : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "cc", "memory");
    }
  }

#pragma omp parallel for
  for (int y = 4 * (y_len / 4); y < y_len; ++y) {
    const float* ptr0 = inptr + y * ldin;
    float* outptr_row_col = outptr_row + y * 6;
    int i = 0;
    for (; i < x_len - 5; i += 6) {
      float* ptr_out = outptr_row_col;
      asm volatile(
          "vld1.32 {d0-d2}, [%[ptr0]]!        @ load r0, 6 elements\n"
          "cmp %[has_alpha], #0\n"
          "beq  0f\n"                      /* check whether alpha == 1? */
          "vmul.f32   q0, q0, %q[alpha]\n" /* mul alpha */
          "vmul.f32   d2, d2, %e[alpha]\n" /* mul alpha */
          "0: \n"
          "vst1.32 {d0-d2}, [%[outptr]]!      @ write to output ptr\n"
          : [ptr0] "+r"(ptr0), [outptr] "+r"(ptr_out)
          : [has_alpha] "r"(has_alpha), [alpha] "w"(valpha)
          : "q0", "q1", "cc", "memory");
      outptr_row_col += stride_out;
    }
    if (right_pad > 0) {
      float* ptr_out = outptr_row_col;
      asm volatile(
          "vld1.32 {d0-d2}, [%[ptr0]]!        @ load r0, 6 elements\n"
          "cmp %[has_alpha], #0\n"
          "beq  0f\n"                      /* check whether alpha == 1? */
          "vmul.f32   q0, q0, %q[alpha]\n" /* mul alpha */
          "vmul.f32   d2, d2, %e[alpha]\n" /* mul alpha */
          "0: \n"
          "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
          "vbif   d2, %e[vzero], %e[vmask2]   @ bit select, pad zero\n"
          "vst1.32 {d0-d2}, [%[outptr]]!      @ write to output ptr\n"
          : [ptr0] "+r"(ptr0), [outptr] "+r"(ptr_out)
          : [vmask1] "w"(vmask1),
            [vmask2] "w"(vmask2),
            [vzero] "w"(vzero),
            [has_alpha] "r"(has_alpha),
            [alpha] "w"(valpha)
          : "q0", "q1", "cc", "memory");
    }
  }
}

void prepackA_4x8(float* outptr,
                  const float* inptr,
                  float alpha,
                  int ldin,
                  int m0,
                  int mmax,
                  int k0,
                  int kmax) {
  int x_len = kmax - k0;
  float zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(float) * x_len);

  bool has_alpha = fabsf(alpha - 1.f) > 1e-8f;
  float32x4_t valpha = vdupq_n_f32(alpha);

  for (int y = m0; y < mmax; y += 4) {
    const float* inptr0 = inptr + y * ldin + k0;
    const float* inptr1 = inptr0 + ldin;
    const float* inptr2 = inptr1 + ldin;
    const float* inptr3 = inptr2 + ldin;

    int x = x_len;
    if ((y + 3) >= mmax) {
      switch ((y + 3) - mmax) {
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

    for (; x > 7; x -= 8) {
      asm volatile(
          "vld4.32  {d0-d3}, [%[inptr0]]!   @ zip load r0, "
          "q0,q1=r00,r04,r01,r05,r02,r06,r03,r07\n"
          "vld4.32  {d4-d7}, [%[inptr1]]!   @ zip load r1, "
          "q2,q3=r10,r14,r11,r15,r12,r16,r13,r17\n"
          "vld4.32  {d8-d11}, [%[inptr2]]!  @ zip load r2, "
          "q4,q5=r20,r24,r21,r25,r22,r26,r23,r27\n"
          "vld4.32  {d12-d15}, [%[inptr3]]! @ zip load r3, "
          "q6,q7=r30,r34,r31,r35,r32,r36,r33,r37\n"
          "cmp %[has_alpha], #0\n"
          "beq  0f\n"                      /* check whether alpha == 1? */
          "vmul.f32   q0, q0, %q[alpha]\n" /* mul alpha */
          "vmul.f32   q1, q1, %q[alpha]\n" /* mul alpha */
          "vmul.f32   q2, q2, %q[alpha]\n" /* mul alpha */
          "vmul.f32   q3, q3, %q[alpha]\n" /* mul alpha */
          "vmul.f32   q4, q4, %q[alpha]\n" /* mul alpha */
          "vmul.f32   q5, q5, %q[alpha]\n" /* mul alpha */
          "vmul.f32   q6, q6, %q[alpha]\n" /* mul alpha */
          "vmul.f32   q7, q7, %q[alpha]\n" /* mul alpha */
          "0: \n"
          "vtrn.32  q0, q2                  @ trans data: q0=r00,r10,r01,r11; "
          "q2=r04,r14,r05,r15\n"
          "vtrn.32  q4, q6                  @ trans data: q4=r20,r30,r21,r31; "
          "q6=r24,r34,r25,r35\n"

          "vswp     d1, d8                  @ swap d1, d8, q0=r00,r10,r20,r30; "
          "q4=r01,r11,r21,r31\n"
          "vst1.32  {d0-d1},  [%[outptr]]!  @ write q0:r00,r10,r20,r30\n"
          "vst1.32  {d8-d9},  [%[outptr]]!  @ write q4:r01,r11,r21,r31\n"

          "vtrn.32  q1, q3                  @ trans data: q1=r02,r12,r03,r13; "
          "q3=r06,r16,r07,r17\n"
          "vtrn.32  q5, q7                  @ trans data: q5=r22,r32,r23,r33; "
          "q7=r26,r36,r27,r37\n"

          "vswp     d3, d10                 @ swap d3, d10, "
          "q1=r02,r12,r22,r32; q5=r03,r13,r23,r33\n"
          "vst1.32  {d2-d3},  [%[outptr]]!  @ write q1:r02,r12,r22,r32\n"
          "vst1.32  {d10-d11},[%[outptr]]!  @ write q5:r03,r13,r23,r33\n"

          "vswp     d5, d12                 @ swap d5, d12,q2=r04,r14,r24,r34; "
          "q6=r05,r15,r25,r35\n"
          "vst1.32  {d4-d5},  [%[outptr]]!  @ write q2:r04,r14,r24,r34\n"
          "vst1.32  {d12-d13},[%[outptr]]!  @ write q6:r05,r15,r25,r35\n"

          "vswp     d7, d14                 @ swap d7, d14, "
          "q3=r06,r16,r26,r36; q7=r07,r17,r27,r37\n"
          "vst1.32  {d6-d7},  [%[outptr]]!  @ write q3:r06,r16,r26,r36\n"
          "vst1.32  {d14-d15},[%[outptr]]!  @ write q7:r07,r17,r27,r37\n"
          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outptr] "+r"(outptr)
          : [has_alpha] "r"(has_alpha), [alpha] "w"(valpha)
          : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "cc", "memory");
    }

    for (; x > 0; x--) {
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

void prepackA_trans_4x8(float* outptr,
                        const float* in,
                        float alpha,
                        int ldin,
                        int m0,
                        int mmax,
                        int k0,
                        int kmax) {
  auto inptr = in + k0 * ldin + m0;
  bool has_alpha = fabsf(alpha - 1.f) > 1e-8f;
  float32x4_t valpha = vdupq_n_f32(alpha);

  uint32_t mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int x_len = mmax - m0;
  int y_len = kmax - k0;
  int right_remain = x_len - 4 * (x_len / 4);
  int right_pad = 4 - right_remain;
  if (right_remain == 0) {
    right_pad = 0;
  }

  int stride_out = 4 * y_len;

  float32x4_t vzero = vdupq_n_f32(0.f);
  uint32x4_t vmask1 =
      vcltq_u32(vld1q_u32(mask_buffer), vdupq_n_u32(right_remain));

#pragma omp parallel for
  for (int y = 0; y < y_len - 3; y += 4) {
    const float* ptr0 = inptr + y * ldin;
    const float* ptr1 = ptr0 + ldin;
    const float* ptr2 = ptr1 + ldin;
    const float* ptr3 = ptr2 + ldin;

    float* outptr_row_col = outptr + y * 4;
    int i = 0;
    for (; i < x_len - 3; i += 4) {
      float* ptr_out = outptr_row_col;
      asm volatile(
          "vld1.32 {d0-d1}, [%[ptr0]]!        @ load r0, 4 elements\n"
          "vld1.32 {d2-d3}, [%[ptr1]]!        @ load r1, 4 elements\n"
          "vld1.32 {d4-d5}, [%[ptr2]]!        @ load r2, 4 elements\n"
          "vld1.32 {d6-d7}, [%[ptr3]]!        @ load r3, 4 elements\n"
          "cmp %[has_alpha], #0\n"
          "beq  0f\n"                      /* check whether alpha == 1? */
          "vmul.f32   q0, q0, %q[alpha]\n" /* mul alpha */
          "vmul.f32   q1, q1, %q[alpha]\n" /* mul alpha */
          "vmul.f32   q2, q2, %q[alpha]\n" /* mul alpha */
          "vmul.f32   q3, q3, %q[alpha]\n" /* mul alpha */
          "0: \n"
          "vst1.32 {d0-d1}, [%[outptr]]!      @ write to output ptr\n"
          "vst1.32 {d2-d3}, [%[outptr]]!      @ write to output ptr\n"
          "vst1.32 {d4-d5}, [%[outptr]]!      @ write to output ptr\n"
          "vst1.32 {d6-d7}, [%[outptr]]!      @ write to output ptr\n"
          : [outptr] "+r"(ptr_out),
            [ptr0] "+r"(ptr0),
            [ptr1] "+r"(ptr1),
            [ptr2] "+r"(ptr2),
            [ptr3] "+r"(ptr3)
          : [has_alpha] "r"(has_alpha), [alpha] "w"(valpha)
          : "q0", "q1", "q2", "q3", "cc", "memory");
      outptr_row_col += stride_out;
    }
    if (right_pad > 0) {
      float* ptr_out = outptr_row_col;
      asm volatile(
          "vld1.32 {d0-d1}, [%[ptr0]]!        @ load r0, 4 elements\n"
          "vld1.32 {d2-d3}, [%[ptr1]]!        @ load r1, 4 elements\n"
          "vld1.32 {d4-d5}, [%[ptr2]]!        @ load r2, 4 elements\n"
          "vld1.32 {d6-d7}, [%[ptr3]]!        @ load r3, 4 elements\n"
          "cmp %[has_alpha], #0\n"
          "beq  0f\n"                      /* check whether alpha == 1? */
          "vmul.f32   q0, q0, %q[alpha]\n" /* mul alpha */
          "vmul.f32   q1, q1, %q[alpha]\n" /* mul alpha */
          "vmul.f32   q2, q2, %q[alpha]\n" /* mul alpha */
          "vmul.f32   q3, q3, %q[alpha]\n" /* mul alpha */
          "0: \n"
          "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
          "vbif   q1, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
          "vbif   q2, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
          "vbif   q3, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
          "vst1.32 {d0-d1}, [%[outptr]]!      @ write to output ptr\n"
          "vst1.32 {d2-d3}, [%[outptr]]!      @ write to output ptr\n"
          "vst1.32 {d4-d5}, [%[outptr]]!      @ write to output ptr\n"
          "vst1.32 {d6-d7}, [%[outptr]]!      @ write to output ptr\n"
          : [outptr] "+r"(ptr_out),
            [ptr0] "+r"(ptr0),
            [ptr1] "+r"(ptr1),
            [ptr2] "+r"(ptr2),
            [ptr3] "+r"(ptr3)
          : [vmask1] "w"(vmask1),
            [vzero] "w"(vzero),
            [has_alpha] "r"(has_alpha),
            [alpha] "w"(valpha)
          : "q0", "q1", "q2", "q3", "cc", "memory");
    }
  }

#pragma omp parallel for
  for (int y = 4 * (y_len / 4); y < y_len; ++y) {
    const float* ptr0 = inptr + y * ldin;
    float* outptr_row_col = outptr + y * 4;
    int i = 0;
    for (; i < x_len - 3; i += 4) {
      float* ptr_out = outptr_row_col;
      // clang-format off
      asm volatile(
          "vld1.32 {d0-d1}, [%[ptr0]]!        @ load r0, 4 elements\n"
          "cmp %[has_alpha], #0\n"
          "beq  0f\n"                      /* check whether alpha == 1? */
          "vmul.f32   q0, q0, %q[alpha]\n" /* mul alpha */
          "0: \n"
          "vst1.32 {d0-d1}, [%[outptr]]!      @ write to output ptr\n"
          : [ptr0] "+r"(ptr0), [outptr] "+r"(ptr_out)
          : [has_alpha] "r"(has_alpha), [alpha] "w"(valpha)
          : "q0", "q1", "cc", "memory");
      // clang-format on
      outptr_row_col += stride_out;
    }
    if (right_pad > 0) {
      float* ptr_out = outptr_row_col;
      // clang-format off
      asm volatile(
          "vld1.32 {d0-d1}, [%[ptr0]]!        @ load r0, 4 elements\n"
          "cmp %[has_alpha], #0\n"
          "beq  0f\n"                      /* check whether alpha == 1? */
          "vmul.f32   q0, q0, %q[alpha]\n" /* mul alpha */
          "0: \n"
          "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
          "vst1.32 {d0-d1}, [%[outptr]]!      @ write to output ptr\n"
          : [ptr0] "+r"(ptr0), [outptr] "+r"(ptr_out)
          : [vmask1] "w"(vmask1),
            [vzero] "w"(vzero),
            [has_alpha] "r"(has_alpha),
            [alpha] "w"(valpha)
          : "q0", "q1", "cc", "memory");
      // clang-format on
    }
  }
}

#endif  // __aarch64__

/**
* \brief input data is transpose
* for arm-v7a, transform data to block x k x 8 layout
* for arm-v8a, transform data to block x k x 12 layout or block x k x 8 layout
*/
#ifdef __aarch64__
void loadb(
    float *out, const float *in, int ldin, int k0, int kmax, int n0, int nmax) {
  auto outptr = reinterpret_cast<uint32_t *>(out);
  auto inptr = reinterpret_cast<const uint32_t *>(in) + k0 * ldin + n0;
  uint32_t mask_buffer[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int x_len = nmax - n0;
  int y_len = kmax - k0;
  int right_remain = x_len - 12 * (x_len / 12);
  int right_pad = 12 - right_remain;

  uint32_t *outptr_row = outptr;
  int stride_out = 12 * y_len;
  int cnt_y = 4 * (y_len / 4);

  uint32x4_t vzero = vdupq_n_u32(0);
  uint32x4_t vmask1 =
      vcltq_u32(vld1q_u32(mask_buffer), vdupq_n_u32(right_remain));
  uint32x4_t vmask2 =
      vcltq_u32(vld1q_u32(mask_buffer + 4), vdupq_n_u32(right_remain));
  uint32x4_t vmask3 =
      vcltq_u32(vld1q_u32(mask_buffer + 8), vdupq_n_u32(right_remain));

#pragma omp parallel for
  for (int y = 0; y < y_len - 3; y += 4) {
    const uint32_t *ptr0 = inptr + y * ldin;
    const uint32_t *ptr1 = ptr0 + ldin;
    const uint32_t *ptr2 = ptr1 + ldin;
    const uint32_t *ptr3 = ptr2 + ldin;
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

    uint32_t *outptr_row_col = outptr_row + y * 12;

    int i = 0;
    for (; i < x_len - 11; i += 12) {
      uint32x4_t vr00 = vld1q_u32(ptr0);
      uint32x4_t vr01 = vld1q_u32(ptr0 + 4);
      uint32x4_t vr02 = vld1q_u32(ptr0 + 8);

      uint32x4_t vr10 = vld1q_u32(ptr1);
      uint32x4_t vr11 = vld1q_u32(ptr1 + 4);
      uint32x4_t vr12 = vld1q_u32(ptr1 + 8);

      vst1q_u32(outptr_row_col, vr00);
      vst1q_u32(outptr_row_col + 4, vr01);
      vst1q_u32(outptr_row_col + 8, vr02);

      uint32x4_t vr20 = vld1q_u32(ptr2);
      uint32x4_t vr21 = vld1q_u32(ptr2 + 4);
      uint32x4_t vr22 = vld1q_u32(ptr2 + 8);

      vst1q_u32(outptr_row_col + 12, vr10);
      vst1q_u32(outptr_row_col + 16, vr11);
      vst1q_u32(outptr_row_col + 20, vr12);

      uint32x4_t vr30 = vld1q_u32(ptr3);
      uint32x4_t vr31 = vld1q_u32(ptr3 + 4);
      uint32x4_t vr32 = vld1q_u32(ptr3 + 8);

      vst1q_u32(outptr_row_col + 24, vr20);
      vst1q_u32(outptr_row_col + 28, vr21);
      vst1q_u32(outptr_row_col + 32, vr22);

      vst1q_u32(outptr_row_col + 36, vr30);
      vst1q_u32(outptr_row_col + 40, vr31);
      vst1q_u32(outptr_row_col + 44, vr32);

      ptr0 += 12;
      ptr1 += 12;
      ptr2 += 12;
      ptr3 += 12;

      outptr_row_col += stride_out;
    }
    if (right_remain > 0) {
      uint32x4_t vr00 = vld1q_u32(ptr0);
      uint32x4_t vr01 = vld1q_u32(ptr0 + 4);
      uint32x4_t vr02 = vld1q_u32(ptr0 + 8);

      uint32x4_t vr10 = vld1q_u32(ptr1);
      uint32x4_t vr11 = vld1q_u32(ptr1 + 4);
      uint32x4_t vr12 = vld1q_u32(ptr1 + 8);

      uint32x4_t vr00_1 = vbslq_u32(vmask1, vr00, vzero);
      uint32x4_t vr01_1 = vbslq_u32(vmask2, vr01, vzero);
      uint32x4_t vr02_1 = vbslq_u32(vmask3, vr02, vzero);

      uint32x4_t vr20 = vld1q_u32(ptr2);
      uint32x4_t vr21 = vld1q_u32(ptr2 + 4);
      uint32x4_t vr22 = vld1q_u32(ptr2 + 8);

      vst1q_u32(outptr_row_col, vr00_1);
      vst1q_u32(outptr_row_col + 4, vr01_1);
      vst1q_u32(outptr_row_col + 8, vr02_1);

      uint32x4_t vr10_1 = vbslq_u32(vmask1, vr10, vzero);
      uint32x4_t vr11_1 = vbslq_u32(vmask2, vr11, vzero);
      uint32x4_t vr12_1 = vbslq_u32(vmask3, vr12, vzero);

      uint32x4_t vr30 = vld1q_u32(ptr3);
      uint32x4_t vr31 = vld1q_u32(ptr3 + 4);
      uint32x4_t vr32 = vld1q_u32(ptr3 + 8);

      vst1q_u32(outptr_row_col + 12, vr10_1);
      vst1q_u32(outptr_row_col + 16, vr11_1);
      vst1q_u32(outptr_row_col + 20, vr12_1);

      uint32x4_t vr20_1 = vbslq_u32(vmask1, vr20, vzero);
      uint32x4_t vr21_1 = vbslq_u32(vmask2, vr21, vzero);
      uint32x4_t vr22_1 = vbslq_u32(vmask3, vr22, vzero);

      uint32x4_t vr30_1 = vbslq_u32(vmask1, vr30, vzero);
      uint32x4_t vr31_1 = vbslq_u32(vmask2, vr31, vzero);
      uint32x4_t vr32_1 = vbslq_u32(vmask3, vr32, vzero);

      vst1q_u32(outptr_row_col + 24, vr20_1);
      vst1q_u32(outptr_row_col + 28, vr21_1);
      vst1q_u32(outptr_row_col + 32, vr22_1);

      vst1q_u32(outptr_row_col + 36, vr30_1);
      vst1q_u32(outptr_row_col + 40, vr31_1);
      vst1q_u32(outptr_row_col + 44, vr32_1);
    }
  }

#pragma omp parallel for
  for (int y = cnt_y; y < y_len; ++y) {
    const uint32_t *ptr0 = inptr + y * ldin;
    uint32_t *outptr_row_col = outptr_row + y * 12;

    int i = 0;
    for (; i < x_len - 11; i += 12) {
      uint32x4_t vr0 = vld1q_u32(ptr0);
      uint32x4_t vr1 = vld1q_u32(ptr0 + 4);
      uint32x4_t vr2 = vld1q_u32(ptr0 + 8);
      vst1q_u32(outptr_row_col, vr0);
      vst1q_u32(outptr_row_col + 4, vr1);
      vst1q_u32(outptr_row_col + 8, vr2);

      ptr0 += 12;

      outptr_row_col += stride_out;
    }
    if (right_remain > 0) {
      uint32x4_t vr0 = vld1q_u32(ptr0);
      uint32x4_t vr1 = vld1q_u32(ptr0 + 4);
      uint32x4_t vr2 = vld1q_u32(ptr0 + 8);

      uint32x4_t vr0_1 = vbslq_u32(vmask1, vr0, vzero);
      uint32x4_t vr1_1 = vbslq_u32(vmask2, vr1, vzero);
      uint32x4_t vr2_1 = vbslq_u32(vmask3, vr2, vzero);

      vst1q_u32(outptr_row_col, vr0_1);
      vst1q_u32(outptr_row_col + 4, vr1_1);
      vst1q_u32(outptr_row_col + 8, vr2_1);
    }
  }
}

void loadb_trans(
    float *out, const float *in, int ldin, int k0, int kmax, int n0, int nmax) {
  int x_len = kmax - k0;
  uint32_t zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(uint32_t) * x_len);
  auto outptr = reinterpret_cast<uint32_t *>(out);
  auto inptr = reinterpret_cast<const uint32_t *>(in);

  //! data B is not transposed, transpose B to k * 12
  for (int y = n0; y < nmax; y += 12) {
    const uint32_t *inptr0 = inptr + y * ldin + k0;
    const uint32_t *inptr1 = inptr0 + ldin;
    const uint32_t *inptr2 = inptr1 + ldin;
    const uint32_t *inptr3 = inptr2 + ldin;
    const uint32_t *inptr4 = inptr3 + ldin;
    const uint32_t *inptr5 = inptr4 + ldin;
    const uint32_t *inptr6 = inptr5 + ldin;
    const uint32_t *inptr7 = inptr6 + ldin;
    const uint32_t *inptr8 = inptr7 + ldin;
    const uint32_t *inptr9 = inptr8 + ldin;
    const uint32_t *inptr10 = inptr9 + ldin;
    const uint32_t *inptr11 = inptr10 + ldin;

    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]        \n"
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
          [ptr11] "r"(inptr11)
        : "memory");

    int x = x_len;

    //! cope with row index exceed real size, set to zero buffer
    if ((y + 11) >= nmax) {
      switch ((y + 11) - nmax) {
        case 10:
          inptr1 = zerobuff;
        case 9:
          inptr2 = zerobuff;
        case 8:
          inptr3 = zerobuff;
        case 7:
          inptr4 = zerobuff;
        case 6:
          inptr5 = zerobuff;
        case 5:
          inptr6 = zerobuff;
        case 4:
          inptr7 = zerobuff;
        case 3:
          inptr8 = zerobuff;
        case 2:
          inptr9 = zerobuff;
        case 1:
          inptr10 = zerobuff;
        case 0:
          inptr11 = zerobuff;
        default:
          break;
      }
    }
    for (; x > 7; x -= 8) {
      // clang-format off
      asm volatile(
          "ldp    q0, q1, [%[inptr0]], #32\n" /* r0, a0~a7 */
          "ldp    q2, q3, [%[inptr1]], #32\n" /* r1, b0~b7 */
          "ldp    q4, q5, [%[inptr2]], #32\n" /* r2, c0~c7 */
          "ldp    q6, q7, [%[inptr3]], #32\n" /* r3, d0~d7 */

          "zip1   v16.4s, v0.4s, v4.4s\n" /* a0c0a1c1 */
          "zip1   v17.4s, v2.4s, v6.4s\n" /* b0d0b1d1 */
          "prfm   pldl1keep, [%[inptr0], #128] \n"

          "ldp    q8, q9, [%[inptr4]], #32\n"   /* r4, e0~e7 */
          "ldp    q10, q11, [%[inptr5]], #32\n" /* r5, f0~f7 */
          "ldp    q12, q13, [%[inptr6]], #32\n" /* r6, g0~g7 */
          "ldp    q14, q15, [%[inptr7]], #32\n" /* r7, h0~h7 */

          "zip1   v18.4s, v8.4s, v12.4s\n"  /* e0g0e1g1 */
          "zip1   v19.4s, v10.4s, v14.4s\n" /* f0h0f1h1 */
          "prfm   pldl1keep, [%[inptr1], #128]\n"
          "zip1   v20.4s, v16.4s, v17.4s\n" /* a0b0c0d0 */
          "zip1   v21.4s, v18.4s, v19.4s\n" /* e0f0g0h0 */
          "prfm   pldl1keep, [%[inptr2], #128]\n"
          "zip2   v22.4s, v16.4s, v17.4s\n" /* a1b1c1d1 */
          "zip2   v23.4s, v18.4s, v19.4s\n" /* e1f1g1h1 */

          "ldp    q24, q25, [%[inptr8]], #32\n"  /* r8, i0~i7 */
          "ldp    q26, q27, [%[inptr9]], #32\n"  /* r9, j0~j7 */
          "ldp    q28, q29, [%[inptr10]], #32\n" /* r10, k0~k7 */
          "ldp    q30, q31, [%[inptr11]], #32\n" /* r11, l0~l7 */

          "stp    q20, q21, [%[outptr]], #32\n" /* save a0~h0 */
          "prfm   pldl1keep, [%[inptr3], #128]\n"

          "zip1   v16.4s, v24.4s, v28.4s\n" /* i0k0i1k1 */
          "zip1   v17.4s, v26.4s, v30.4s\n" /* j0l0j1l1 */
          "prfm   pldl1keep, [%[inptr4], #128]\n"
          "zip1   v18.4s, v16.4s, v17.4s\n" /* i0j0k0l0 */
          "zip2   v19.4s, v16.4s, v17.4s\n" /* i1j1k1l1 */
          "prfm   pldl1keep, [%[inptr5], #128]\n"
          "zip2   v16.4s, v0.4s, v4.4s\n" /* a2c2a3c3 */
          "zip2   v17.4s, v2.4s, v6.4s\n" /* b2d2b3d3 */

          "str    q18, [%[outptr]], #16\n"      /* save j0~l0 */
          "stp    q22, q23, [%[outptr]], #32\n" /* save a1~h1 */
          "str    q19, [%[outptr]], #16\n"      /* save j1~l1 */

          "zip2   v18.4s, v8.4s, v12.4s\n"  /* e2g2e3g3 */
          "zip2   v19.4s, v10.4s, v14.4s\n" /* f2h2f3h3 */
          "prfm   pldl1keep, [%[inptr6], #128]\n"
          "zip1   v20.4s, v16.4s, v17.4s\n" /* a2b2c2d2 */
          "zip1   v21.4s, v18.4s, v19.4s\n" /* e2f2g2h2 */
          "prfm   pldl1keep, [%[inptr7], #128]\n"
          "zip2   v22.4s, v16.4s, v17.4s\n" /* a3b3c3d3 */
          "zip2   v23.4s, v18.4s, v19.4s\n" /* e3f3g3h3 */
          "prfm   pldl1keep, [%[inptr8], #128]\n"
          "zip2   v16.4s, v24.4s, v28.4s\n" /* i2k2i3k3 */
          "zip2   v17.4s, v26.4s, v30.4s\n" /* j2l2j3l3 */

          "stp    q20, q21, [%[outptr]], #32\n" /* save a2~h2 */

          "zip1   v18.4s, v16.4s, v17.4s\n" /* i2j2k2l2 */
          "zip2   v19.4s, v16.4s, v17.4s\n" /* i3j3k3l3 */
          "prfm   pldl1keep, [%[inptr9], #128]\n"
          "zip1   v16.4s, v1.4s, v5.4s\n" /* a4c4a5c5 */
          "zip1   v17.4s, v3.4s, v7.4s\n" /* b4d4b5d5 */

          "str    q18, [%[outptr]], #16\n"      /* save i2~l2 */
          "stp    q22, q23, [%[outptr]], #32\n" /* save a3~h3 */
          "str    q19, [%[outptr]], #16\n"      /* save i3~l3 */

          "zip1   v18.4s, v9.4s, v13.4s\n"  /* e4g4e5g5 */
          "zip1   v19.4s, v11.4s, v15.4s\n" /* f4h4f5h5 */
          "prfm   pldl1keep, [%[inptr10], #128]\n"
          "zip1   v20.4s, v16.4s, v17.4s\n" /* a4b4c4d4 */
          "zip1   v21.4s, v18.4s, v19.4s\n" /* e4f4g4h4 */
          "prfm   pldl1keep, [%[inptr11], #128]\n"
          "zip2   v22.4s, v16.4s, v17.4s\n" /* a5b5c5d5 */
          "zip2   v23.4s, v18.4s, v19.4s\n" /* e5f5g5h5 */
          "zip1   v16.4s, v25.4s, v29.4s\n" /* i4k4i5k5 */
          "zip1   v17.4s, v27.4s, v31.4s\n" /* j4l4j5l5 */

          "stp    q20, q21, [%[outptr]], #32\n" /* save a4~h4 */

          "zip1   v18.4s, v16.4s, v17.4s\n" /* i4j4k4l4 */
          "zip2   v19.4s, v16.4s, v17.4s\n" /* i5j5k5l5 */
          "zip2   v16.4s, v1.4s, v5.4s\n"   /* a6c6a7c7 */
          "zip2   v17.4s, v3.4s, v7.4s\n"   /* b6d6b7d7 */

          "str    q18, [%[outptr]], #16\n"      /* save i4~l4 */
          "stp    q22, q23, [%[outptr]], #32\n" /* save a5~h5 */
          "str    q19, [%[outptr]], #16\n"      /* save i5~l5 */

          "zip2   v18.4s, v9.4s, v13.4s\n"  /* e6g6e7g7 */
          "zip2   v19.4s, v11.4s, v15.4s\n" /* f6h6f7h7 */
          "zip1   v20.4s, v16.4s, v17.4s\n" /* a6b6c6d6 */
          "zip1   v21.4s, v18.4s, v19.4s\n" /* e6f6g6h6 */
          "zip2   v22.4s, v16.4s, v17.4s\n" /* a7b7c7d7 */
          "zip2   v23.4s, v18.4s, v19.4s\n" /* e7f7g7h7 */
          "zip2   v16.4s, v25.4s, v29.4s\n" /* i6k6i7k7 */
          "zip2   v17.4s, v27.4s, v31.4s\n" /* j6l6j7l7 */

          "stp    q20, q21, [%[outptr]], #32\n" /* save a6~h6 */

          "zip1   v18.4s, v16.4s, v17.4s\n" /* i6j6k6l6 */
          "zip2   v19.4s, v16.4s, v17.4s\n" /* i7j7k7l7 */

          "str    q18, [%[outptr]], #16\n"      /* save i6~l6 */
          "stp    q22, q23, [%[outptr]], #32\n" /* save a7~h7 */
          "str    q19, [%[outptr]], #16\n"      /* save i7~l7 */
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
            [outptr] "+r"(outptr)
          :
          : "v0","v1","v2","v3","v4","v5",
            "v6","v7","v8","v9","v10","v11","v12",
            "v13","v14","v15","v16","v17","v18","v19",
            "v20","v21","v22","v23","v24","v25","v26",
            "v27","v28","v29","v30","v31","cc","memory");
      // clang-format on
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
    }
  }
}

void loadb_eight(
    float *out, const float *in, int ldin, int k0, int kmax, int n0, int nmax) {
  auto outptr = reinterpret_cast<uint32_t *>(out);
  auto inptr = reinterpret_cast<const uint32_t *>(in) + k0 * ldin + n0;
  uint32_t mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int x_len = nmax - n0;
  int y_len = kmax - k0;
  int right_remain = x_len - 8 * (x_len / 8);
  int right_pad = 8 - right_remain;

  uint32_t *outptr_row = outptr;
  int stride_out = 8 * y_len;
  int cnt_y = 4 * (y_len / 4);

  uint32x4_t vzero = vdupq_n_u32(0);
  uint32x4_t vmask1 =
      vcltq_u32(vld1q_u32(mask_buffer), vdupq_n_u32(right_remain));
  uint32x4_t vmask2 =
      vcltq_u32(vld1q_u32(mask_buffer + 4), vdupq_n_u32(right_remain));

#pragma omp parallel for
  for (int y = 0; y < y_len - 3; y += 4) {
    const uint32_t *ptr0 = inptr + y * ldin;
    const uint32_t *ptr1 = ptr0 + ldin;
    const uint32_t *ptr2 = ptr1 + ldin;
    const uint32_t *ptr3 = ptr2 + ldin;
    uint32_t *outptr_row_col = outptr_row + y * 8;
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
    int i = 0;
    for (; i < x_len - 7; i += 8) {
      uint32_t *ptr_out = outptr_row_col;
      asm volatile(
          "ldp q0, q1, [%[ptr0]], #32\n"    // load r0, 8 elements
          "ldp q2, q3, [%[ptr1]], #32\n"    // load r1, 8 elements
          "stp q0, q1, [%[outptr]], #32\n"  // write to output ptr
          "stp q2, q3, [%[outptr]], #32\n"  // write to output ptr
          "ldp q0, q1, [%[ptr2]], #32\n"    // load r0, 8 elements
          "ldp q2, q3, [%[ptr3]], #32\n"    // load r1, 8 elements
          "stp q0, q1, [%[outptr]], #32\n"  // write to output ptr
          "stp q2, q3, [%[outptr]], #32\n"  // write to output ptr
          : [outptr] "+r"(ptr_out),
            [ptr0] "+r"(ptr0),
            [ptr1] "+r"(ptr1),
            [ptr2] "+r"(ptr2),
            [ptr3] "+r"(ptr3)
          :
          : "v0", "v1", "v2", "v3", "cc", "memory");
      outptr_row_col += stride_out;
    }
    if (right_remain > 0) {
      uint32_t *ptr_out = outptr_row_col;
      asm volatile(
          "ldp q0, q1, [%[ptr0]], #32\n"
          "ldp q2, q3, [%[ptr1]], #32\n"
          "bif v0.16b, %[vzero].16b, %[vmask1].16b\n"
          "bif v1.16b, %[vzero].16b, %[vmask2].16b\n"
          "bif v2.16b, %[vzero].16b, %[vmask1].16b\n"
          "bif v3.16b, %[vzero].16b, %[vmask2].16b\n"
          "stp q0, q1, [%[outptr]], #32\n"
          "ldp q0, q1, [%[ptr2]], #32\n"
          "stp q2, q3, [%[outptr]], #32\n"
          "ldp q2, q3, [%[ptr3]], #32\n"
          "bif v0.16b, %[vzero].16b, %[vmask1].16b\n"
          "bif v1.16b, %[vzero].16b, %[vmask2].16b\n"
          "bif v2.16b, %[vzero].16b, %[vmask1].16b\n"
          "bif v3.16b, %[vzero].16b, %[vmask2].16b\n"
          "stp q0, q1, [%[outptr]], #32\n"
          "stp q2, q3, [%[outptr]], #32\n"
          : [outptr] "+r"(ptr_out),
            [ptr0] "+r"(ptr0),
            [ptr1] "+r"(ptr1),
            [ptr2] "+r"(ptr2),
            [ptr3] "+r"(ptr3)
          : [vmask1] "w"(vmask1), [vmask2] "w"(vmask2), [vzero] "w"(vzero)
          : "v0", "v1", "v2", "v3", "cc", "memory");
    }
  }
#pragma omp parallel for
  for (int y = cnt_y; y < y_len; ++y) {
    const uint32_t *ptr0 = inptr + y * ldin;
    uint32_t *outptr_row_col = outptr_row + y * 8;
    int i = 0;
    for (; i < x_len - 7; i += 8) {
      uint32_t *ptr_out = outptr_row_col;
      asm volatile(
          "ldp q0, q1, [%[ptr0]], #32\n"
          "stp q0, q1, [%[outptr]], #32\n"
          : [ptr0] "+r"(ptr0), [outptr] "+r"(ptr_out)
          :
          : "v0", "v1", "cc", "memory");
      outptr_row_col += stride_out;
    }
    if (right_remain > 0) {
      uint32_t *ptr_out = outptr_row_col;
      asm volatile(
          "ldp q0, q1, [%[ptr0]], #32\n"
          "bif v0.16b, %[vzero].16b, %[vmask1].16b\n"
          "bif v1.16b, %[vzero].16b, %[vmask2].16b\n"
          "stp q0, q1, [%[outptr]], #32\n"
          : [ptr0] "+r"(ptr0), [outptr] "+r"(ptr_out)
          : [vmask1] "w"(vmask1), [vmask2] "w"(vmask2), [vzero] "w"(vzero)
          : "v0", "v1", "cc", "memory");
    }
  }
}

void loadb_trans_eight(
    float *out, const float *in, int ldin, int k0, int kmax, int n0, int nmax) {
  int x_len = kmax - k0;
  uint32_t zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(uint32_t) * x_len);

  auto outptr = reinterpret_cast<uint32_t *>(out);
  auto inptr = reinterpret_cast<const uint32_t *>(in);
  //! data B is not transposed, transpose B to k * 8
  for (int y = n0; y < nmax; y += 8) {
    const uint32_t *inptr0 = inptr + y * ldin + k0;
    const uint32_t *inptr1 = inptr0 + ldin;
    const uint32_t *inptr2 = inptr1 + ldin;
    const uint32_t *inptr3 = inptr2 + ldin;
    const uint32_t *inptr4 = inptr3 + ldin;
    const uint32_t *inptr5 = inptr4 + ldin;
    const uint32_t *inptr6 = inptr5 + ldin;
    const uint32_t *inptr7 = inptr6 + ldin;

    int x = x_len;
    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]        \n"
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
        :
        : [ptr0] "r"(inptr0),
          [ptr1] "r"(inptr1),
          [ptr2] "r"(inptr2),
          [ptr3] "r"(inptr3),
          [ptr4] "r"(inptr4),
          [ptr5] "r"(inptr5),
          [ptr6] "r"(inptr6),
          [ptr7] "r"(inptr7)
        : "memory");

    //! cope with row index exceed real size, set to zero buffer
    if ((y + 7) >= nmax) {
      switch ((y + 7) - nmax) {
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

    for (; x > 7; x -= 8) {
      // clang-format off
      //! zip load 8 elements (2 neon Q registers) from each of 8 rows
      asm volatile(
          "ldp  q0, q1, [%[inptr0]], #32\n"  // load r0, a0~a7
          "ldp  q2, q3, [%[inptr1]], #32\n"  // load r1, b0~b7
          "ldp  q4, q5, [%[inptr2]], #32\n"  // load r2, c0~c7
          "ldp  q6, q7, [%[inptr3]], #32\n"  // load r3, d0~d7
          "trn1 v8.4s, v0.4s, v2.4s\n"       // a0b0a2b2
          "trn2 v9.4s, v0.4s, v2.4s\n"       // a1b1a3b3
          "trn1 v10.4s, v1.4s, v3.4s\n"      // a4b4a6b6
          "trn2 v11.4s, v1.4s, v3.4s\n"      // a5b5a7b7
          "ldp  q16, q17, [%[inptr4]], #32\n"// load r4, e0~e7
          "ldp  q18, q19, [%[inptr5]], #32\n"// load r5, f0~f7
          "ldp  q20, q21, [%[inptr6]], #32\n"// load r6, g0~g7
          "ldp  q22, q23, [%[inptr7]], #32\n"// load r7, h0~h7
          "trn1 v12.4s, v4.4s, v6.4s\n"      // c0d0c2d2
          "trn2 v13.4s, v4.4s, v6.4s\n"      // c1d1c3d3
          "trn1 v14.4s, v5.4s, v7.4s\n"      // c4d4c6d6
          "trn2 v15.4s, v5.4s, v7.4s\n"      // c5d5c7d7

          "trn1 v24.4s, v16.4s, v18.4s\n"    // e0f0e2f2
          "trn2 v25.4s, v16.4s, v18.4s\n"    // e1f1e3f3
          "trn1 v28.4s, v20.4s, v22.4s\n"    // g0h0e2f2
          "trn2 v29.4s, v20.4s, v22.4s\n"    // g1h1e3f3
          "trn1 v26.4s, v17.4s, v19.4s\n"    // e4f4e6f6
          "trn2 v27.4s, v17.4s, v19.4s\n"    // e5f5e7f7
          "trn1 v30.4s, v21.4s, v23.4s\n"    // g4h4e6f6
          "trn2 v31.4s, v21.4s, v23.4s\n"    // g5h5e7f7

          "trn1 v0.2d, v8.2d, v12.2d\n"      // a0b0c0d0
          "trn1 v1.2d, v24.2d, v28.2d\n"     // e0f0g0h0
          "trn1 v2.2d, v9.2d, v13.2d\n"      // a1b1c1d1
          "trn1 v3.2d, v25.2d, v29.2d\n"     // e1f1g1h1
          "trn2 v4.2d, v8.2d, v12.2d\n"      // a2b2c2d2
          "trn2 v5.2d, v24.2d, v28.2d\n"     // e2f2g2h2
          "stp q0, q1, [%[outptr]], #32\n"   // save q0, q1, a0~h0
          "trn2 v6.2d, v9.2d, v13.2d\n"      // a3b3c3d3
          "trn2 v7.2d, v25.2d, v29.2d\n"     // e3f3g3h3
          "stp q2, q3, [%[outptr]], #32\n"   // save q0, q1, a1~h1

          "trn1 v16.2d, v10.2d, v14.2d\n"    // a4b4c4d4
          "trn1 v17.2d, v26.2d, v30.2d\n"    // e4f4g4h4
          "stp q4, q5, [%[outptr]], #32\n"   // save q0, q1, a2~h2
          "trn1 v18.2d, v11.2d, v15.2d\n"    // a5b5c5d5
          "trn1 v19.2d, v27.2d, v31.2d\n"    // e5f5g5h5
          "stp q6, q7, [%[outptr]], #32\n"   // save q0, q1, a3~h3
          "trn2 v20.2d, v10.2d, v14.2d\n"    // a6b6c6d6
          "trn2 v21.2d, v26.2d, v30.2d\n"    // e6f6g6h6
          "stp q16, q17, [%[outptr]], #32\n" // save q0, q1, a4~h4
          "trn2 v22.2d, v11.2d, v15.2d\n"    // a7b7c7d7
          "trn2 v23.2d, v27.2d, v31.2d\n"    // e7f7g7h7
          "stp q18, q19, [%[outptr]], #32\n" // save q0, q1, a5~h5
          "stp q20, q21, [%[outptr]], #32\n" // save q0, q1, a6~h6
          "stp q22, q23, [%[outptr]], #32\n" // save q0, q1, a7~h7
          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [inptr4] "+r"(inptr4),
            [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6),
            [inptr7] "+r"(inptr7),
            [outptr] "+r"(outptr)
          :
          : "v0","v1","v2","v3","v4","v5",
            "v6","v7","v8","v9","v10","v11","v12",
            "v13","v14","v15","v16","v17","v18","v19",
            "v20","v21","v22","v23","v24","v25","v26",
            "v27","v28","v29","v30","v31","cc","memory");
      // clang-format on
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
    }
  }
}

#else  // __aarch64__
void loadb(
    float* out, const float* in, int ldin, int k0, int kmax, int n0, int nmax) {
  auto outptr = reinterpret_cast<uint32_t*>(out);
  auto inptr = reinterpret_cast<const uint32_t*>(in) + k0 * ldin + n0;
  uint32_t mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int x_len = nmax - n0;
  int y_len = kmax - k0;
  int right_remain = x_len - 8 * (x_len / 8);
  int right_pad = 8 - right_remain;

  uint32_t* outptr_row = outptr;
  int stride_out = 8 * y_len;

  uint32x4_t vzero = vdupq_n_u32(0);
  uint32x4_t vmask1 =
      vcltq_u32(vld1q_u32(mask_buffer), vdupq_n_u32(right_remain));
  uint32x4_t vmask2 =
      vcltq_u32(vld1q_u32(mask_buffer + 4), vdupq_n_u32(right_remain));

#pragma omp parallel for
  for (int y = 0; y < y_len - 3; y += 4) {
    const uint32_t* ptr0 = inptr + y * ldin;
    const uint32_t* ptr1 = ptr0 + ldin;
    const uint32_t* ptr2 = ptr1 + ldin;
    const uint32_t* ptr3 = ptr2 + ldin;
    uint32_t* outptr_row_col = outptr_row + y * 8;
    int i = 0;
    for (; i < x_len - 7; i += 8) {
      uint32_t* ptr_out = outptr_row_col;
      asm volatile(
          "vld1.32 {d0-d3}, [%[ptr0]]!        @ load r0, 8 elements\n"
          "vld1.32 {d4-d7}, [%[ptr1]]!        @ load r1, 8 elements\n"
          "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
          "vst1.32 {d4-d7}, [%[outptr]]!      @ write to output ptr\n"

          "vld1.32 {d0-d3}, [%[ptr2]]!        @ load r2, 8 elements\n"
          "vld1.32 {d4-d7}, [%[ptr3]]!        @ load r3, 8 elements\n"
          "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
          "vst1.32 {d4-d7}, [%[outptr]]!      @ write to output ptr\n"
          : [outptr] "+r"(ptr_out),
            [ptr0] "+r"(ptr0),
            [ptr1] "+r"(ptr1),
            [ptr2] "+r"(ptr2),
            [ptr3] "+r"(ptr3)
          :
          : "q0", "q1", "q2", "q3", "cc", "memory");
      outptr_row_col += stride_out;
    }
    if (right_remain > 0) {
      uint32_t* ptr_out = outptr_row_col;
      asm volatile(
          "vld1.32 {d0-d3}, [%[ptr0]]!        @ load r0, 8 elements\n"
          "vld1.32 {d4-d7}, [%[ptr1]]!        @ load r1, 8 elements\n"
          "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
          "vbif   q1, %q[vzero], %q[vmask2]   @ bit select, pad zero\n"
          //"vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
          "vbif   q2, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
          "vbif   q3, %q[vzero], %q[vmask2]   @ bit select, pad zero\n"
          "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
          "vst1.32 {d4-d7}, [%[outptr]]!      @ write to output ptr\n"

          "vld1.32 {d0-d3}, [%[ptr2]]!        @ load r2, 8 elements\n"
          "vld1.32 {d4-d7}, [%[ptr3]]!        @ load r3, 8 elements\n"
          "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
          "vbif   q1, %q[vzero], %q[vmask2]   @ bit select, pad zero\n"
          //"vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
          "vbif   q2, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
          "vbif   q3, %q[vzero], %q[vmask2]   @ bit select, pad zero\n"
          "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
          "vst1.32 {d4-d7}, [%[outptr]]!      @ write to output ptr\n"
          : [outptr] "+r"(ptr_out),
            [ptr0] "+r"(ptr0),
            [ptr1] "+r"(ptr1),
            [ptr2] "+r"(ptr2),
            [ptr3] "+r"(ptr3)
          : [vmask1] "w"(vmask1), [vmask2] "w"(vmask2), [vzero] "w"(vzero)
          : "q0", "q1", "q2", "q3", "cc", "memory");
    }
  }
#pragma omp parallel for
  for (int y = 4 * (y_len / 4); y < y_len; ++y) {
    const uint32_t* ptr0 = inptr + y * ldin;
    uint32_t* outptr_row_col = outptr_row + y * 8;
    int i = 0;
    for (; i < x_len - 7; i += 8) {
      uint32_t* ptr_out = outptr_row_col;
      asm volatile(
          "vld1.32 {d0-d3}, [%[ptr0]]!        @ load r0, 8 elements\n"
          "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
          : [ptr0] "+r"(ptr0), [outptr] "+r"(ptr_out)
          :
          : "q0", "q1", "cc", "memory");
      outptr_row_col += stride_out;
    }
    if (right_remain > 0) {
      uint32_t* ptr_out = outptr_row_col;
      asm volatile(
          "vld1.32 {d0-d3}, [%[ptr0]]!        @ load r0, 8 elements\n"
          "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
          "vbif   q1, %q[vzero], %q[vmask2]   @ bit select, pad zero\n"
          "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
          : [ptr0] "+r"(ptr0), [outptr] "+r"(ptr_out)
          : [vmask1] "w"(vmask1), [vmask2] "w"(vmask2), [vzero] "w"(vzero)
          : "q0", "q1", "cc", "memory");
    }
  }
}

void loadb_trans(
    float* out, const float* in, int ldin, int k0, int kmax, int n0, int nmax) {
  int x_len = kmax - k0;
  uint32_t zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(uint32_t) * x_len);

  auto outptr = reinterpret_cast<uint32_t*>(out);
  auto inptr = reinterpret_cast<const uint32_t*>(in);
  //! data B is not transposed, transpose B to k * 8
  for (int y = n0; y < nmax; y += 8) {
    const uint32_t* inptr0 = inptr + y * ldin + k0;
    const uint32_t* inptr1 = inptr0 + ldin;
    const uint32_t* inptr2 = inptr1 + ldin;
    const uint32_t* inptr3 = inptr2 + ldin;
    const uint32_t* inptr4 = inptr3 + ldin;
    const uint32_t* inptr5 = inptr4 + ldin;
    const uint32_t* inptr6 = inptr5 + ldin;
    const uint32_t* inptr7 = inptr6 + ldin;

    int x = x_len;

    //! cope with row index exceed real size, set to zero buffer
    if ((y + 7) >= nmax) {
      switch ((y + 7) - nmax) {
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

    for (; x > 7; x -= 8) {
      //! zip load 8 elements (2 neon Q registers) from each of 8 rows
      asm volatile(
          "vld4.32  {d0-d3}, [%[inptr0]]!   @ zip load r0, "
          "q0,q1=r00,r04,r01,r05,r02,r06,r03,r07\n"
          "vld4.32  {d4-d7}, [%[inptr1]]!   @ zip load r1, "
          "q2,q3=r10,r14,r11,r15,r12,r16,r13,r17\n"
          "vtrn.32  q0, q2                  @ trans data: q0=r00,r10,r01,r11; "
          "q2=r04,r14,r05,r15\n"
          "vst1.32  {d0},    [%[outptr]]!   @ write d0(q0,low),r00,r10\n"

          "vld4.32  {d8-d11}, [%[inptr2]]!  @ zip load r2, "
          "q4,q5=r20,r24,r21,r25,r22,r26,r23,r27\n"
          "vld4.32  {d12-d15}, [%[inptr3]]! @ zip load r3, "
          "q6,q7=r30,r34,r31,r35,r32,r36,r33,r37\n"
          "vtrn.32  q4, q6                  @ trans data: q4=r20,r30,r21,r31; "
          "q6=r24,r34,r25,r35\n"
          "vst1.32  {d8},    [%[outptr]]!   @ write d8(q4,low),r20,r30\n"

          "vld4.32  {d16-d19}, [%[inptr4]]! @ zip load r4, "
          "q8,q9=r40,r44,r41,r45,r42,r46,r43,r47\n"
          "vld4.32  {d20-d23}, [%[inptr5]]! @ zip load r5, "
          "q10,q11=r50,r54,r51,r55,r52,r56,r53,r57\n"
          "vtrn.32  q8, q10                 @ trans data: q8=r40,r50,r41,r51; "
          "q10=r44,r54,r45,r55\n"
          "vst1.32  {d16},    [%[outptr]]!  @ write d16(q8,low),r40,r50\n"

          "vld4.32  {d24-d27}, [%[inptr6]]! @ zip load r6, "
          "q12,q13=r60,r64,r61,r65,r62,r66,r63,r67\n"
          "vld4.32  {d28-d31}, [%[inptr7]]! @ zip load r7, "
          "q14,q15=r70,r74,r71,r75,r72,r76,r73,r77\n"
          "vtrn.32  q12, q14                @ trans data:q12=r60,r70,r61,r71; "
          "q14=r64,r74,r65,r75\n"
          "vst1.32  {d24},    [%[outptr]]!  @ write d24(q8,low),r60,r70\n"

          //"pld      [%[inptr0], #128]       @ preload r0 data to cache, fill
          // pipeline\n"
          "vst1.32  {d1},     [%[outptr]]!  @ write d1(q0,high),r01,r11\n"
          "vst1.32  {d9},     [%[outptr]]!  @ write d9(q4,high),r21,r31\n"
          "vst1.32  {d17},    [%[outptr]]!  @ write d17(q8,high),r41,r51\n"
          "vst1.32  {d25},    [%[outptr]]!  @ write d25(q12,high),r61,r71\n"

          "vtrn.32  q1, q3                  @ trans data: q1=r02,r12,r03,r13; "
          "q3=r06,r16,r07,r17\n"
          "vst1.32  {d2},     [%[outptr]]!  @ write d2(q1,low),r02,r12\n"
          "vtrn.32  q5, q7                  @ trans data: q5=r22,r32,r23,r33; "
          "q7=r26,r36,r27,r37\n"
          "vst1.32  {d10},    [%[outptr]]!  @ write d10(q5,low),r22,r32\n"
          "vtrn.32  q9, q11                 @ trans data: q9=r42,r52,r43,r53; "
          "q11=r46,r56,r47,r57\n"
          "vst1.32  {d18},    [%[outptr]]!  @ write d18(q9,low),r42,r52\n"
          "vtrn.32  q13, q15                @ trans data:q13=r62,r72,r63,r73; "
          "q15=r66,r76,r67,r77\n"
          "vst1.32  {d26},    [%[outptr]]!  @ write d18(q9,low),r62,r72\n"

          //"pld      [%[inptr1], #128]       @ preload r1 data to cache, fill
          // pipeline\n"
          "vst1.32  {d3},     [%[outptr]]!  @ write d3(q1,high),r03,r13\n"
          "vst1.32  {d11},    [%[outptr]]!  @ write d11(q5,high),r23,r33\n"
          "vst1.32  {d19},    [%[outptr]]!  @ write d19(q9,high),r43,r53\n"
          "vst1.32  {d27},    [%[outptr]]!  @ write d27(q13,high),r63,r73\n"

          //"pld      [%[inptr2], #128]       @ preload r2 data to cache, fill
          // pipeline\n"
          "vst1.32  {d4},     [%[outptr]]!  @ write d4(q2,low),r04,r14\n"
          "vst1.32  {d12},    [%[outptr]]!  @ write d12(q6,low),r24,r34\n"
          "vst1.32  {d20},    [%[outptr]]!  @ write d20(q10,low),r44,r54\n"
          "vst1.32  {d28},    [%[outptr]]!  @ write d28(q14,low),r64,r74\n"

          //"pld      [%[inptr3], #128]       @ preload r3 data to cache, fill
          // pipeline\n"
          "vst1.32  {d5},     [%[outptr]]!  @ write d5(q2,high),r05,r15\n"
          "vst1.32  {d13},    [%[outptr]]!  @ write d13(q6,high),r25,r35\n"
          "vst1.32  {d21},    [%[outptr]]!  @ write d21(q10,high),r45,r55\n"
          "vst1.32  {d29},    [%[outptr]]!  @ write d29(q14,high),r65,r75\n"

          //"pld      [%[inptr4], #128]       @ preload r4 data to cache, fill
          // pipeline\n"
          "vst1.32  {d6},     [%[outptr]]!  @ write d6(q3,low),r06,r16\n"
          "vst1.32  {d14},    [%[outptr]]!  @ write d14(q7,low),r26,r36\n"
          "vst1.32  {d22},    [%[outptr]]!  @ write d22(q11,low),r46,r56\n"
          "vst1.32  {d30},    [%[outptr]]!  @ write d30(q15,low),r66,r76\n"

          //"pld      [%[inptr5], #128]       @ preload r5 data to cache, fill
          // pipeline\n"
          "vst1.32  {d7},     [%[outptr]]!  @ write d7(q3,high),r07,r17\n"
          "vst1.32  {d15},    [%[outptr]]!  @ write d15(q7,high),r27,r37\n"
          "vst1.32  {d23},    [%[outptr]]!  @ write d23(q11,high),r47,r57\n"
          "vst1.32  {d31},    [%[outptr]]!  @ write d31(q15,high),r67,r77\n"
          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [inptr4] "+r"(inptr4),
            [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6),
            [inptr7] "+r"(inptr7),
            [outptr] "+r"(outptr)
          :
          : "q0",
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "q7",
            "q8",
            "q9",
            "q10",
            "q11",
            "q12",
            "q13",
            "q14",
            "q15",
            "cc",
            "memory");
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
    }
  }
}

#endif  // __aarch64__

#ifdef __aarch64__
void sgemm_prepacked_8x12(bool is_transB,
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
  size_t l2_cache = ctx->llc_size() > 0 ? ctx->llc_size() : 512 * 1024;
  auto workspace = ctx->workspace_data<float>();
  int threads = ctx->threads();

  auto act_type = act_param.active_type;
  float alpha[4] = {0.f, 0.f, 0.f, 0.f};
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    }
  }
  //! MBLOCK * x (result) + MBLOCK * k (A) + x * k (B) = l2
  int x_block = (l2_cache - (MBLOCK * K)) / (sizeof(float) * (K + MBLOCK));
  x_block /= NBLOCK;
  x_block *= NBLOCK;
  int x_num = (N + (x_block - 1)) / x_block;
  x_block = (N + x_num - 1) / x_num;
  x_block = (x_block + NBLOCK - 1) / NBLOCK;
  x_block *= NBLOCK;
  x_block = x_block < NBLOCK ? NBLOCK : x_block;

  // unroll 2 loop
  int tail_pre = (K & (KBLOCK - 1));
  int k_pre = ((K + KBLOCK - 1) / KBLOCK) - 1;

  bool flag_p_remain = false;
  int remain = 0;
  if (tail_pre == 0) {
    tail_pre = KBLOCK;
  }

  int has_beta = fabsf(beta) > 1e-8f ? 1 : 0;

  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    if (xmax > N) {
      xmax = N;
    }
    int bblocks = (xmax - x0 + NBLOCK - 1) / NBLOCK;
    remain = xmax - x0 - (bblocks - 1) * NBLOCK;
    if (remain > 0) {
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
        bias_local[0] = bias[y];
        bias_local[1] = bias[y + 1];
        bias_local[2] = bias[y + 2];
        bias_local[3] = bias[y + 3];
        bias_local[4] = bias[y + 4];
        bias_local[5] = bias[y + 5];
        bias_local[6] = bias[y + 6];
        bias_local[7] = bias[y + 7];
      }

      float cout0[NBLOCK];
      float cout1[NBLOCK];
      float cout2[NBLOCK];
      float cout3[NBLOCK];
      float cout4[NBLOCK];
      float cout5[NBLOCK];
      float cout6[NBLOCK];
      float cout7[NBLOCK];

      float *c_ptr0 = C + y * ldc + x0;
      float *c_ptr1 = c_ptr0 + ldc;
      float *c_ptr2 = c_ptr1 + ldc;
      float *c_ptr3 = c_ptr2 + ldc;
      float *c_ptr4 = c_ptr3 + ldc;
      float *c_ptr5 = c_ptr4 + ldc;
      float *c_ptr6 = c_ptr5 + ldc;
      float *c_ptr7 = c_ptr6 + ldc;

      float *pout0 = c_ptr0;
      float *pout1 = c_ptr1;
      float *pout2 = c_ptr2;
      float *pout3 = c_ptr3;
      float *pout4 = c_ptr4;
      float *pout5 = c_ptr5;
      float *pout6 = c_ptr6;
      float *pout7 = c_ptr7;

      const float *a_ptr_l = A_packed + y * K;
      const float *b_ptr = b_pannel;
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
          if (has_beta) {
            for (int i = 0; i < remain; ++i) {
              cout0[i] = pout0[i];
              cout1[i] = pout1[i];
              cout2[i] = pout2[i];
              cout3[i] = pout3[i];
              cout4[i] = pout4[i];
              cout5[i] = pout5[i];
              cout6[i] = pout6[i];
              cout7[i] = pout7[i];
            }
          }
        }
        const float *a_ptr = a_ptr_l;
        int tail = tail_pre;
        int k = k_pre;
        // clang-format off
        asm volatile(
            "prfm   pldl1keep, [%[a_ptr]]\n"       /* preload a*/
            "ldp	q2, q3, [%[bias_ptr]]\n"         /* load bias to q2, q3*/
            "dup	v8.4s,  v2.s[0]\n"               /* out0 = 0 */
            "dup	v9.4s,  v2.s[0]\n"               /* out1 = 0*/
            "dup	v10.4s, v2.s[0]\n"               /* out2 = 0*/
            "prfm   pldl1keep, [%[b_ptr]]\n"       /* preload b*/
            "dup	v11.4s, v2.s[1]\n"               /* out3 = 0*/
            "dup	v12.4s, v2.s[1]\n"               /* out4 = 0*/
            "prfm   pldl1keep, [%[b_ptr], #64]\n"  /* preload b*/
            "dup	v13.4s, v2.s[1]\n"               /* out5 = 0*/
            "prfm   pldl1keep, [%[a_ptr], #64]\n"  /* preload a*/
            "dup	v14.4s, v2.s[2]\n"               /* out6 = 0*/
            "prfm   pldl1keep, [%[b_ptr], #128]\n" /* preload b*/
            "dup	v15.4s, v2.s[2]\n"               /* out7 = 0*/
            "prfm   pldl1keep, [%[a_ptr], #128]\n" /* preload a*/
            "dup	v16.4s, v2.s[2]\n"               /* out8 = 0*/
            "prfm   pldl1keep, [%[b_ptr], #192]\n" /* preload b*/
            "dup	v17.4s, v2.s[3]\n"               /* out9 = 0*/
            "prfm   pldl1keep, [%[b_ptr], #256]\n" /* preload b*/
            "dup	v18.4s, v2.s[3]\n"               /* out10 = 0*/
            "prfm   pldl1keep, [%[a_ptr], #192]\n" /* preload a*/
            "dup	v19.4s, v2.s[3]\n"               /* out11 = 0*/
            "prfm   pldl1keep, [%[b_ptr], #320]\n" /* preload b*/
            "dup	v20.4s, v3.s[0]\n"               /* out12 = 0*/
            "prfm   pldl1keep, [%[a_ptr], #256]\n" /* preload a*/
            "dup	v21.4s, v3.s[0]\n"               /* out13 = 0*/
            "prfm   pldl1keep, [%[b_ptr], #384]\n" /* preload b*/
            "dup	v22.4s, v3.s[0]\n"               /* out14 = 0*/
            "dup	v23.4s, v3.s[1]\n"               /* out15 = 0*/
            "dup	v24.4s, v3.s[1]\n"               /* out16 = 0*/
            "dup	v25.4s, v3.s[1]\n"               /* out17 = 0*/
            "dup	v26.4s, v3.s[2]\n"               /* out18 = 0*/
            "dup	v27.4s, v3.s[2]\n"               /* out19 = 0*/
            "dup	v28.4s, v3.s[2]\n"               /* out20 = 0*/
            "dup	v29.4s, v3.s[3]\n"               /* out21 = 0*/
            "dup	v30.4s, v3.s[3]\n"               /* out22 = 0*/
            "dup	v31.4s, v3.s[3]\n"               /* out23 = 0*/
            "cbz    %w[has_beta], 0f\n"            /* check beta == 0? */
            /* process beta */
            "dup    v7.4s, %w[beta]\n"                    /* beta to vector */
            "ld1    {v0.4s, v1.4s, v2.4s}, [%[c_ptr0]]\n" /* load output r0 */
            "ld1    {v3.4s, v4.4s, v5.4s}, [%[c_ptr1]]\n" /* load output r1 */
            "fmla   v8.4s, v0.4s, v7.4s\n"  /* cr00 += beta * c_r00*/
            "fmla   v9.4s, v1.4s, v7.4s\n"  /* cr01 += beta * c_r01*/
            "fmla   v10.4s, v2.4s, v7.4s\n" /* cr02 += beta * c_r02*/
            "ld1    {v0.4s, v1.4s, v2.4s}, [%[c_ptr2]]\n" /* load output r2*/
            "fmla   v11.4s, v3.4s, v7.4s\n" /* cr10 += beta * c_r10*/
            "fmla   v12.4s, v4.4s, v7.4s\n" /* cr11 += beta * c_r11*/
            "fmla   v13.4s, v5.4s, v7.4s\n" /* cr12 += beta * c_r12*/
            "ld1    {v3.4s, v4.4s, v5.4s}, [%[c_ptr3]]\n" /* load output r3*/
            "fmla   v14.4s, v0.4s, v7.4s\n" /* cr20 += beta * c_r20*/
            "fmla   v15.4s, v1.4s, v7.4s\n" /* cr21 += beta * c_r21*/
            "fmla   v16.4s, v2.4s, v7.4s\n" /* cr22 += beta * c_r22*/
            "ld1    {v0.4s, v1.4s, v2.4s}, [%[c_ptr4]]\n" /* load output r4*/
            "fmla   v17.4s, v3.4s, v7.4s\n" /* cr30 += beta * c_r30*/
            "fmla   v18.4s, v4.4s, v7.4s\n" /* cr31 += beta * c_r31*/
            "fmla   v19.4s, v5.4s, v7.4s\n" /* cr32 += beta * c_r32*/
            "ld1    {v3.4s, v4.4s, v5.4s}, [%[c_ptr5]]\n" /* load output r5*/
            "fmla   v20.4s, v0.4s, v7.4s\n" /* cr40 += beta * c_r40*/
            "fmla   v21.4s, v1.4s, v7.4s\n" /* cr41 += beta * c_r41*/
            "fmla   v22.4s, v2.4s, v7.4s\n" /* cr42 += beta * c_r42*/
            "ld1    {v0.4s, v1.4s, v2.4s}, [%[c_ptr6]]\n" /* load output r6*/
            "fmla   v23.4s, v3.4s, v7.4s\n" /* cr50 += beta * c_r50*/
            "fmla   v24.4s, v4.4s, v7.4s\n" /* cr51 += beta * c_r51*/
            "fmla   v25.4s, v5.4s, v7.4s\n" /* cr52 += beta * c_r52*/
            "ld1    {v3.4s, v4.4s, v5.4s}, [%[c_ptr7]]\n" /* load output r7*/
            "fmla   v26.4s, v0.4s, v7.4s\n"  /* cr60 += beta * c_r60*/
            "fmla   v27.4s, v1.4s, v7.4s\n"  /* cr61 += beta * c_r61*/
            "fmla   v28.4s, v2.4s, v7.4s\n"  /* cr62 += beta * c_r62*/
            "fmla   v29.4s, v3.4s, v7.4s\n"  /* cr70 += beta * c_r70*/
            "fmla   v30.4s, v4.4s, v7.4s\n"  /* cr71 += beta * c_r71*/
            "fmla   v31.4s, v5.4s, v7.4s\n"  /* cr72 += beta * c_r72*/
            "0: \n"                          /* check loop count */
            "ldp	q0, q1, [%[a_ptr]], #32\n" /* load a00,a01 to q0, q1*/
            "ldp	q4, q5, [%[b_ptr]], #32\n" /* load b0, b1 to q4, q5*/
            "cbz	%w[k], 2f\n"               /* check loop count > 0 */
            /* main loop */
            /* unrool 0*/
            "1:\n"                              /* main loop */
            "fmla 	v8.4s ,  v4.4s,  v0.s[0]\n" /* out0 = b0 * a00[0], b0 =q4 */
            "fmla  	v11.4s ,  v4.4s,  v0.s[1]\n" /* out1 = b0 * a00[1], b0 =q4
                                                    */
            "ldp	q6, q7, [%[b_ptr]], #32\n"     /* load b2, b0 to q6, q7 */
            "fmla	v14.4s,  v4.4s,  v0.s[2]\n"   /* out2 = b0 * a00[2], b0 =q4 */
            "fmla	v17.4s,  v4.4s,  v0.s[3]\n"   /* out3 = b0 * a00[3], b0 =q4 */
            "ldp	q2, q3, [%[a_ptr]], #32\n"    /* load a10, a11 to q3, q4 */
            "fmla 	v20.4s,  v4.4s,  v1.s[0]\n" /* out4 = b0 * a01[0], b0 =q4 */
            "fmla	v23.4s,  v4.4s,  v1.s[1]\n"   /* out5 = b0 * a01[1], b0 =q4 */
            "fmla	v26.4s,  v4.4s,  v1.s[2]\n"   /* out6 = b0 * a01[2], b0 =q4 */
            "fmla	v29.4s,  v4.4s,  v1.s[3]\n"   /* out7 = b0 * a01[3], b0 =q4 */

            "fmla	v9.4s,  v5.4s,  v0.s[0]\n"  /* out8 = b1 * a00[0], b1 =q5 */
            "fmla	v12.4s,  v5.4s,  v0.s[1]\n" /* out9 = b1 * a00[1], b1 =q5 */
            "fmla	v15.4s,  v5.4s,  v0.s[2]\n" /* out10 = b1 * a00[2], b1 =q5*/
            "fmla	v18.4s,  v5.4s,  v0.s[3]\n" /* out11 = b1 * a00[3], b1 =q5*/
            "fmla	v21.4s,  v5.4s,  v1.s[0]\n" /* out12 = b1 * a01[0], b1 =q5*/
            "fmla	v24.4s,  v5.4s,  v1.s[1]\n" /* out13 = b1 * a01[1], b1 =q5*/
            "fmla	v27.4s,  v5.4s,  v1.s[2]\n" /* out14 = b1 * a01[2], b1 =q5*/
            "fmla	v30.4s,  v5.4s,  v1.s[3]\n" /* out15 = b1 * a01[3], b1 =q5*/

            "ldp	q4, q5, [%[b_ptr]], #32\n" /* load b1, b2 to q4, q5 */

            "fmla	v10.4s,  v6.4s,  v0.s[0]\n" /* out16 = b2 * a00[0], b2 =q6*/
            "fmla	v13.4s,  v6.4s,  v0.s[1]\n" /* out17 = b2 * a00[1], b2 =q6*/
            "prfm   pldl1keep, [%[b_ptr], #384]\n"
            "fmla	v16.4s,  v6.4s,  v0.s[2]\n" /* out18 = b2 * a00[2], b2 =q6*/
            "fmla	v19.4s,  v6.4s,  v0.s[3]\n" /* out19 = b2 * a00[3], b2 =q6*/
            "fmla	v22.4s,  v6.4s,  v1.s[0]\n" /* out20 = b2 * a00[0], b2 =q6*/
            "fmla	v25.4s,  v6.4s,  v1.s[1]\n" /* out21 = b2 * a00[1], b2 =q6*/
            "fmla	v28.4s,  v6.4s,  v1.s[2]\n" /* out22 = b2 * a00[2], b2 =q6*/
            "fmla	v31.4s,  v6.4s,  v1.s[3]\n" /* out23 = b2 * a00[3], b2 =q6*/

            "ldp	q0, q1, [%[a_ptr]], #32\n" /* load a00, a01 to q0, q1 */

            /* unrool 1 */
            "fmla 	v8.4s ,  v7.4s,  v2.s[0]\n" /* out0 = b0 * a10[0], b0 =q7 */
            "fmla	v11.4s ,  v7.4s,  v2.s[1]\n"  /* out1 = b0 * a10[1], b0 =q7 */
            "fmla	v14.4s,  v7.4s,  v2.s[2]\n"   /* out2 = b0 * a10[2], b0 =q7 */
            "prfm   pldl1keep, [%[a_ptr], #256]\n"
            "fmla	v17.4s,  v7.4s,  v2.s[3]\n"   /* out3 = b0 * a10[3], b0 =q7 */
            "fmla 	v20.4s,  v7.4s,  v3.s[0]\n" /* out4 = b0 * a11[0], b0 =q7 */
            "fmla   v23.4s,  v7.4s,  v3.s[1]\n" /* out5 = b0 * a11[1], b0 = q7*/
            "fmla	v26.4s,  v7.4s,  v3.s[2]\n"   /* out6 = b0 * a11[2], b0 =q7 */
            "fmla	v29.4s,  v7.4s,  v3.s[3]\n"   /* out7 = b0 * a11[3], b0 =q7 */

            "ldp	q6, q7, [%[b_ptr]], #32\n" /* load b0, b1 to q6, q7 */

            "fmla	v9.4s,  v4.4s,  v2.s[0]\n"  /* out8 = b0 * a10[0], b1 =q4 */
            "fmla	v12.4s,  v4.4s,  v2.s[1]\n" /* out9 = b0 * a10[1], b1 =q4 */
            "fmla	v15.4s,  v4.4s,  v2.s[2]\n" /* out10 = b1 * a10[2], b1 =q4*/
            "fmla	v18.4s,  v4.4s,  v2.s[3]\n" /* out11 = b1 * a10[3], b1 =q4*/
            "fmla	v21.4s,  v4.4s,  v3.s[0]\n" /* out12 = b1 * a10[0], b1 =q4*/
            "fmla	v24.4s,  v4.4s,  v3.s[1]\n" /* out13 = b1 * a10[1], b1 =q4*/
            "fmla	v27.4s,  v4.4s,  v3.s[2]\n" /* out14 = b1 * a10[2], b1 =q4*/
            "fmla	v30.4s,  v4.4s,  v3.s[3]\n" /* out15 = b1 * a10[3], b1 =q4*/

            "fmla	v10.4s,  v5.4s,  v2.s[0]\n" /* out16 = b2 * a10[0], b2 =q5*/
            "fmla	v13.4s,  v5.4s,  v2.s[1]\n" /* out17 = b2 * a10[0], b2 =q5*/
            "fmla	v16.4s,  v5.4s,  v2.s[2]\n" /* out18 = b2 * a10[0], b2 =q5*/
            "fmla	v19.4s,  v5.4s,  v2.s[3]\n" /* out19 = b2 * a10[0], b2 =q5*/
            "fmla	v22.4s,  v5.4s,  v3.s[0]\n" /* out20 = b2 * a10[0], b2 =q5*/
            "fmla	v25.4s,  v5.4s,  v3.s[1]\n" /* out21 = b2 * a10[0], b2 =q5*/
            "fmla	v28.4s,  v5.4s,  v3.s[2]\n" /* out22 = b2 * a10[0], b2 =q5*/
            "fmla	v31.4s,  v5.4s,  v3.s[3]\n" /* out23 = b2 * a10[0], b2 =q5*/
            "ldp	q4, q5, [%[b_ptr]], #32\n"  /* load b2, b0 to q4, q5 */
            /* unrool 2*/
            "fmla 	v8.4s ,  v6.4s,  v0.s[0]\n" /* out0 = b0 * a00[0], b0 =q6 */
            "fmla  	v11.4s ,  v6.4s,  v0.s[1]\n" /* out1 = b0 * a00[1], b0 =q6
                                                    */
            "ldp	q2, q3, [%[a_ptr]], #32\n"     /* load a10, a11 to q3, q4*/
            "fmla	v14.4s,  v6.4s,  v0.s[2]\n"    /* out2 = b0 * a00[2], b0 =q6*/
            "fmla	v17.4s,  v6.4s,  v0.s[3]\n"    /* out3 = b0 * a00[3], b0 =q6*/
            "fmla 	v20.4s,  v6.4s,  v1.s[0]\n"  /* out4 = b0 * a01[0], b0 =q6*/
            "fmla	v23.4s,  v6.4s,  v1.s[1]\n"    /* out5 = b0 * a01[1], b0 =q6*/
            "fmla	v26.4s,  v6.4s,  v1.s[2]\n"    /* out6 = b0 * a01[2], b0 =q6*/
            "fmla	v29.4s,  v6.4s,  v1.s[3]\n"    /* out7 = b0 * a01[3], b0 =q6*/
            "fmla	v9.4s,  v7.4s,  v0.s[0]\n"     /* out8 = b1 * a00[0], b1 =q7*/
            "fmla	v12.4s,  v7.4s,  v0.s[1]\n"    /* out9 = b1 * a00[1], b1 =q7*/
            "prfm   pldl1keep, [%[b_ptr], #384]\n"
            "fmla	v15.4s,  v7.4s,  v0.s[2]\n" /* out10 = b1 * a00[2], b1 =q7*/
            "fmla	v18.4s,  v7.4s,  v0.s[3]\n" /* out11 = b1 * a00[3], b1 =q7*/
            "fmla	v21.4s,  v7.4s,  v1.s[0]\n" /* out12 = b1 * a01[0], b1 =q7*/
            "fmla	v24.4s,  v7.4s,  v1.s[1]\n" /* out13 = b1 * a01[1], b1 =q7*/
            "fmla	v27.4s,  v7.4s,  v1.s[2]\n" /* out14 = b1 * a01[2], b1 =q7*/
            "fmla	v30.4s,  v7.4s,  v1.s[3]\n" /* out15 = b1 * a01[3], b1 =q7*/

            "ldp	q6, q7, [%[b_ptr]], #32\n" /* load b1, b2 to q6, q7*/

            "fmla	v10.4s,  v4.4s,  v0.s[0]\n" /* out16 = b2 * a00[0], b2 =q4*/
            "fmla	v13.4s,  v4.4s,  v0.s[1]\n" /* out17 = b2 * a00[1], b2 =q4*/
            "fmla	v16.4s,  v4.4s,  v0.s[2]\n" /* out18 = b2 * a00[2], b2 =q4*/
            "fmla	v19.4s,  v4.4s,  v0.s[3]\n" /* out19 = b2 * a00[3], b2 =q4*/
            "fmla	v22.4s,  v4.4s,  v1.s[0]\n" /* out20 = b2 * a00[0], b2 =q4*/
            "fmla	v25.4s,  v4.4s,  v1.s[1]\n" /* out21 = b2 * a00[1], b2 =q4*/
            "fmla	v28.4s,  v4.4s,  v1.s[2]\n" /* out22 = b2 * a00[2], b2 =q4*/
            "fmla	v31.4s,  v4.4s,  v1.s[3]\n" /* out23 = b2 * a00[3], b2 =q4*/
            "ldp	q0, q1, [%[a_ptr]], #32\n"  /* load a00, a01 to q0, q1*/
            /* unrool 3*/
            "fmla 	v8.4s ,  v5.4s,  v2.s[0]\n" /* out0 = b0 * a10[0], b0 =q5*/
            "fmla	v11.4s ,  v5.4s,  v2.s[1]\n"  /* out1 = b0 * a10[1], b0 =q5*/
            "fmla	v14.4s,  v5.4s,  v2.s[2]\n"   /* out2 = b0 * a10[2], b0 =q5*/
            "fmla	v17.4s,  v5.4s,  v2.s[3]\n"   /* out3 = b0 * a10[3], b0 =q5*/
            "fmla 	v20.4s,  v5.4s,  v3.s[0]\n" /* out4 = b0 * a11[0], b0 =q5*/
            "fmla   v23.4s,  v5.4s,  v3.s[1]\n" /* out5 = b0 * a11[1], b0 =q5*/
            "fmla	v26.4s,  v5.4s,  v3.s[2]\n"   /* out6 = b0 * a11[2], b0 =q5*/
            "fmla	v29.4s,  v5.4s,  v3.s[3]\n"   /* out7 = b0 * a11[3], b0 =q5*/
            "ldp	q4, q5, [%[b_ptr]], #32\n"    /* load b0, b1 to q4, q5*/
            "fmla	v9.4s,  v6.4s,  v2.s[0]\n"    /* out8 = b0 * a10[0], b1 =q6*/
            "fmla	v12.4s,  v6.4s,  v2.s[1]\n"   /* out9 = b0 * a10[1], b1 =q6*/
            "prfm   pldl1keep, [%[a_ptr], #256]\n"
            "fmla	v15.4s,  v6.4s,  v2.s[2]\n" /* out10 = b1 * a10[2], b1 =q6*/
            "fmla	v18.4s,  v6.4s,  v2.s[3]\n" /* out11 = b1 * a10[3], b1 =q6*/
            "fmla	v21.4s,  v6.4s,  v3.s[0]\n" /* out12 = b1 * a10[0], b1 =q6*/
            "fmla	v24.4s,  v6.4s,  v3.s[1]\n" /* out13 = b1 * a10[1], b1 =q6*/
            "fmla	v27.4s,  v6.4s,  v3.s[2]\n" /* out14 = b1 * a10[2], b1 =q6*/
            "prfm   pldl1keep, [%[b_ptr], #384]\n"
            "fmla	v30.4s,  v6.4s,  v3.s[3]\n" /* out15 = b1 * a10[3], b1 =q6*/
            "fmla	v10.4s,  v7.4s,  v2.s[0]\n" /* out16 = b2 * a10[0], b2 =q7*/
            "fmla	v13.4s,  v7.4s,  v2.s[1]\n" /* out17 = b2 * a10[0], b2 =q7*/
            "fmla	v16.4s,  v7.4s,  v2.s[2]\n" /* out18 = b2 * a10[0], b2 =q7*/
            "fmla	v19.4s,  v7.4s,  v2.s[3]\n" /* out19 = b2 * a10[0], b2 =q7*/
            "fmla	v22.4s,  v7.4s,  v3.s[0]\n" /* out20 = b2 * a10[0], b2 =q7*/
            "fmla	v25.4s,  v7.4s,  v3.s[1]\n" /* out21 = b2 * a10[0], b2 =q7*/
            "subs	%w[k], %w[k], #1\n"         /* loop count - 1*/
            "fmla	v28.4s,  v7.4s,  v3.s[2]\n" /* out22 = b2 * a10[0], b2 =q7*/
            "fmla	v31.4s,  v7.4s,  v3.s[3]\n" /* out23 = b2 * a10[0], b2 =q7*/
            "bne	1b\n"
            "2:\n"                            /* process tail*/
            "subs		%w[tail], %w[tail], #1\n" /* tail--*/
            "beq		3f\n"                     /*jump to tail = 1*/
            /* final unrool 0*/
            /* unrool 0, tail > 1*/
            "fmla 	v8.4s ,  v4.4s,  v0.s[0]\n"  /* out0 = b0 * a00[0], b0 =q4*/
            "fmla  	v11.4s ,  v4.4s,  v0.s[1]\n" /* out1 = b0 * a00[1], b0 =q4*/
            "ldp	q6, q7, [%[b_ptr]], #32\n"     /* load b2, b0 to q6, q7*/
            "fmla	v14.4s,  v4.4s,  v0.s[2]\n"    /* out2 = b0 * a00[2], b0 =q4*/
            "fmla	v17.4s,  v4.4s,  v0.s[3]\n"    /* out3 = b0 * a00[3], b0 =q4*/
            "ldp	q2, q3, [%[a_ptr]], #32\n"     /* load a10, a11 to q2, q3*/
            "fmla 	v20.4s,  v4.4s,  v1.s[0]\n"  /* out4 = b0 * a01[0], b0 =q4*/
            "fmla	v23.4s,  v4.4s,  v1.s[1]\n"    /* out5 = b0 * a01[1], b0 =q4*/
            "fmla	v26.4s,  v4.4s,  v1.s[2]\n"    /* out6 = b0 * a01[2], b0 =q4*/
            "fmla	v29.4s,  v4.4s,  v1.s[3]\n"    /* out7 = b0 * a01[3], b0 =q4*/
            "subs	%w[tail], %w[tail], #1\n"      /* tail--*/
            "fmla	v9.4s,  v5.4s,  v0.s[0]\n"     /* out8 = b1 * a00[0], b1 =q5*/
            "fmla	v12.4s,  v5.4s,  v0.s[1]\n"    /* out9 = b1 * a00[1], b1 =q5*/
            "fmla	v15.4s,  v5.4s,  v0.s[2]\n" /* out10 = b1 * a00[2], b1 =q5*/
            "fmla	v18.4s,  v5.4s,  v0.s[3]\n" /* out11 = b1 * a00[3], b1 =q5*/
            "fmla	v21.4s,  v5.4s,  v1.s[0]\n" /* out12 = b1 * a01[0], b1 =q5*/
            "fmla	v24.4s,  v5.4s,  v1.s[1]\n" /* out13 = b1 * a01[1], b1 =q5*/
            "fmla	v27.4s,  v5.4s,  v1.s[2]\n" /* out14 = b1 * a01[2], b1 =q5*/
            "fmla	v30.4s,  v5.4s,  v1.s[3]\n" /* out15 = b1 * a01[3], b1 =q5*/
            "ldp	q4, q5, [%[b_ptr]], #32\n"  /* load b1, b2 to q4, q5*/
            "fmla	v10.4s,  v6.4s,  v0.s[0]\n" /* out16 = b2 * a00[0], b2 =q6*/
            "fmla	v13.4s,  v6.4s,  v0.s[1]\n" /* out17 = b2 * a00[1], b2 =q6*/
            "fmla	v16.4s,  v6.4s,  v0.s[2]\n" /* out18 = b2 * a00[2], b2 =q6*/
            "fmla	v19.4s,  v6.4s,  v0.s[3]\n" /* out19 = b2 * a00[3], b2 =q6*/
            "fmla	v22.4s,  v6.4s,  v1.s[0]\n" /* out20 = b2 * a00[0], b2 =q6*/
            "fmla	v25.4s,  v6.4s,  v1.s[1]\n" /* out21 = b2 * a00[1], b2 =q6*/
            "fmla	v28.4s,  v6.4s,  v1.s[2]\n" /* out22 = b2 * a00[2], b2 =q6*/
            "fmla	v31.4s,  v6.4s,  v1.s[3]\n" /* out23 = b2 * a00[3], b2 =q6*/
            "beq		4f\n"                     /*jump to tail = 2*/
            /* unrool 1, tail > 2*/
            "ldp	q0, q1, [%[a_ptr]], #32\n"    /* load a00, a01 to q0, q1*/
            "fmla 	v8.4s ,  v7.4s,  v2.s[0]\n" /* out0 = b0 * a10[0], b0 =q7*/
            "fmla	v11.4s ,  v7.4s,  v2.s[1]\n"  /* out1 = b0 * a10[1], b0 =q7*/
            "fmla	v14.4s,  v7.4s,  v2.s[2]\n"   /* out2 = b0 * a10[2], b0 =q7*/
            "fmla	v17.4s,  v7.4s,  v2.s[3]\n"   /* out3 = b0 * a10[3], b0 =q7*/
            "fmla 	v20.4s,  v7.4s,  v3.s[0]\n" /* out4 = b0 * a11[0], b0 =q7*/
            "fmla   v23.4s,  v7.4s,  v3.s[1]\n" /* out5 = b0 * a11[1], b0 =q7*/
            "fmla	v26.4s,  v7.4s,  v3.s[2]\n"   /* out6 = b0 * a11[2], b0 =q7*/
            "fmla	v29.4s,  v7.4s,  v3.s[3]\n"   /* out7 = b0 * a11[3], b0 =q7*/
            "ldp	q6, q7, [%[b_ptr]], #32\n"    /* load b0, b1 to q6, q7*/
            "fmla	v9.4s,  v4.4s,  v2.s[0]\n"    /* out8 = b0 * a10[0], b1 =q4*/
            "fmla	v12.4s,  v4.4s,  v2.s[1]\n"   /* out9 = b0 * a10[1], b1 =q4*/
            "fmla	v15.4s,  v4.4s,  v2.s[2]\n"   /* out10 = b1 * a10[2], b1 =q4*/
            "fmla	v18.4s,  v4.4s,  v2.s[3]\n"   /* out11 = b1 * a10[3], b1 =q4*/
            "fmla	v21.4s,  v4.4s,  v3.s[0]\n"   /* out12 = b1 * a10[0], b1 =q4*/
            "fmla	v24.4s,  v4.4s,  v3.s[1]\n"   /* out13 = b1 * a10[1], b1 =q4*/
            "fmla	v27.4s,  v4.4s,  v3.s[2]\n"   /* out14 = b1 * a10[2], b1 =q4*/
            "fmla	v30.4s,  v4.4s,  v3.s[3]\n"   /* out15 = b1 * a10[3], b1 =q4*/
            "subs	%w[tail], %w[tail], #1\n"     /* tail--*/
            "fmla	v10.4s,  v5.4s,  v2.s[0]\n"   /* out16 = b2 * a10[0], b2 =q5*/
            "fmla	v13.4s,  v5.4s,  v2.s[1]\n"   /* out17 = b2 * a10[0], b2 =q5*/
            "fmla	v16.4s,  v5.4s,  v2.s[2]\n"   /* out18 = b2 * a10[0], b2 =q5*/
            "fmla	v19.4s,  v5.4s,  v2.s[3]\n"   /* out19 = b2 * a10[0], b2 =q5*/
            "fmla	v22.4s,  v5.4s,  v3.s[0]\n"   /* out20 = b2 * a10[0], b2 =q5*/
            "fmla	v25.4s,  v5.4s,  v3.s[1]\n"   /* out21 = b2 * a10[0], b2 =q5*/
            "fmla	v28.4s,  v5.4s,  v3.s[2]\n"   /* out22 = b2 * a10[0], b2 =q5*/
            "fmla	v31.4s,  v5.4s,  v3.s[3]\n"   /* out23 = b2 * a10[0], b2 =q5*/
            "beq		5f\n"                       /*jump to tail = 3*/
            /* unrool 2, tail = 4*/
            "ldp	q4, q5, [%[b_ptr]], #32\n"     /* load b2, b0 to q4, q5*/
            "fmla 	v8.4s ,  v6.4s,  v0.s[0]\n"  /* out0 = b0 * a00[0], b0 =q6*/
            "fmla  	v11.4s ,  v6.4s,  v0.s[1]\n" /* out1 = b0 * a00[1], b0 =q6*/
            "ldp	q2, q3, [%[a_ptr]], #32\n"     /* load a10, a11 to q3, q4*/
            "fmla	v14.4s,  v6.4s,  v0.s[2]\n"    /* out2 = b0 * a00[2], b0 =q6*/
            "fmla	v17.4s,  v6.4s,  v0.s[3]\n"    /* out3 = b0 * a00[3], b0 =q6*/
            "fmla 	v20.4s,  v6.4s,  v1.s[0]\n"  /* out4 = b0 * a01[0], b0 =q6*/
            "fmla	v23.4s,  v6.4s,  v1.s[1]\n"    /* out5 = b0 * a01[1], b0 =q6*/
            "fmla	v26.4s,  v6.4s,  v1.s[2]\n"    /* out6 = b0 * a01[2], b0 =q6*/
            "fmla	v29.4s,  v6.4s,  v1.s[3]\n"    /* out7 = b0 * a01[3], b0 =q6*/
            "fmla	v9.4s,  v7.4s,  v0.s[0]\n"     /* out8 = b1 * a00[0], b1 =q7*/
            "fmla	v12.4s,  v7.4s,  v0.s[1]\n"    /* out9 = b1 * a00[1], b1 =q7*/
            "fmla	v15.4s,  v7.4s,  v0.s[2]\n" /* out10 = b1 * a00[2], b1 =q7*/
            "fmla	v18.4s,  v7.4s,  v0.s[3]\n" /* out11 = b1 * a00[3], b1 =q7*/
            "fmla	v21.4s,  v7.4s,  v1.s[0]\n" /* out12 = b1 * a01[0], b1 =q7*/
            "fmla	v24.4s,  v7.4s,  v1.s[1]\n" /* out13 = b1 * a01[1], b1 =q7*/
            "fmla	v27.4s,  v7.4s,  v1.s[2]\n" /* out14 = b1 * a01[2], b1 =q7*/
            "fmla	v30.4s,  v7.4s,  v1.s[3]\n" /* out15 = b1 * a01[3], b1 =q7*/
            "ldp	q6, q7, [%[b_ptr]], #32\n"  /* load b1, b2 to q6, q7*/
            "fmla	v10.4s,  v4.4s,  v0.s[0]\n" /* out16 = b2 * a00[0], b2 =q4*/
            "fmla	v13.4s,  v4.4s,  v0.s[1]\n" /* out17 = b2 * a00[1], b2 =q4*/
            "fmla	v16.4s,  v4.4s,  v0.s[2]\n" /* out18 = b2 * a00[2], b2 =q4*/
            "fmla	v19.4s,  v4.4s,  v0.s[3]\n" /* out19 = b2 * a00[3], b2 =q4*/
            "fmla	v22.4s,  v4.4s,  v1.s[0]\n" /* out20 = b2 * a00[0], b2 =q4*/
            "fmla	v25.4s,  v4.4s,  v1.s[1]\n" /* out21 = b2 * a00[1], b2 =q4*/
            "fmla	v28.4s,  v4.4s,  v1.s[2]\n" /* out22 = b2 * a00[2], b2 =q4*/
            "fmla	v31.4s,  v4.4s,  v1.s[3]\n" /* out23 = b2 * a00[3], b2 =q4*/
            /* unrool 3, tail = 4*/
            "fmla 	v8.4s ,  v5.4s,  v2.s[0]\n" /* out0 = b0 * a10[0], b0 =q5*/
            "fmla	v11.4s ,  v5.4s,  v2.s[1]\n"  /* out1 = b0 * a10[1], b0 =q5*/
            "fmla	v14.4s,  v5.4s,  v2.s[2]\n"   /* out2 = b0 * a10[2], b0 =q5*/
            "fmla	v17.4s,  v5.4s,  v2.s[3]\n"   /* out3 = b0 * a10[3], b0 =q5*/
            "fmla 	v20.4s,  v5.4s,  v3.s[0]\n" /* out4 = b0 * a11[0], b0 =q5*/
            "fmla   v23.4s,  v5.4s,  v3.s[1]\n" /* out5 = b0 * a11[1], b0 =q5*/
            "fmla	v26.4s,  v5.4s,  v3.s[2]\n"   /* out6 = b0 * a11[2], b0 =q5*/
            "fmla	v29.4s,  v5.4s,  v3.s[3]\n"   /* out7 = b0 * a11[3], b0 =q5*/
            "fmla	v9.4s,  v6.4s,  v2.s[0]\n"    /* out8 = b0 * a10[0], b1 =q6*/
            "fmla	v12.4s,  v6.4s,  v2.s[1]\n"   /* out9 = b1 * a10[1], b1 =q6*/
            "fmla	v15.4s,  v6.4s,  v2.s[2]\n"   /* out10 = b1 * a10[2], b1 =q6*/
            "fmla	v18.4s,  v6.4s,  v2.s[3]\n"   /* out11 = b1 * a10[3], b1 =q6*/
            "fmla	v21.4s,  v6.4s,  v3.s[0]\n"   /* out12 = b1 * a10[0], b1 =q6*/
            "fmla	v24.4s,  v6.4s,  v3.s[1]\n"   /* out13 = b1 * a10[1], b1 =q6*/
            "fmla	v27.4s,  v6.4s,  v3.s[2]\n"   /* out14 = b1 * a10[2], b1 =q6*/
            "fmla	v30.4s,  v6.4s,  v3.s[3]\n"   /* out15 = b1 * a10[3], b1 =q6*/
            "fmla	v10.4s,  v7.4s,  v2.s[0]\n"   /* out16 = b2 * a10[0], b2 =q7*/
            "fmla	v13.4s,  v7.4s,  v2.s[1]\n"   /* out17 = b2 * a10[0], b2 =q7*/
            "fmla	v16.4s,  v7.4s,  v2.s[2]\n"   /* out18 = b2 * a10[0], b2 =q7*/
            "fmla	v19.4s,  v7.4s,  v2.s[3]\n"   /* out19 = b2 * a10[0], b2 =q7*/
            "fmla	v22.4s,  v7.4s,  v3.s[0]\n"   /* out20 = b2 * a10[0], b2 =q7*/
            "fmla	v25.4s,  v7.4s,  v3.s[1]\n"   /* out21 = b2 * a10[0], b2 =q7*/
            "fmla	v28.4s,  v7.4s,  v3.s[2]\n"   /* out22 = b2 * a10[0], b2 =q7*/
            "fmla	v31.4s,  v7.4s,  v3.s[3]\n"   /* out23 = b2 * a10[0], b2 =q7*/
            "b		11f\n"
            /* tails==1 final tail*/
            "3: \n"                            /* tail=1*/
            "ldr	q6, [%[b_ptr]], #16\n"       /* load b2 to q6*/
            "fmla v8.4s ,  v4.4s,  v0.s[0]\n"  /* out0 = b0 * a10[0], b0 =q5*/
            "fmla	v11.4s ,  v4.4s,  v0.s[1]\n" /* out1 = b0 * a10[1], b0 =q5*/
            "fmla	v14.4s,  v4.4s,  v0.s[2]\n"  /* out2 = b0 * a10[2], b0 =q5*/
            "fmla	v17.4s,  v4.4s,  v0.s[3]\n"  /* out3 = b0 * a10[3], b0 =q5*/
            "fmla v20.4s,  v4.4s,  v1.s[0]\n"  /* out4 = b0 * a11[0], b0 =q5*/
            "fmla v23.4s,  v4.4s,  v1.s[1]\n"  /* out5 = b0 * a11[1], b0 = q5*/
            "fmla	v26.4s,  v4.4s,  v1.s[2]\n"  /* out6 = b0 * a11[2], b0 =q5*/
            "fmla	v29.4s,  v4.4s,  v1.s[3]\n"  /* out7 = b0 * a11[3], b0 =q5*/
            "fmla	v9.4s,  v5.4s,  v0.s[0]\n"   /* out8 = b0 * a10[0], b1 =q6*/
            "fmla	v12.4s,  v5.4s,  v0.s[1]\n"  /* out9 = b1 * a10[1], b1 =q6*/
            "fmla	v15.4s,  v5.4s,  v0.s[2]\n"  /* out10 = b1 * a10[2], b1 =q6*/
            "fmla	v18.4s,  v5.4s,  v0.s[3]\n"  /* out11 = b1 * a10[3], b1 =q6*/
            "fmla	v21.4s,  v5.4s,  v1.s[0]\n"  /* out12 = b1 * a10[0], b1 =q6*/
            "fmla	v24.4s,  v5.4s,  v1.s[1]\n"  /* out13 = b1 * a10[1], b1 =q6*/
            "fmla	v27.4s,  v5.4s,  v1.s[2]\n"  /* out14 = b1 * a10[2], b1 =q6*/
            "fmla	v30.4s,  v5.4s,  v1.s[3]\n"  /* out15 = b1 * a10[3], b1 =q6*/
            "fmla	v10.4s,  v6.4s,  v0.s[0]\n"  /* out16 = b2 * a10[0], b2 =q7*/
            "fmla	v13.4s,  v6.4s,  v0.s[1]\n"  /* out17 = b2 * a10[0], b2 =q7*/
            "fmla	v16.4s,  v6.4s,  v0.s[2]\n"  /* out18 = b2 * a10[0], b2 =q7*/
            "fmla	v19.4s,  v6.4s,  v0.s[3]\n"  /* out19 = b2 * a10[0], b2 =q7*/
            "fmla	v22.4s,  v6.4s,  v1.s[0]\n"  /* out20 = b2 * a10[0], b2 =q7*/
            "fmla	v25.4s,  v6.4s,  v1.s[1]\n"  /* out21 = b2 * a10[0], b2 =q7*/
            "fmla	v28.4s,  v6.4s,  v1.s[2]\n"  /* out22 = b2 * a10[0], b2 =q7*/
            "fmla	v31.4s,  v6.4s,  v1.s[3]\n"  /* out23 = b2 * a10[0], b2 =q7*/
            "b		11f\n"
            /* tails==2 final tail*/
            "4:\n"                              /* tail = 2*/
            "fmla 	v8.4s ,  v7.4s,  v2.s[0]\n" /* out0 = b0 * a10[0], b0 =q5*/
            "fmla	v11.4s ,  v7.4s,  v2.s[1]\n"  /* out1 = b0 * a10[1], b0 =q5*/
            "fmla	v14.4s,  v7.4s,  v2.s[2]\n"   /* out2 = b0 * a10[2], b0 =q5*/
            "fmla	v17.4s,  v7.4s,  v2.s[3]\n"   /* out3 = b0 * a10[3], b0 =q5*/
            "fmla 	v20.4s,  v7.4s,  v3.s[0]\n" /* out4 = b0 * a11[0], b0 =q5*/
            "fmla   v23.4s,  v7.4s,  v3.s[1]\n" /* out5 = b0 * a11[1], b0 = q5*/
            "fmla	v26.4s,  v7.4s,  v3.s[2]\n"   /* out6 = b0 * a11[2], b0 =q5*/
            "fmla	v29.4s,  v7.4s,  v3.s[3]\n"   /* out7 = b0 * a11[3], b0 =q5*/
            "fmla	v9.4s,  v4.4s,  v2.s[0]\n"    /* out8 = b0 * a10[0], b1 =q6*/
            "fmla	v12.4s,  v4.4s,  v2.s[1]\n"   /* out9 = b1 * a10[1], b1 =q6*/
            "fmla	v15.4s,  v4.4s,  v2.s[2]\n"   /* out10 = b1 * a10[2], b1 =q6*/
            "fmla	v18.4s,  v4.4s,  v2.s[3]\n"   /* out11 = b1 * a10[3], b1 =q6*/
            "fmla	v21.4s,  v4.4s,  v3.s[0]\n"   /* out12 = b1 * a10[0], b1 =q6*/
            "fmla	v24.4s,  v4.4s,  v3.s[1]\n"   /* out13 = b1 * a10[1], b1 =q6*/
            "fmla	v27.4s,  v4.4s,  v3.s[2]\n"   /* out14 = b1 * a10[2], b1 =q6*/
            "fmla	v30.4s,  v4.4s,  v3.s[3]\n"   /* out15 = b1 * a10[3], b1 =q6*/
            "fmla	v10.4s,  v5.4s,  v2.s[0]\n"   /* out16 = b2 * a10[0], b2 =q7*/
            "fmla	v13.4s,  v5.4s,  v2.s[1]\n"   /* out17 = b2 * a10[0], b2 =q7*/
            "fmla	v16.4s,  v5.4s,  v2.s[2]\n"   /* out18 = b2 * a10[0], b2 =q7*/
            "fmla	v19.4s,  v5.4s,  v2.s[3]\n"   /* out19 = b2 * a10[0], b2 =q7*/
            "fmla	v22.4s,  v5.4s,  v3.s[0]\n"   /* out20 = b2 * a10[0], b2 =q7*/
            "fmla	v25.4s,  v5.4s,  v3.s[1]\n"   /* out21 = b2 * a10[0], b2 =q7*/
            "fmla	v28.4s,  v5.4s,  v3.s[2]\n"   /* out22 = b2 * a10[0], b2 =q7*/
            "fmla	v31.4s,  v5.4s,  v3.s[3]\n"   /* out23 = b2 * a10[0], b2 =q7*/
            "b		11f\n"
            /* tails==3 final tail*/
            "5:\n"                              /* tail = 3*/
            "ldr	q4, [%[b_ptr]], #16\n"        /* load b2, b0 to q4*/
            "fmla 	v8.4s ,  v6.4s,  v0.s[0]\n" /* out0 = b0 * a10[0], b0 =q5*/
            "fmla	v11.4s ,  v6.4s,  v0.s[1]\n"  /* out1 = b0 * a10[1], b0 =q5*/
            "fmla	v14.4s,  v6.4s,  v0.s[2]\n"   /* out2 = b0 * a10[2], b0 =q5*/
            "fmla	v17.4s,  v6.4s,  v0.s[3]\n"   /* out3 = b0 * a10[3], b0 =q5*/
            "fmla 	v20.4s,  v6.4s,  v1.s[0]\n" /* out4 = b0 * a11[0], b0 =q5*/
            "fmla   v23.4s,  v6.4s,  v1.s[1]\n" /* out5 = b0 * a11[1], b0 = q5*/
            "fmla	v26.4s,  v6.4s,  v1.s[2]\n"   /* out6 = b0 * a11[2], b0 =q5*/
            "fmla	v29.4s,  v6.4s,  v1.s[3]\n"   /* out7 = b0 * a11[3], b0 =q5*/
            "fmla	v9.4s,  v7.4s,  v0.s[0]\n"    /* out8 = b0 * a10[0], b1 =q6*/
            "fmla	v12.4s,  v7.4s,  v0.s[1]\n"   /* out9 = b1 * a10[1], b1 =q6*/
            "fmla	v15.4s,  v7.4s,  v0.s[2]\n"   /* out10 = b1 * a10[2], b1 =q6*/
            "fmla	v18.4s,  v7.4s,  v0.s[3]\n"   /* out11 = b1 * a10[3], b1 =q6*/
            "fmla	v21.4s,  v7.4s,  v1.s[0]\n"   /* out12 = b1 * a10[0], b1 =q6*/
            "fmla	v24.4s,  v7.4s,  v1.s[1]\n"   /* out13 = b1 * a10[1], b1 =q6*/
            "fmla	v27.4s,  v7.4s,  v1.s[2]\n"   /* out14 = b1 * a10[2], b1 =q6*/
            "fmla	v30.4s,  v7.4s,  v1.s[3]\n"   /* out15 = b1 * a10[3], b1 =q6*/
            "fmla	v10.4s,  v4.4s,  v0.s[0]\n"   /* out16 = b2 * a10[0], b2 =q7*/
            "fmla	v13.4s,  v4.4s,  v0.s[1]\n"   /* out17 = b2 * a10[0], b2 =q7*/
            "fmla	v16.4s,  v4.4s,  v0.s[2]\n"   /* out18 = b2 * a10[0], b2 =q7*/
            "fmla	v19.4s,  v4.4s,  v0.s[3]\n"   /* out19 = b2 * a10[0], b2 =q7*/
            "fmla	v22.4s,  v4.4s,  v1.s[0]\n"   /* out20 = b2 * a10[0], b2 =q7*/
            "fmla	v25.4s,  v4.4s,  v1.s[1]\n"   /* out21 = b2 * a10[0], b2 =q7*/
            "fmla	v28.4s,  v4.4s,  v1.s[2]\n"   /* out22 = b2 * a10[0], b2 =q7*/
            "fmla	v31.4s,  v4.4s,  v1.s[3]\n"   /* out23 = b2 * a10[0], b2 =q7*/

            "11:                           \n"   /* check activation */
            "cmp    %w[flag_act],   #1     \n"   /* check if has relu */
            "bne    12f                    \n"   /* jump if no relu */
            "movi   v0.4s,  #0             \n"   /* for relu*/
            "fmax   v8.4s,  v8.4s,  v0.4s  \n"   /* relu*/
            "fmax   v9.4s,  v9.4s,  v0.4s  \n"   /* relu*/
            "fmax   v10.4s, v10.4s, v0.4s  \n"   /* relu*/
            "fmax   v11.4s, v11.4s, v0.4s  \n"   /* relu*/
            "fmax   v12.4s, v12.4s, v0.4s  \n"   /* relu*/
            "fmax   v13.4s, v13.4s, v0.4s  \n"   /* relu*/
            "fmax   v14.4s, v14.4s, v0.4s  \n"   /* relu*/
            "fmax   v15.4s, v15.4s, v0.4s  \n"   /* relu*/
            "fmax   v16.4s, v16.4s, v0.4s  \n"   /* relu*/
            "fmax   v17.4s, v17.4s, v0.4s  \n"   /* relu*/
            "fmax   v18.4s, v18.4s, v0.4s  \n"   /* relu*/
            "fmax   v19.4s, v19.4s, v0.4s  \n"   /* relu*/
            "fmax   v20.4s, v20.4s, v0.4s  \n"   /* relu*/
            "fmax   v21.4s, v21.4s, v0.4s  \n"   /* relu*/
            "fmax   v22.4s, v22.4s, v0.4s  \n"   /* relu*/
            "fmax   v23.4s, v23.4s, v0.4s  \n"   /* relu*/
            "fmax   v24.4s, v24.4s, v0.4s  \n"   /* relu*/
            "fmax   v25.4s, v25.4s, v0.4s  \n"   /* relu*/
            "fmax   v26.4s, v26.4s, v0.4s  \n"   /* relu*/
            "fmax   v27.4s, v27.4s, v0.4s  \n"   /* relu*/
            "fmax   v28.4s, v28.4s, v0.4s  \n"   /* relu*/
            "fmax   v29.4s, v29.4s, v0.4s  \n"   /* relu*/
            "fmax   v30.4s, v30.4s, v0.4s  \n"   /* relu*/
            "fmax   v31.4s, v31.4s, v0.4s  \n"   /* relu*/
            "b      20f                    \n"   /* relu end */
            //! no act
            "12:                           \n"   /* no relu */
            "cmp   %w[flag_act],  #0       \n"   /* check no act */
            "beq   20f                     \n"   /* no act end */
            //! relu6
            "cmp    %w[flag_act],  #2      \n"    /* check if has relu6 */
            "bne    13f                    \n"    /* jump if no relu6 */
            "movi   v0.4s, #0              \n"    /* for relu6 */
            "ld1    {v1.4s}, [%[alpha]]    \n"    /* relu6 alpha */
            "fmax   v8.4s,  v8.4s,  v0.4s  \n"    /* relu6 */
            "fmax   v9.4s,  v9.4s,  v0.4s  \n"    /* relu6 */
            "fmax   v10.4s, v10.4s, v0.4s  \n"    /* relu6 */
            "fmax   v11.4s, v11.4s, v0.4s  \n"    /* relu6 */
            "fmax   v12.4s, v12.4s, v0.4s  \n"    /* relu6 */
            "fmax   v13.4s, v13.4s, v0.4s  \n"    /* relu6 */
            "fmax   v14.4s, v14.4s, v0.4s  \n"    /* relu6 */
            "fmax   v15.4s, v15.4s, v0.4s  \n"    /* relu6 */
            "fmax   v16.4s, v16.4s, v0.4s  \n"    /* relu6 */
            "fmax   v17.4s, v17.4s, v0.4s  \n"    /* relu6 */
            "fmax   v18.4s, v18.4s, v0.4s  \n"    /* relu6 */
            "fmax   v19.4s, v19.4s, v0.4s  \n"    /* relu6 */
            "fmax   v20.4s, v20.4s, v0.4s  \n"    /* relu6 */
            "fmax   v21.4s, v21.4s, v0.4s  \n"    /* relu6 */
            "fmax   v22.4s, v22.4s, v0.4s  \n"    /* relu6 */
            "fmax   v23.4s, v23.4s, v0.4s  \n"    /* relu6 */
            "fmax   v24.4s, v24.4s, v0.4s  \n"    /* relu6 */
            "fmax   v25.4s, v25.4s, v0.4s  \n"    /* relu6 */
            "fmax   v26.4s, v26.4s, v0.4s  \n"    /* relu6 */
            "fmax   v27.4s, v27.4s, v0.4s  \n"    /* relu6 */
            "fmax   v28.4s, v28.4s, v0.4s  \n"    /* relu6 */
            "fmax   v29.4s, v29.4s, v0.4s  \n"    /* relu6 */
            "fmax   v30.4s, v30.4s, v0.4s  \n"    /* relu6 */
            "fmax   v31.4s, v31.4s, v0.4s  \n"    /* relu6 */
            "fmin   v8.4s,  v8.4s,  v1.4s  \n"    /* relu6 */
            "fmin   v9.4s,  v9.4s,  v1.4s  \n"    /* relu6 */
            "fmin   v10.4s, v10.4s, v1.4s  \n"    /* relu6 */
            "fmin   v11.4s, v11.4s, v1.4s  \n"    /* relu6 */
            "fmin   v12.4s, v12.4s, v1.4s  \n"    /* relu6 */
            "fmin   v13.4s, v13.4s, v1.4s  \n"    /* relu6 */
            "fmin   v14.4s, v14.4s, v1.4s  \n"    /* relu6 */
            "fmin   v15.4s, v15.4s, v1.4s  \n"    /* relu6 */
            "fmin   v16.4s, v16.4s, v1.4s  \n"    /* relu6 */
            "fmin   v17.4s, v17.4s, v1.4s  \n"    /* relu6 */
            "fmin   v18.4s, v18.4s, v1.4s  \n"    /* relu6 */
            "fmin   v19.4s, v19.4s, v1.4s  \n"    /* relu6 */
            "fmin   v20.4s, v20.4s, v1.4s  \n"    /* relu6 */
            "fmin   v21.4s, v21.4s, v1.4s  \n"    /* relu6 */
            "fmin   v22.4s, v22.4s, v1.4s  \n"    /* relu6 */
            "fmin   v23.4s, v23.4s, v1.4s  \n"    /* relu6 */
            "fmin   v24.4s, v24.4s, v1.4s  \n"    /* relu6 */
            "fmin   v25.4s, v25.4s, v1.4s  \n"    /* relu6 */
            "fmin   v26.4s, v26.4s, v1.4s  \n"    /* relu6 */
            "fmin   v27.4s, v27.4s, v1.4s  \n"    /* relu6 */
            "fmin   v28.4s, v28.4s, v1.4s  \n"    /* relu6 */
            "fmin   v29.4s, v29.4s, v1.4s  \n"    /* relu6 */
            "fmin   v30.4s, v30.4s, v1.4s  \n"    /* relu6 */
            "fmin   v31.4s, v31.4s, v1.4s  \n"    /* relu6 */
            "b      20f                    \n"    /* relu6 end */
            //! leakey relu
            "13:                                \n" /* otherwise is leakey relu */
            "movi   v0.4s,    #0                \n" /* for leakey relu */
            "ld1    {v1.4s},  [%[alpha]]        \n" /* leakey relu alpha */
            "fcmge  v2.4s,    v8.4s,    v0.4s   \n" /* vcgeq_f32 */
            "fmul   v3.4s,    v8.4s,    v1.4s   \n" /* vmulq_f32 */
            "fcmge  v4.4s,    v9.4s,    v0.4s   \n" /* vcgeq_f32 */
            "fmul   v5.4s,    v9.4s,    v1.4s   \n" /* vmulq_f32 */
            "fcmge  v6.4s,    v10.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v7.4s,    v10.4s,   v1.4s   \n" /* vmulq_f32 */
            "bif    v8.16b,   v3.16b,   v2.16b  \n" /* choose*/
            "bif    v9.16b,   v5.16b,   v4.16b  \n" /* choose*/
            "bif    v10.16b,  v7.16b,   v6.16b  \n" /* choose*/
            "fcmge  v2.4s,    v11.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v3.4s,    v11.4s,   v1.4s   \n" /* vmulq_f32 */
            "bif    v11.16b,  v3.16b,   v2.16b  \n" /* choose*/
            "fcmge  v2.4s,    v12.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v3.4s,    v12.4s,   v1.4s   \n" /* vmulq_f32 */
            "fcmge  v4.4s,    v13.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v5.4s,    v13.4s,   v1.4s   \n" /* vmulq_f32 */
            "fcmge  v6.4s,    v14.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v7.4s,    v14.4s,   v1.4s   \n" /* vmulq_f32 */
            "bif    v12.16b,  v3.16b,   v2.16b  \n" /* choose*/
            "bif    v13.16b,  v5.16b,   v4.16b  \n" /* choose*/
            "bif    v14.16b,  v7.16b,   v6.16b  \n" /* choose*/
            "fcmge  v2.4s,    v15.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v3.4s,    v15.4s,   v1.4s   \n" /* vmulq_f32 */
            "bif    v15.16b,  v3.16b,   v2.16b  \n" /* choose*/
            "fcmge  v2.4s,    v16.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v3.4s,    v16.4s,   v1.4s   \n" /* vmulq_f32 */
            "fcmge  v4.4s,    v17.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v5.4s,    v17.4s,   v1.4s   \n" /* vmulq_f32 */
            "fcmge  v6.4s,    v18.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v7.4s,    v18.4s,   v1.4s   \n" /* vmulq_f32 */
            "bif    v16.16b,  v3.16b,   v2.16b  \n" /* choose*/
            "bif    v17.16b,  v5.16b,   v4.16b  \n" /* choose*/
            "bif    v18.16b,  v7.16b,   v6.16b  \n" /* choose*/
            "fcmge  v2.4s,    v19.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v3.4s,    v19.4s,   v1.4s   \n" /* vmulq_f32 */
            "bif    v19.16b,  v3.16b,   v2.16b  \n" /* choose*/
            "fcmge  v2.4s,    v20.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v3.4s,    v20.4s,   v1.4s   \n" /* vmulq_f32 */
            "fcmge  v4.4s,    v21.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v5.4s,    v21.4s,   v1.4s   \n" /* vmulq_f32 */
            "fcmge  v6.4s,    v22.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v7.4s,    v22.4s,   v1.4s   \n" /* vmulq_f32 */
            "bif    v20.16b,  v3.16b,   v2.16b  \n" /* choose*/
            "bif    v21.16b,  v5.16b,   v4.16b  \n" /* choose*/
            "bif    v22.16b,  v7.16b,   v6.16b  \n" /* choose*/
            "fcmge  v2.4s,    v23.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v3.4s,    v23.4s,   v1.4s   \n" /* vmulq_f32 */
            "bif    v23.16b,  v3.16b,   v2.16b  \n" /* choose*/
            "fcmge  v2.4s,    v24.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v3.4s,    v24.4s,   v1.4s   \n" /* vmulq_f32 */
            "fcmge  v4.4s,    v25.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v5.4s,    v25.4s,   v1.4s   \n" /* vmulq_f32 */
            "fcmge  v6.4s,    v26.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v7.4s,    v26.4s,   v1.4s   \n" /* vmulq_f32 */
            "bif    v24.16b,  v3.16b,   v2.16b  \n" /* choose*/
            "bif    v25.16b,  v5.16b,   v4.16b  \n" /* choose*/
            "bif    v26.16b,  v7.16b,   v6.16b  \n" /* choose*/
            "fcmge  v2.4s,    v27.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v3.4s,    v27.4s,   v1.4s   \n" /* vmulq_f32 */
            "bif    v27.16b,  v3.16b,   v2.16b  \n" /* choose*/
            "fcmge  v2.4s,    v28.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v3.4s,    v28.4s,   v1.4s   \n" /* vmulq_f32 */
            "fcmge  v4.4s,    v29.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v5.4s,    v29.4s,   v1.4s   \n" /* vmulq_f32 */
            "fcmge  v6.4s,    v30.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v7.4s,    v30.4s,   v1.4s   \n" /* vmulq_f32 */
            "bif    v28.16b,  v3.16b,   v2.16b  \n" /* choose*/
            "bif    v29.16b,  v5.16b,   v4.16b  \n" /* choose*/
            "bif    v30.16b,  v7.16b,   v6.16b  \n" /* choose*/
            "fcmge  v2.4s,    v31.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v3.4s,    v31.4s,   v1.4s   \n" /* vmulq_f32 */
            "bif    v31.16b,  v3.16b,   v2.16b  \n" /* choose*/
            "20:                                \n" /* act end */

            "st1 {v8.4s, v9.4s, v10.4s},[%[c_ptr0]], #48\n"   /* store r0 */
            "st1 {v11.4s, v12.4s, v13.4s},[%[c_ptr1]], #48\n" /* store r1 */
            "st1 {v14.4s, v15.4s, v16.4s},[%[c_ptr2]], #48\n" /* store r2 */
            "st1 {v17.4s, v18.4s, v19.4s},[%[c_ptr3]], #48\n" /* store r3 */
            "st1 {v20.4s, v21.4s, v22.4s},[%[c_ptr4]], #48\n" /* store r4 */
            "st1 {v23.4s, v24.4s, v25.4s},[%[c_ptr5]], #48\n" /* store r5 */
            "st1 {v26.4s, v27.4s, v28.4s},[%[c_ptr6]], #48\n" /* store r6 */
            "st1 {v29.4s, v30.4s, v31.4s},[%[c_ptr7]], #48\n" /* store r7 */

            : [a_ptr] "+r"(a_ptr),
              [b_ptr] "+r"(b_ptr),
              [k] "+r"(k),
              [tail] "+r"(tail),
              [c_ptr0] "+r"(c_ptr0),
              [c_ptr1] "+r"(c_ptr1),
              [c_ptr2] "+r"(c_ptr2),
              [c_ptr3] "+r"(c_ptr3),
              [c_ptr4] "+r"(c_ptr4),
              [c_ptr5] "+r"(c_ptr5),
              [c_ptr6] "+r"(c_ptr6),
              [c_ptr7] "+r"(c_ptr7)
            : [bias_ptr] "r"(bias_local),
              [has_beta] "r"(has_beta),
              [beta] "r"(beta),
              [alpha] "r"(alpha),
              [flag_act] "r"(flag_act)
            : "cc","memory",
              "v0","v1","v2","v3","v4","v5","v6","v7",
              "v8","v9","v10","v11","v12","v13",
              "v14","v15","v16","v17","v18","v19",
              "v20","v21","v22","v23","v24","v25",
              "v26","v27","v28","v29","v30","v31");
        // clang-format on
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

void sgemm_prepacked_8x12_a53(bool is_transB,
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
  size_t l2_cache = ctx->llc_size() > 0 ? ctx->llc_size() : 512 * 1024;
  auto workspace = ctx->workspace_data<float>();
  int threads = ctx->threads();

  auto act_type = act_param.active_type;
  float alpha[4] = {0.f, 0.f, 0.f, 0.f};
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    }
  }
  //! MBLOCK * x (result) + MBLOCK * k (A) + x * k (B) = l2
  int x_block = (l2_cache - (MBLOCK * K)) / (sizeof(float) * (K + MBLOCK));
  x_block /= NBLOCK;
  x_block *= NBLOCK;
  int x_num = (N + (x_block - 1)) / x_block;
  x_block = (N + x_num - 1) / x_num;
  x_block = (x_block + NBLOCK - 1) / NBLOCK;
  x_block *= NBLOCK;
  x_block = x_block < NBLOCK ? NBLOCK : x_block;

  // unroll 2 loop
  int tail_pre = (K & (KBLOCK - 1));
  int k_pre = ((K + KBLOCK - 1) / KBLOCK) - 1;

  bool flag_p_remain = false;
  int remain = 0;
  if (tail_pre == 0) {
    tail_pre = KBLOCK;
  }

  int has_beta = fabsf(beta) > 1e-8f ? 1 : 0;

  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    if (xmax > N) {
      xmax = N;
    }
    int bblocks = (xmax - x0 + NBLOCK - 1) / NBLOCK;
    remain = xmax - x0 - (bblocks - 1) * NBLOCK;
    if (remain > 0) {
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
        bias_local[0] = bias[y];
        bias_local[1] = bias[y + 1];
        bias_local[2] = bias[y + 2];
        bias_local[3] = bias[y + 3];
        bias_local[4] = bias[y + 4];
        bias_local[5] = bias[y + 5];
        bias_local[6] = bias[y + 6];
        bias_local[7] = bias[y + 7];
      }

      float cout0[NBLOCK];
      float cout1[NBLOCK];
      float cout2[NBLOCK];
      float cout3[NBLOCK];
      float cout4[NBLOCK];
      float cout5[NBLOCK];
      float cout6[NBLOCK];
      float cout7[NBLOCK];

      float *c_ptr0 = C + y * ldc + x0;
      float *c_ptr1 = c_ptr0 + ldc;
      float *c_ptr2 = c_ptr1 + ldc;
      float *c_ptr3 = c_ptr2 + ldc;
      float *c_ptr4 = c_ptr3 + ldc;
      float *c_ptr5 = c_ptr4 + ldc;
      float *c_ptr6 = c_ptr5 + ldc;
      float *c_ptr7 = c_ptr6 + ldc;

      float *pout0 = c_ptr0;
      float *pout1 = c_ptr1;
      float *pout2 = c_ptr2;
      float *pout3 = c_ptr3;
      float *pout4 = c_ptr4;
      float *pout5 = c_ptr5;
      float *pout6 = c_ptr6;
      float *pout7 = c_ptr7;

      const float *a_ptr_l = A_packed + y * K;
      const float *b_ptr = b_pannel;
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
          if (has_beta) {
            for (int i = 0; i < remain; ++i) {
              cout0[i] = pout0[i];
              cout1[i] = pout1[i];
              cout2[i] = pout2[i];
              cout3[i] = pout3[i];
              cout4[i] = pout4[i];
              cout5[i] = pout5[i];
              cout6[i] = pout6[i];
              cout7[i] = pout7[i];
            }
          }
        }
        const float *a_ptr = a_ptr_l;
        int tail = tail_pre;
        int k = k_pre;
        // clang-format off
        asm volatile(
            "ldr q2, [%[bias_ptr]]\n"
            "prfm   pldl1keep, [%[a_ptr]]\n"
            "prfm   pldl1keep, [%[b_ptr]]\n"
            "dup	v8.4s,  v2.s[0]\n"
            "ldr q0, [%[a_ptr]]\n"
            "dup	v9.4s,  v2.s[0]\n"
            "ldr q4, [%[b_ptr]]\n"
            "dup	v10.4s, v2.s[0]\n"
            "ldr q3, [%[bias_ptr], #0x10]\n"
            "dup	v11.4s, v2.s[1]\n"
            "prfm   pldl1keep, [%[a_ptr], #64]\n"
            "dup	v12.4s, v2.s[1]\n"
            "prfm   pldl1keep, [%[b_ptr], #64]\n"
            "dup	v13.4s, v2.s[1]\n"
            "prfm   pldl1keep, [%[a_ptr], #128]\n"
            "dup	v14.4s, v2.s[2]\n"
            "prfm   pldl1keep, [%[b_ptr], #128]\n"
            "dup	v15.4s, v2.s[2]\n"
            "ldr q1, [%[a_ptr], #16]\n"
            "dup	v16.4s, v2.s[2]\n"
            "ldr q5, [%[b_ptr], #16]\n"
            "dup	v17.4s, v2.s[3]\n"
            "prfm   pldl1keep, [%[b_ptr], #192]\n"
            "dup	v18.4s, v2.s[3]\n"
            "prfm   pldl1keep, [%[b_ptr], #256]\n"
            "dup	v19.4s, v2.s[3]\n"
            "prfm   pldl1keep, [%[a_ptr], #192]\n"
            "dup	v20.4s, v3.s[0]\n"
            "prfm   pldl1keep, [%[b_ptr], #320]\n"
            "dup	v21.4s, v3.s[0]\n"
            "prfm   pldl1keep, [%[a_ptr], #256]\n"
            "dup	v22.4s, v3.s[0]\n"
            "prfm   pldl1keep, [%[b_ptr], #384]\n"
            "dup	v23.4s, v3.s[1]\n"
            "dup	v24.4s, v3.s[1]\n"
            "dup	v25.4s, v3.s[1]\n"
            "dup	v26.4s, v3.s[2]\n"
            "cmp %w[has_beta], #1\n"
            "dup	v27.4s, v3.s[2]\n"
            "dup	v28.4s, v3.s[2]\n"
            "dup	v29.4s, v3.s[3]\n" 
            "dup	v30.4s, v3.s[3]\n"
            "dup	v31.4s, v3.s[3]\n"
            "blt 0f\n"
            // process beta
            "dup    v7.4s, %w[beta]\n"                    // beta to vector
            "ld1    {v0.4s, v1.4s, v2.4s}, [%[c_ptr0]]\n" // load output r0
            "ld1    {v3.4s, v4.4s, v5.4s}, [%[c_ptr1]]\n" // load output r1
            "fmla   v8.4s, v0.4s, v7.4s\n"  // cr00 += beta * c_r00
            "fmla   v9.4s, v1.4s, v7.4s\n"  // cr01 += beta * c_r01
            "fmla   v10.4s, v2.4s, v7.4s\n" // cr02 += beta * c_r02
            "ld1    {v0.4s, v1.4s, v2.4s}, [%[c_ptr2]]\n" // load output r2
            "fmla   v11.4s, v3.4s, v7.4s\n" // cr10 += beta * c_r10
            "fmla   v12.4s, v4.4s, v7.4s\n" // cr11 += beta * c_r11
            "fmla   v13.4s, v5.4s, v7.4s\n" // cr12 += beta * c_r12
            "ld1    {v3.4s, v4.4s, v5.4s}, [%[c_ptr3]]\n" // load output r3
            "fmla   v14.4s, v0.4s, v7.4s\n" // cr20 += beta * c_r20
            "fmla   v15.4s, v1.4s, v7.4s\n" // cr21 += beta * c_r21
            "fmla   v16.4s, v2.4s, v7.4s\n" // cr22 += beta * c_r22
            "ld1    {v0.4s, v1.4s, v2.4s}, [%[c_ptr4]]\n" // load output r4
            "fmla   v17.4s, v3.4s, v7.4s\n" // cr30 += beta * c_r30
            "fmla   v18.4s, v4.4s, v7.4s\n" // cr31 += beta * c_r31
            "fmla   v19.4s, v5.4s, v7.4s\n" // cr32 += beta * c_r32
            "ld1    {v3.4s, v4.4s, v5.4s}, [%[c_ptr5]]\n" // load output r5
            "fmla   v20.4s, v0.4s, v7.4s\n" // cr40 += beta * c_r40
            "fmla   v21.4s, v1.4s, v7.4s\n" // cr41 += beta * c_r41
            "fmla   v22.4s, v2.4s, v7.4s\n" // cr42 += beta * c_r42
            "ld1    {v0.4s, v1.4s, v2.4s}, [%[c_ptr6]]\n" // load output r6
            "fmla   v23.4s, v3.4s, v7.4s\n" // cr50 += beta * c_r50
            "fmla   v24.4s, v4.4s, v7.4s\n" // cr51 += beta * c_r51
            "fmla   v25.4s, v5.4s, v7.4s\n" // cr52 += beta * c_r52
            "ld1    {v3.4s, v4.4s, v5.4s}, [%[c_ptr7]]\n" // load output r7
            "fmla   v26.4s, v0.4s, v7.4s\n"  // cr60 += beta * c_r60
            "fmla   v27.4s, v1.4s, v7.4s\n"  // cr61 += beta * c_r61
            "ldr q0, [%[a_ptr]]\n"
            "fmla   v28.4s, v2.4s, v7.4s\n"  // cr62 += beta * c_r62
            "ldr q1, [%[a_ptr], #16]\n"
            "fmla   v29.4s, v3.4s, v7.4s\n"  // cr70 += beta * c_r70
            "fmla   v30.4s, v4.4s, v7.4s\n"  // cr71 += beta * c_r71
            "fmla   v31.4s, v5.4s, v7.4s\n"  // cr72 += beta * c_r72
            "ldp	q4, q5, [%[b_ptr]]\n"
            "0: \n"
            "cbz	%w[k], 2f\n"               // check loop count > 0
            // main loop
            // unrool 0
            "1:\n"
            "ldr d6, [%[b_ptr], #0x20]\n"
            "nop\n"
            "fmla 	v8.4s ,  v4.4s,  v0.s[0]\n" // out0 = b0 * a00[0], b0 =q4
            "ldr x20, [%[b_ptr], #0x28]\n"
            "fmla  	v11.4s ,  v4.4s,  v0.s[1]\n" // out1 = b0 * a00[1], b0 =q4
            "ldr d2, [%[a_ptr], #0x20]\n"
            "ins v6.d[1], x20\n"
            "fmla	v14.4s,  v4.4s,  v0.s[2]\n"   // out2 = b0 * a00[2], b0 =q4
            "ldr x20, [%[a_ptr], #0x28]\n"
            "fmla	v17.4s,  v4.4s,  v0.s[3]\n"   // out3 = b0 * a00[3], b0 =q4
            "fmla 	v20.4s,  v4.4s,  v1.s[0]\n" // out4 = b0 * a01[0], b0 =q4
            "ldr d3, [%[a_ptr], #0x30]\n"
            "ins v2.d[1], x20\n"
            "fmla	v23.4s,  v4.4s,  v1.s[1]\n"   // out5 = b0 * a01[1], b0 =q4
            "ldr x20, [%[a_ptr], #0x38]\n"
            "fmla	v26.4s,  v4.4s,  v1.s[2]\n"   // out6 = b0 * a01[2], b0 =q4
            "fmla	v29.4s,  v4.4s,  v1.s[3]\n"   // out7 = b0 * a01[3], b0 =q4
            "ldr d7, [%[b_ptr], #0x30]\n"
            "ins v3.d[1], x20\n"

            "fmla	v9.4s,  v5.4s,  v0.s[0]\n"  // out8 = b1 * a00[0], b1 =q5
            "ldr x20, [%[b_ptr], #0x38]\n"
            "fmla	v12.4s,  v5.4s,  v0.s[1]\n" // out9 = b1 * a00[1], b1 =q5
            "fmla	v15.4s,  v5.4s,  v0.s[2]\n" // out10 = b1 * a00[2], b1 =q5
            "ldr d4, [%[b_ptr], #0x40]\n"
            "ins v7.d[1], x20\n"
            "fmla	v18.4s,  v5.4s,  v0.s[3]\n" // out11 = b1 * a00[3], b1 =q5
            "ldr x20, [%[b_ptr], #0x48]\n"
            "fmla	v21.4s,  v5.4s,  v1.s[0]\n" // out12 = b1 * a01[0], b1 =q5
            "fmla	v24.4s,  v5.4s,  v1.s[1]\n" // out13 = b1 * a01[1], b1 =q5
            "ins v4.d[1], x20\n"
            "fmla	v27.4s,  v5.4s,  v1.s[2]\n" // out14 = b1 * a01[2], b1 =q5
            "fmla	v30.4s,  v5.4s,  v1.s[3]\n" // out15 = b1 * a01[3], b1 =q5
            
            "ldr d5, [%[b_ptr], #0x50]\n"
            "fmla	v10.4s,  v6.4s,  v0.s[0]\n" // out16 = b2 * a00[0], b2 =q6
            "fmla	v13.4s,  v6.4s,  v0.s[1]\n" // out17 = b2 * a00[1], b2 =q6
            "ldr x20, [%[b_ptr], #0x58]\n"
            "fmla	v16.4s,  v6.4s,  v0.s[2]\n" // out18 = b2 * a00[2], b2 =q6
            "fmla	v19.4s,  v6.4s,  v0.s[3]\n" // out19 = b2 * a00[3], b2 =q6
            "nop\n"
            "ins v5.d[1], x20\n"
          
            "fmla	v22.4s,  v6.4s,  v1.s[0]\n" // out20 = b2 * a00[0], b2 =q6
            "fmla	v25.4s,  v6.4s,  v1.s[1]\n" // out21 = b2 * a00[1], b2 =q6
            "fmla	v28.4s,  v6.4s,  v1.s[2]\n" // out22 = b2 * a00[2], b2 =q6
            "nop\n"
            "fmla	v31.4s,  v6.4s,  v1.s[3]\n" // out23 = b2 * a00[3], b2 =q6

            // unrool 1
            "ldr d6, [%[b_ptr], #0x60]\n"
            "nop\n"
            "fmla 	v8.4s ,  v7.4s,  v2.s[0]\n" // out0 = b0 * a10[0], b0 =q7
            "ldr x20, [%[b_ptr], #0x68]\n"
            "fmla	v11.4s ,  v7.4s,  v2.s[1]\n"  // out1 = b0 * a10[1], b0 =q7
            "fmla	v14.4s,  v7.4s,  v2.s[2]\n"   // out2 = b0 * a10[2], b0 =q7
            "ldr d0, [%[a_ptr], #0x40]\n"
            "ins v6.d[1], x20\n"
            "fmla	v17.4s,  v7.4s,  v2.s[3]\n"   // out3 = b0 * a10[3], b0 =q7
            "ldr x20, [%[a_ptr], #0x48]\n"
            "fmla 	v20.4s,  v7.4s,  v3.s[0]\n" // out4 = b0 * a11[0], b0 =q7
            "fmla   v23.4s,  v7.4s,  v3.s[1]\n" // out5 = b0 * a11[1], b0 = q7
            "ldr d1, [%[a_ptr], #0x50]\n"
            "ins v0.d[1], x20\n"
            "fmla	v26.4s,  v7.4s,  v3.s[2]\n"   // out6 = b0 * a11[2], b0 =q7
            "ldr x20, [%[a_ptr], #0x58]\n"
            "fmla	v29.4s,  v7.4s,  v3.s[3]\n"   // out7 = b0 * a11[3], b0 =q7

            "fmla	v9.4s,  v4.4s,  v2.s[0]\n"  // out8 = b0 * a10[0], b1 =q4
            "ldr d7, [%[b_ptr], #0x70]\n"
            "ins v1.d[1], x20\n"
            "fmla	v12.4s,  v4.4s,  v2.s[1]\n" // out9 = b0 * a10[1], b1 =q4
            "ldr x20, [%[b_ptr], #0x78]\n"
            "fmla	v15.4s,  v4.4s,  v2.s[2]\n" // out10 = b1 * a10[2], b1 =q4
            "fmla	v18.4s,  v4.4s,  v2.s[3]\n" // out11 = b1 * a10[3], b1 =q4
            "nop\n"
            "ins v7.d[1], x20\n"
            "fmla	v21.4s,  v4.4s,  v3.s[0]\n" // out12 = b1 * a10[0], b1 =q4
            "fmla	v24.4s,  v4.4s,  v3.s[1]\n" // out13 = b1 * a10[1], b1 =q4
            "fmla	v27.4s,  v4.4s,  v3.s[2]\n" // out14 = b1 * a10[2], b1 =q4
            "nop\n""nop\n"
            "fmla	v30.4s,  v4.4s,  v3.s[3]\n" // out15 = b1 * a10[3], b1 =q4

            "fmla	v10.4s,  v5.4s,  v2.s[0]\n" // out16 = b2 * a10[0], b2 =q5
            "fmla	v13.4s,  v5.4s,  v2.s[1]\n" // out17 = b2 * a10[0], b2 =q5
            "ldr d4, [%[b_ptr], #0x80]\n"
            "nop\n"
            "fmla	v16.4s,  v5.4s,  v2.s[2]\n" // out18 = b2 * a10[0], b2 =q5
            "ldr x20, [%[b_ptr], #0x88]\n"
            "fmla	v19.4s,  v5.4s,  v2.s[3]\n" // out19 = b2 * a10[0], b2 =q5
            "fmla	v22.4s,  v5.4s,  v3.s[0]\n" // out20 = b2 * a10[0], b2 =q5
            "ldr d2, [%[a_ptr], #0x60]\n"
            "ins v4.d[1], x20\n"
            "fmla	v25.4s,  v5.4s,  v3.s[1]\n" // out21 = b2 * a10[0], b2 =q5
            "ldr x20, [%[a_ptr], #0x68]\n"
            "fmla	v28.4s,  v5.4s,  v3.s[2]\n" // out22 = b2 * a10[0], b2 =q5
            "fmla	v31.4s,  v5.4s,  v3.s[3]\n" // out23 = b2 * a10[0], b2 =q5
            "ldr d5, [%[b_ptr], #0x90]\n"
            "ins v2.d[1], x20\n"
            // unrool 2
            "fmla 	v8.4s ,  v6.4s,  v0.s[0]\n" // out0 = b0 * a00[0], b0 =q6
            "ldr x20, [%[b_ptr], #0x98]\n"
            "fmla  	v11.4s ,  v6.4s,  v0.s[1]\n" // out1 = b0 * a00[1], b0 =q6
            "fmla	v14.4s,  v6.4s,  v0.s[2]\n"    // out2 = b0 * a00[2], b0 =q6
            "ldr d3, [%[a_ptr], #0x70]\n"
            "ins v5.d[1], x20\n"
            "fmla	v17.4s,  v6.4s,  v0.s[3]\n"    // out3 = b0 * a00[3], b0 =q6
            "ldr x20, [%[a_ptr], #0x78]\n"
            "fmla 	v20.4s,  v6.4s,  v1.s[0]\n"  // out4 = b0 * a01[0], b0 =q6
            "fmla	v23.4s,  v6.4s,  v1.s[1]\n"    // out5 = b0 * a01[1], b0 =q6
            "prfm   pldl1keep, [%[a_ptr], #320]\n"
            "ins v3.d[1], x20\n"
            "fmla	v26.4s,  v6.4s,  v1.s[2]\n"    // out6 = b0 * a01[2], b0 =q6
            "fmla	v29.4s,  v6.4s,  v1.s[3]\n"    // out7 = b0 * a01[3], b0 =q6

            "ldr d6, [%[b_ptr], #0xa0]\n"
            "nop\n"
            "fmla	v9.4s,  v7.4s,  v0.s[0]\n"     // out8 = b1 * a00[0], b1 =q7
            "ldr x20, [%[b_ptr], #0xa8]\n"
            "fmla	v12.4s,  v7.4s,  v0.s[1]\n"    // out9 = b1 * a00[1], b1 =q7
            "fmla	v15.4s,  v7.4s,  v0.s[2]\n" // out10 = b1 * a00[2], b1 =q7
            "nop\n"
            "prfm   pldl1keep, [%[b_ptr], #448]\n"
            "ins v6.d[1], x20\n"
            "fmla	v18.4s,  v7.4s,  v0.s[3]\n" // out11 = b1 * a00[3], b1 =q7
            "fmla	v21.4s,  v7.4s,  v1.s[0]\n" // out12 = b1 * a01[0], b1 =q7
            "fmla	v24.4s,  v7.4s,  v1.s[1]\n" // out13 = b1 * a01[1], b1 =q7
            "nop\n""nop\n"
            "fmla	v27.4s,  v7.4s,  v1.s[2]\n" // out14 = b1 * a01[2], b1 =q7
            "fmla	v30.4s,  v7.4s,  v1.s[3]\n" // out15 = b1 * a01[3], b1 =q7

            "fmla	v10.4s,  v4.4s,  v0.s[0]\n" // out16 = b2 * a00[0], b2 =q4
            "ldr d7, [%[b_ptr], #0xb0]\n"
            "nop\n"
            "fmla	v13.4s,  v4.4s,  v0.s[1]\n" // out17 = b2 * a00[1], b2 =q4
            "ldr x20, [%[b_ptr], #0xb8]\n"
            "fmla	v16.4s,  v4.4s,  v0.s[2]\n" // out18 = b2 * a00[2], b2 =q4
            "fmla	v19.4s,  v4.4s,  v0.s[3]\n" // out19 = b2 * a00[3], b2 =q4
            "ldr d0,  [%[a_ptr], #0x80]\n"
            "ins v7.d[1], x20\n"
            "fmla	v22.4s,  v4.4s,  v1.s[0]\n" // out20 = b2 * a00[0], b2 =q4
            "ldr x20,  [%[a_ptr], #0x88]\n"
            "fmla	v25.4s,  v4.4s,  v1.s[1]\n" // out21 = b2 * a00[1], b2 =q4
            "fmla	v28.4s,  v4.4s,  v1.s[2]\n" // out22 = b2 * a00[2], b2 =q4
            "prfm   pldl1keep, [%[b_ptr], #512]\n"
            "ins v0.d[1], x20\n"
            "fmla	v31.4s,  v4.4s,  v1.s[3]\n" // out23 = b2 * a00[3], b2 =q4
            // unrool 3
            "ldr d4, [%[b_ptr], #0xc0]\n"
            "nop\n"
            "fmla 	v8.4s ,  v5.4s,  v2.s[0]\n" // out0 = b0 * a10[0], b0 =q5
            "ldr x20, [%[b_ptr], #0xc8]\n"
            "fmla	v11.4s ,  v5.4s,  v2.s[1]\n"  // out1 = b0 * a10[1], b0 =q5
            "fmla	v14.4s,  v5.4s,  v2.s[2]\n"   // out2 = b0 * a10[2], b0 =q5
            "ldr d1, [%[a_ptr], #0x90]\n"
            "ins v4.d[1], x20\n"
            "fmla	v17.4s,  v5.4s,  v2.s[3]\n"   // out3 = b0 * a10[3], b0 =q5
            "ldr x20, [%[a_ptr], #0x98]\n"
            "fmla 	v20.4s,  v5.4s,  v3.s[0]\n" // out4 = b0 * a11[0], b0 =q5
            "fmla   v23.4s,  v5.4s,  v3.s[1]\n" // out5 = b0 * a11[1], b0 =q5
            "nop\n"
            "ins v1.d[1], x20\n"
            "fmla	v26.4s,  v5.4s,  v3.s[2]\n"   // out6 = b0 * a11[2], b0 =q5
            "fmla	v29.4s,  v5.4s,  v3.s[3]\n"   // out7 = b0 * a11[3], b0 =q5
            
            "fmla	v9.4s,  v6.4s,  v2.s[0]\n"    // out8 = b0 * a10[0], b1 =q6
            "nop\n""nop\n"
            "fmla	v12.4s,  v6.4s,  v2.s[1]\n"   // out9 = b0 * a10[1], b1 =q6
            "fmla	v15.4s,  v6.4s,  v2.s[2]\n" // out10 = b1 * a10[2], b1 =q6
            "fmla	v18.4s,  v6.4s,  v2.s[3]\n" // out11 = b1 * a10[3], b1 =q6

            "ldr d5, [%[b_ptr], #0xd0]\n"
            "nop\n"
            "fmla	v21.4s,  v6.4s,  v3.s[0]\n" // out12 = b1 * a10[0], b1 =q6
            "ldr x20, [%[b_ptr], #0xd8]\n"
            "fmla	v24.4s,  v6.4s,  v3.s[1]\n" // out13 = b1 * a10[1], b1 =q6
            "fmla	v27.4s,  v6.4s,  v3.s[2]\n" // out14 = b1 * a10[2], b1 =q6
            "add %[a_ptr], %[a_ptr], #0x80\n"
            "fmla	v30.4s,  v6.4s,  v3.s[3]\n" // out15 = b1 * a10[3], b1 =q6
            "add %[b_ptr], %[b_ptr], #0xc0\n"
            "fmla	v10.4s,  v7.4s,  v2.s[0]\n" // out16 = b2 * a10[0], b2 =q7
            "nop\n"
            "ins v5.d[1], x20\n"
            "fmla	v13.4s,  v7.4s,  v2.s[1]\n" // out17 = b2 * a10[0], b2 =q7
            "fmla	v16.4s,  v7.4s,  v2.s[2]\n" // out18 = b2 * a10[0], b2 =q7
            "fmla	v19.4s,  v7.4s,  v2.s[3]\n" // out19 = b2 * a10[0], b2 =q7
            "nop\n"
            "fmla	v22.4s,  v7.4s,  v3.s[0]\n" // out20 = b2 * a10[0], b2 =q7
            "subs	%w[k], %w[k], #1\n"         // loop count - 1
            "fmla	v25.4s,  v7.4s,  v3.s[1]\n" // out21 = b2 * a10[0], b2 =q7
            "fmla	v28.4s,  v7.4s,  v3.s[2]\n" // out22 = b2 * a10[0], b2 =q7
            "nop\n"
            "fmla	v31.4s,  v7.4s,  v3.s[3]\n" // out23 = b2 * a10[0], b2 =q7
            "bne	1b\n"
            "2:\n"                            // process tail
            "subs		%w[tail], %w[tail], #1\n" // tail--
            "beq		3f\n"                     //jump to tail = 1
            // final unrool 0
            // unrool 0, tail > 1
            "ldr d6, [%[b_ptr], #0x20]\n"
            "fmla 	v8.4s ,  v4.4s,  v0.s[0]\n"  // out0 = b0 * a00[0], b0 =q4
            "fmla  	v11.4s ,  v4.4s,  v0.s[1]\n" // out1 = b0 * a00[1], b0 =q4
            "ldr x20, [%[b_ptr], #0x28]\n"
            "fmla	v14.4s,  v4.4s,  v0.s[2]\n"    // out2 = b0 * a00[2], b0 =q4
            "fmla	v17.4s,  v4.4s,  v0.s[3]\n"    // out3 = b0 * a00[3], b0 =q4
            "ldr d2, [%[a_ptr], #0x20]\n"
            "ins v6.d[1], x20\n"
            "fmla 	v20.4s,  v4.4s,  v1.s[0]\n"  // out4 = b0 * a01[0], b0 =q4
            "ldr x20, [%[a_ptr], #0x28]\n"
            "fmla	v23.4s,  v4.4s,  v1.s[1]\n"    // out5 = b0 * a01[1], b0 =q4
            "fmla	v26.4s,  v4.4s,  v1.s[2]\n"    // out6 = b0 * a01[2], b0 =q4
            "ldr d7, [%[b_ptr], #0x30]\n"
            "ins v2.d[1], x20\n"
            "fmla	v29.4s,  v4.4s,  v1.s[3]\n"    // out7 = b0 * a01[3], b0 =q4
            "subs	%w[tail], %w[tail], #1\n"      // tail--
            "fmla	v9.4s,  v5.4s,  v0.s[0]\n"     // out8 = b1 * a00[0], b1 =q5
            "ldr x20, [%[b_ptr], #0x38]\n"
            "fmla	v12.4s,  v5.4s,  v0.s[1]\n"    // out9 = b1 * a00[1], b1 =q5
            "fmla	v15.4s,  v5.4s,  v0.s[2]\n" // out10 = b1 * a00[2], b1 =q5
            "ldr d3, [%[a_ptr], #0x30]\n"
            "ins v7.d[1], x20\n"
            "fmla	v18.4s,  v5.4s,  v0.s[3]\n" // out11 = b1 * a00[3], b1 =q5
            "ldr x20, [%[a_ptr], #0x38]\n"
            "fmla	v21.4s,  v5.4s,  v1.s[0]\n" // out12 = b1 * a01[0], b1 =q5
            "fmla	v24.4s,  v5.4s,  v1.s[1]\n" // out13 = b1 * a01[1], b1 =q5
            "ldr d4, [%[b_ptr], #0x40]\n"
            "ins v3.d[1], x20\n"
            "fmla	v27.4s,  v5.4s,  v1.s[2]\n" // out14 = b1 * a01[2], b1 =q5
            "ldr x20, [%[b_ptr], #0x48]\n"
            "fmla	v30.4s,  v5.4s,  v1.s[3]\n" // out15 = b1 * a01[3], b1 =q5
            "fmla	v10.4s,  v6.4s,  v0.s[0]\n" // out16 = b2 * a00[0], b2 =q6
            "ldr d5, [%[b_ptr], #0x50]\n"
            "ins v4.d[1], x20\n"
            "fmla	v13.4s,  v6.4s,  v0.s[1]\n" // out17 = b2 * a00[1], b2 =q6
            "ldr x20, [%[b_ptr], #0x58]\n"
            "fmla	v16.4s,  v6.4s,  v0.s[2]\n" // out18 = b2 * a00[2], b2 =q6
            "fmla	v19.4s,  v6.4s,  v0.s[3]\n" // out19 = b2 * a00[3], b2 =q6
            "nop\n"
            "ins v5.d[1], x20\n"
            "fmla	v22.4s,  v6.4s,  v1.s[0]\n" // out20 = b2 * a00[0], b2 =q6
            "fmla	v25.4s,  v6.4s,  v1.s[1]\n" // out21 = b2 * a00[1], b2 =q6
            "fmla	v28.4s,  v6.4s,  v1.s[2]\n" // out22 = b2 * a00[2], b2 =q6
            "fmla	v31.4s,  v6.4s,  v1.s[3]\n" // out23 = b2 * a00[3], b2 =q6
            "beq		4f\n"                     //jump to tail = 2
            // unrool 1, tail > 2
            "ldr d0, [%[a_ptr], #0x40]\n"
            "nop\n"
            "fmla 	v8.4s ,  v7.4s,  v2.s[0]\n" // out0 = b0 * a10[0], b0 =q7
            "ldr x20, [%[a_ptr], #0x48]\n"
            "fmla	v11.4s ,  v7.4s,  v2.s[1]\n"  // out1 = b0 * a10[1], b0 =q7
            "fmla	v14.4s,  v7.4s,  v2.s[2]\n"   // out2 = b0 * a10[2], b0 =q7
            "ldr d1, [%[a_ptr], #0x50]\n"
            "ins v0.d[1], x20\n"
            "fmla	v17.4s,  v7.4s,  v2.s[3]\n"   // out3 = b0 * a10[3], b0 =q7
            "ldr x20, [%[a_ptr], #0x58]\n"
            "fmla 	v20.4s,  v7.4s,  v3.s[0]\n" // out4 = b0 * a11[0], b0 =q7
            "fmla   v23.4s,  v7.4s,  v3.s[1]\n" // out5 = b0 * a11[1], b0 =q7
            "ldr d6, [%[b_ptr], #0x60]\n"
            "ins v1.d[1], x20\n"
            "fmla	v26.4s,  v7.4s,  v3.s[2]\n"   // out6 = b0 * a11[2], b0 =q7
            "ldr x20, [%[b_ptr], #0x68]\n"
            "fmla	v29.4s,  v7.4s,  v3.s[3]\n"   // out7 = b0 * a11[3], b0 =q7
            "fmla	v9.4s,  v4.4s,  v2.s[0]\n"    // out8 = b0 * a10[0], b1 =q4
            "ldr d7, [%[b_ptr], #0x70]\n"
            "ins v6.d[1], x20\n"
            "fmla	v12.4s,  v4.4s,  v2.s[1]\n"   // out9 = b0 * a10[1], b1 =q4
            "ldr x20, [%[b_ptr], #0x78]\n"
            "fmla	v15.4s,  v4.4s,  v2.s[2]\n"   // out10 = b1 * a10[2], b1 =q4
            "fmla	v18.4s,  v4.4s,  v2.s[3]\n"   // out11 = b1 * a10[3], b1 =q4
            "nop\n"
            "ins v7.d[1], x20\n"
            "fmla	v21.4s,  v4.4s,  v3.s[0]\n"   // out12 = b1 * a10[0], b1 =q4
            "fmla	v24.4s,  v4.4s,  v3.s[1]\n"   // out13 = b1 * a10[1], b1 =q4
            "fmla	v27.4s,  v4.4s,  v3.s[2]\n"   // out14 = b1 * a10[2], b1 =q4
            "nop\n"
            "fmla	v30.4s,  v4.4s,  v3.s[3]\n"   // out15 = b1 * a10[3], b1 =q4
            "subs	%w[tail], %w[tail], #1\n"     // tail--
            "fmla	v10.4s,  v5.4s,  v2.s[0]\n"   // out16 = b2 * a10[0], b2 =q5
            "fmla	v13.4s,  v5.4s,  v2.s[1]\n"   // out17 = b2 * a10[0], b2 =q5
            "fmla	v16.4s,  v5.4s,  v2.s[2]\n"   // out18 = b2 * a10[0], b2 =q5
            "nop\n""nop\n"
            "fmla	v19.4s,  v5.4s,  v2.s[3]\n"   // out19 = b2 * a10[0], b2 =q5
            "fmla	v22.4s,  v5.4s,  v3.s[0]\n"   // out20 = b2 * a10[0], b2 =q5
            "fmla	v25.4s,  v5.4s,  v3.s[1]\n"   // out21 = b2 * a10[0], b2 =q5
            "nop\n"
            "fmla	v28.4s,  v5.4s,  v3.s[2]\n"   // out22 = b2 * a10[0], b2 =q5
            "fmla	v31.4s,  v5.4s,  v3.s[3]\n"   // out23 = b2 * a10[0], b2 =q5
            "beq		5f\n"                       //jump to tail = 3
            // unrool 2, tail = 4
            "ldr d4, [%[b_ptr], #0x80]\n"
            "nop\n"
            "fmla 	v8.4s ,  v6.4s,  v0.s[0]\n"  // out0 = b0 * a00[0], b0 =q6
            "ldr x20, [%[b_ptr], #0x88]\n"
            "fmla  	v11.4s ,  v6.4s,  v0.s[1]\n" // out1 = b0 * a00[1], b0 =q6
            "fmla	v14.4s,  v6.4s,  v0.s[2]\n"    // out2 = b0 * a00[2], b0 =q6
            "ldr d2, [%[a_ptr], #0x60]\n"
            "ins v4.d[1], x20\n"
            "fmla	v17.4s,  v6.4s,  v0.s[3]\n"    // out3 = b0 * a00[3], b0 =q6
            "ldr x20, [%[a_ptr], #0x68]\n"
            "fmla 	v20.4s,  v6.4s,  v1.s[0]\n"  // out4 = b0 * a01[0], b0 =q6
            "fmla	v23.4s,  v6.4s,  v1.s[1]\n"    // out5 = b0 * a01[1], b0 =q6
            "ldr d5, [%[b_ptr], #0x90]\n"
            "ins v2.d[1], x20\n"
            "fmla	v26.4s,  v6.4s,  v1.s[2]\n"    // out6 = b0 * a01[2], b0 =q6
            "ldr x20, [%[b_ptr], #0x98]\n"
            "fmla	v29.4s,  v6.4s,  v1.s[3]\n"    // out7 = b0 * a01[3], b0 =q6
            "fmla	v9.4s,  v7.4s,  v0.s[0]\n"     // out8 = b1 * a00[0], b1 =q7
            "ldr d3, [%[a_ptr], #0x70]\n"
            "ins v5.d[1], x20\n"
            "fmla	v12.4s,  v7.4s,  v0.s[1]\n"    // out9 = b1 * a00[1], b1 =q7
            "ldr x20, [%[a_ptr], #0x78]\n"
            "fmla	v15.4s,  v7.4s,  v0.s[2]\n" // out10 = b1 * a00[2], b1 =q7
            "fmla	v18.4s,  v7.4s,  v0.s[3]\n" // out11 = b1 * a00[3], b1 =q7
            "ldr d6, [%[b_ptr], #0xa0]\n"
            "ins v3.d[1], x20\n"
            "fmla	v21.4s,  v7.4s,  v1.s[0]\n" // out12 = b1 * a01[0], b1 =q7
            "ldr x20, [%[b_ptr], #0xa8]\n"
            "fmla	v24.4s,  v7.4s,  v1.s[1]\n" // out13 = b1 * a01[1], b1 =q7
            "fmla	v27.4s,  v7.4s,  v1.s[2]\n" // out14 = b1 * a01[2], b1 =q7
            "ins v6.d[1], x20\n"
            "add %[a_ptr], %[a_ptr], #0x80\n"
            "fmla	v30.4s,  v7.4s,  v1.s[3]\n" // out15 = b1 * a01[3], b1 =q7
            "ldr d7, [%[b_ptr], #0xb0]\n"
            "fmla	v10.4s,  v4.4s,  v0.s[0]\n" // out16 = b2 * a00[0], b2 =q4
            "fmla	v13.4s,  v4.4s,  v0.s[1]\n" // out17 = b2 * a00[1], b2 =q4
            "ldr x20, [%[b_ptr], #0xb8]\n"
            "fmla	v16.4s,  v4.4s,  v0.s[2]\n" // out18 = b2 * a00[2], b2 =q4
            "fmla	v19.4s,  v4.4s,  v0.s[3]\n" // out19 = b2 * a00[3], b2 =q4
            "ins v7.d[1], x20\n"
            "add %[b_ptr], %[b_ptr], #0xc0\n"
            "fmla	v22.4s,  v4.4s,  v1.s[0]\n" // out20 = b2 * a00[0], b2 =q4
            "fmla	v25.4s,  v4.4s,  v1.s[1]\n" // out21 = b2 * a00[1], b2 =q4
            "fmla	v28.4s,  v4.4s,  v1.s[2]\n" // out22 = b2 * a00[2], b2 =q4
            "fmla	v31.4s,  v4.4s,  v1.s[3]\n" // out23 = b2 * a00[3], b2 =q4
            // unrool 3, tail = 4
            "fmla 	v8.4s ,  v5.4s,  v2.s[0]\n" // out0 = b0 * a10[0], b0 =q5
            "fmla	v11.4s ,  v5.4s,  v2.s[1]\n"  // out1 = b0 * a10[1], b0 =q5
            "fmla	v14.4s,  v5.4s,  v2.s[2]\n"   // out2 = b0 * a10[2], b0 =q5
            "fmla	v17.4s,  v5.4s,  v2.s[3]\n"   // out3 = b0 * a10[3], b0 =q5
            "fmla 	v20.4s,  v5.4s,  v3.s[0]\n" // out4 = b0 * a11[0], b0 =q5
            "fmla   v23.4s,  v5.4s,  v3.s[1]\n" // out5 = b0 * a11[1], b0 =q5
            "fmla	v26.4s,  v5.4s,  v3.s[2]\n"   // out6 = b0 * a11[2], b0 =q5
            "fmla	v29.4s,  v5.4s,  v3.s[3]\n"   // out7 = b0 * a11[3], b0 =q5
            "fmla	v9.4s,  v6.4s,  v2.s[0]\n"    // out8 = b0 * a10[0], b1 =q6
            "fmla	v12.4s,  v6.4s,  v2.s[1]\n"   // out9 = b1 * a10[1], b1 =q6
            "fmla	v15.4s,  v6.4s,  v2.s[2]\n"   // out10 = b1 * a10[2], b1 =q6
            "fmla	v18.4s,  v6.4s,  v2.s[3]\n"   // out11 = b1 * a10[3], b1 =q6
            "fmla	v21.4s,  v6.4s,  v3.s[0]\n"   // out12 = b1 * a10[0], b1 =q6
            "fmla	v24.4s,  v6.4s,  v3.s[1]\n"   // out13 = b1 * a10[1], b1 =q6
            "fmla	v27.4s,  v6.4s,  v3.s[2]\n"   // out14 = b1 * a10[2], b1 =q6
            "fmla	v30.4s,  v6.4s,  v3.s[3]\n"   // out15 = b1 * a10[3], b1 =q6
            "fmla	v10.4s,  v7.4s,  v2.s[0]\n"   // out16 = b2 * a10[0], b2 =q7
            "fmla	v13.4s,  v7.4s,  v2.s[1]\n"   // out17 = b2 * a10[0], b2 =q7
            "fmla	v16.4s,  v7.4s,  v2.s[2]\n"   // out18 = b2 * a10[0], b2 =q7
            "fmla	v19.4s,  v7.4s,  v2.s[3]\n"   // out19 = b2 * a10[0], b2 =q7
            "fmla	v22.4s,  v7.4s,  v3.s[0]\n"   // out20 = b2 * a10[0], b2 =q7
            "fmla	v25.4s,  v7.4s,  v3.s[1]\n"   // out21 = b2 * a10[0], b2 =q7
            "fmla	v28.4s,  v7.4s,  v3.s[2]\n"   // out22 = b2 * a10[0], b2 =q7
            "fmla	v31.4s,  v7.4s,  v3.s[3]\n"   // out23 = b2 * a10[0], b2 =q7
            "b		11f\n"
            // tails==1 final tail
            "3: \n"                            // tail=1
            "ldr	d6, [%[b_ptr], #0x20]\n"       // load b2 to q6
            "fmla v8.4s ,  v4.4s,  v0.s[0]\n"  // out0 = b0 * a10[0], b0 =q5
            "fmla	v11.4s ,  v4.4s,  v0.s[1]\n" // out1 = b0 * a10[1], b0 =q5
            "ldr x20, [%[b_ptr], #0x28]\n"
            "fmla	v14.4s,  v4.4s,  v0.s[2]\n"  // out2 = b0 * a10[2], b0 =q5
            "fmla	v17.4s,  v4.4s,  v0.s[3]\n"  // out3 = b0 * a10[3], b0 =q5
            "add %[a_ptr], %[a_ptr], #0x20\n"
            "ins v6.d[1], x20\n"
            "fmla v20.4s,  v4.4s,  v1.s[0]\n"  // out4 = b0 * a11[0], b0 =q5
            "fmla v23.4s,  v4.4s,  v1.s[1]\n"  // out5 = b0 * a11[1], b0 = q5
            "add %[b_ptr], %[b_ptr], #0x30\n"
            "fmla	v26.4s,  v4.4s,  v1.s[2]\n"  // out6 = b0 * a11[2], b0 =q5
            "fmla	v29.4s,  v4.4s,  v1.s[3]\n"  // out7 = b0 * a11[3], b0 =q5
            "fmla	v9.4s,  v5.4s,  v0.s[0]\n"   // out8 = b0 * a10[0], b1 =q6
            "fmla	v12.4s,  v5.4s,  v0.s[1]\n"  // out9 = b1 * a10[1], b1 =q6
            "fmla	v15.4s,  v5.4s,  v0.s[2]\n"  // out10 = b1 * a10[2], b1 =q6
            "fmla	v18.4s,  v5.4s,  v0.s[3]\n"  // out11 = b1 * a10[3], b1 =q6
            "fmla	v21.4s,  v5.4s,  v1.s[0]\n"  // out12 = b1 * a10[0], b1 =q6
            "fmla	v24.4s,  v5.4s,  v1.s[1]\n"  // out13 = b1 * a10[1], b1 =q6
            "fmla	v27.4s,  v5.4s,  v1.s[2]\n"  // out14 = b1 * a10[2], b1 =q6
            "fmla	v30.4s,  v5.4s,  v1.s[3]\n"  // out15 = b1 * a10[3], b1 =q6
            "fmla	v10.4s,  v6.4s,  v0.s[0]\n"  // out16 = b2 * a10[0], b2 =q7
            "fmla	v13.4s,  v6.4s,  v0.s[1]\n"  // out17 = b2 * a10[0], b2 =q7
            "fmla	v16.4s,  v6.4s,  v0.s[2]\n"  // out18 = b2 * a10[0], b2 =q7
            "fmla	v19.4s,  v6.4s,  v0.s[3]\n"  // out19 = b2 * a10[0], b2 =q7
            "fmla	v22.4s,  v6.4s,  v1.s[0]\n"  // out20 = b2 * a10[0], b2 =q7
            "fmla	v25.4s,  v6.4s,  v1.s[1]\n"  // out21 = b2 * a10[0], b2 =q7
            "fmla	v28.4s,  v6.4s,  v1.s[2]\n"  // out22 = b2 * a10[0], b2 =q7
            "fmla	v31.4s,  v6.4s,  v1.s[3]\n"  // out23 = b2 * a10[0], b2 =q7
            "b		11f\n"
            // tails==2 final tail
            "4:\n"                              // tail = 2
            "add %[a_ptr], %[a_ptr], #0x40\n"
            "fmla 	v8.4s ,  v7.4s,  v2.s[0]\n" // out0 = b0 * a10[0], b0 =q5
            "fmla	v11.4s ,  v7.4s,  v2.s[1]\n"  // out1 = b0 * a10[1], b0 =q5
            "add %[b_ptr], %[b_ptr], #0x60\n"
            "fmla	v14.4s,  v7.4s,  v2.s[2]\n"   // out2 = b0 * a10[2], b0 =q5
            "fmla	v17.4s,  v7.4s,  v2.s[3]\n"   // out3 = b0 * a10[3], b0 =q5
            "fmla 	v20.4s,  v7.4s,  v3.s[0]\n" // out4 = b0 * a11[0], b0 =q5
            "fmla   v23.4s,  v7.4s,  v3.s[1]\n" // out5 = b0 * a11[1], b0 = q5
            "fmla	v26.4s,  v7.4s,  v3.s[2]\n"   // out6 = b0 * a11[2], b0 =q5
            "fmla	v29.4s,  v7.4s,  v3.s[3]\n"   // out7 = b0 * a11[3], b0 =q5
            "fmla	v9.4s,  v4.4s,  v2.s[0]\n"    // out8 = b0 * a10[0], b1 =q6
            "fmla	v12.4s,  v4.4s,  v2.s[1]\n"   // out9 = b1 * a10[1], b1 =q6
            "fmla	v15.4s,  v4.4s,  v2.s[2]\n"   // out10 = b1 * a10[2], b1 =q6
            "fmla	v18.4s,  v4.4s,  v2.s[3]\n"   // out11 = b1 * a10[3], b1 =q6
            "fmla	v21.4s,  v4.4s,  v3.s[0]\n"   // out12 = b1 * a10[0], b1 =q6
            "fmla	v24.4s,  v4.4s,  v3.s[1]\n"   // out13 = b1 * a10[1], b1 =q6
            "fmla	v27.4s,  v4.4s,  v3.s[2]\n"   // out14 = b1 * a10[2], b1 =q6
            "fmla	v30.4s,  v4.4s,  v3.s[3]\n"   // out15 = b1 * a10[3], b1 =q6
            "fmla	v10.4s,  v5.4s,  v2.s[0]\n"   // out16 = b2 * a10[0], b2 =q7
            "fmla	v13.4s,  v5.4s,  v2.s[1]\n"   // out17 = b2 * a10[0], b2 =q7
            "fmla	v16.4s,  v5.4s,  v2.s[2]\n"   // out18 = b2 * a10[0], b2 =q7
            "fmla	v19.4s,  v5.4s,  v2.s[3]\n"   // out19 = b2 * a10[0], b2 =q7
            "fmla	v22.4s,  v5.4s,  v3.s[0]\n"   // out20 = b2 * a10[0], b2 =q7
            "fmla	v25.4s,  v5.4s,  v3.s[1]\n"   // out21 = b2 * a10[0], b2 =q7
            "fmla	v28.4s,  v5.4s,  v3.s[2]\n"   // out22 = b2 * a10[0], b2 =q7
            "fmla	v31.4s,  v5.4s,  v3.s[3]\n"   // out23 = b2 * a10[0], b2 =q7
            "b		11f\n"
            // tails==3 final tail
            "5:\n"                              // tail = 3
            "ldr	d4, [%[b_ptr], #0x80]\n"        // load b2, b0 to q4
            "nop\n"
            "fmla 	v8.4s ,  v6.4s,  v0.s[0]\n" // out0 = b0 * a10[0], b0 =q5
            "fmla	v11.4s ,  v6.4s,  v0.s[1]\n"  // out1 = b0 * a10[1], b0 =q5
            "ldr x20, [%[b_ptr], #0x88]\n"
            "fmla	v14.4s,  v6.4s,  v0.s[2]\n"   // out2 = b0 * a10[2], b0 =q5
            "fmla	v17.4s,  v6.4s,  v0.s[3]\n"   // out3 = b0 * a10[3], b0 =q5
            "add %[a_ptr], %[a_ptr], #0x60\n"
            "ins v4.d[1], x20\n"
            "fmla 	v20.4s,  v6.4s,  v1.s[0]\n" // out4 = b0 * a11[0], b0 =q5
            "fmla   v23.4s,  v6.4s,  v1.s[1]\n" // out5 = b0 * a11[1], b0 = q5
            "add %[b_ptr], %[b_ptr], #0x90\n"
            "fmla	v26.4s,  v6.4s,  v1.s[2]\n"   // out6 = b0 * a11[2], b0 =q5
            "fmla	v29.4s,  v6.4s,  v1.s[3]\n"   // out7 = b0 * a11[3], b0 =q5
            "fmla	v9.4s,  v7.4s,  v0.s[0]\n"    // out8 = b0 * a10[0], b1 =q6
            "fmla	v12.4s,  v7.4s,  v0.s[1]\n"   // out9 = b1 * a10[1], b1 =q6
            "fmla	v15.4s,  v7.4s,  v0.s[2]\n"   // out10 = b1 * a10[2], b1 =q6
            "fmla	v18.4s,  v7.4s,  v0.s[3]\n"   // out11 = b1 * a10[3], b1 =q6
            "fmla	v21.4s,  v7.4s,  v1.s[0]\n"   // out12 = b1 * a10[0], b1 =q6
            "fmla	v24.4s,  v7.4s,  v1.s[1]\n"   // out13 = b1 * a10[1], b1 =q6
            "fmla	v27.4s,  v7.4s,  v1.s[2]\n"   // out14 = b1 * a10[2], b1 =q6
            "fmla	v30.4s,  v7.4s,  v1.s[3]\n"   // out15 = b1 * a10[3], b1 =q6
            "fmla	v10.4s,  v4.4s,  v0.s[0]\n"   // out16 = b2 * a10[0], b2 =q7
            "fmla	v13.4s,  v4.4s,  v0.s[1]\n"   // out17 = b2 * a10[0], b2 =q7
            "fmla	v16.4s,  v4.4s,  v0.s[2]\n"   // out18 = b2 * a10[0], b2 =q7
            "fmla	v19.4s,  v4.4s,  v0.s[3]\n"   // out19 = b2 * a10[0], b2 =q7
            "fmla	v22.4s,  v4.4s,  v1.s[0]\n"   // out20 = b2 * a10[0], b2 =q7
            "fmla	v25.4s,  v4.4s,  v1.s[1]\n"   // out21 = b2 * a10[0], b2 =q7
            "fmla	v28.4s,  v4.4s,  v1.s[2]\n"   // out22 = b2 * a10[0], b2 =q7
            "fmla	v31.4s,  v4.4s,  v1.s[3]\n"   // out23 = b2 * a10[0], b2 =q7

            "11:                           \n"   // check activation
            "cmp    %w[flag_act],   #1     \n"   // check if has relu
            "bne    12f                    \n"   // jump if no relu
            "movi   v0.4s,  #0             \n"   // for relu
            "fmax   v8.4s,  v8.4s,  v0.4s  \n"   // relu
            "fmax   v9.4s,  v9.4s,  v0.4s  \n"   // relu
            "fmax   v10.4s, v10.4s, v0.4s  \n"   // relu
            "fmax   v11.4s, v11.4s, v0.4s  \n"   // relu
            "fmax   v12.4s, v12.4s, v0.4s  \n"   // relu
            "fmax   v13.4s, v13.4s, v0.4s  \n"   // relu
            "fmax   v14.4s, v14.4s, v0.4s  \n"   // relu
            "fmax   v15.4s, v15.4s, v0.4s  \n"   // relu
            "fmax   v16.4s, v16.4s, v0.4s  \n"   // relu
            "fmax   v17.4s, v17.4s, v0.4s  \n"   // relu
            "fmax   v18.4s, v18.4s, v0.4s  \n"   // relu
            "fmax   v19.4s, v19.4s, v0.4s  \n"   // relu
            "fmax   v20.4s, v20.4s, v0.4s  \n"   // relu
            "fmax   v21.4s, v21.4s, v0.4s  \n"   // relu
            "fmax   v22.4s, v22.4s, v0.4s  \n"   // relu
            "fmax   v23.4s, v23.4s, v0.4s  \n"   // relu
            "fmax   v24.4s, v24.4s, v0.4s  \n"   // relu
            "fmax   v25.4s, v25.4s, v0.4s  \n"   // relu
            "fmax   v26.4s, v26.4s, v0.4s  \n"   // relu
            "fmax   v27.4s, v27.4s, v0.4s  \n"   // relu
            "fmax   v28.4s, v28.4s, v0.4s  \n"   // relu
            "fmax   v29.4s, v29.4s, v0.4s  \n"   // relu
            "fmax   v30.4s, v30.4s, v0.4s  \n"   // relu
            "fmax   v31.4s, v31.4s, v0.4s  \n"   // relu
            "b      20f                    \n"   // relu end
            //! no act
            "12:                           \n"   // no relu
            "cmp   %w[flag_act],  #0       \n"   // check no act
            "beq   20f                     \n"   // no act end
            //! relu6
            "cmp    %w[flag_act],  #2      \n"    // check if has relu6
            "bne    13f                    \n"    // jump if no relu6
            "movi   v0.4s, #0              \n"    // for relu6
            "ld1    {v1.4s}, [%[alpha]]    \n"    // relu6 alpha
            "fmax   v8.4s,  v8.4s,  v0.4s  \n"    // relu6
            "fmax   v9.4s,  v9.4s,  v0.4s  \n"    // relu6
            "fmax   v10.4s, v10.4s, v0.4s  \n"    // relu6
            "fmax   v11.4s, v11.4s, v0.4s  \n"    // relu6
            "fmax   v12.4s, v12.4s, v0.4s  \n"    // relu6
            "fmax   v13.4s, v13.4s, v0.4s  \n"    // relu6
            "fmax   v14.4s, v14.4s, v0.4s  \n"    // relu6
            "fmax   v15.4s, v15.4s, v0.4s  \n"    // relu6
            "fmax   v16.4s, v16.4s, v0.4s  \n"    // relu6
            "fmax   v17.4s, v17.4s, v0.4s  \n"    // relu6
            "fmax   v18.4s, v18.4s, v0.4s  \n"    // relu6
            "fmax   v19.4s, v19.4s, v0.4s  \n"    // relu6
            "fmax   v20.4s, v20.4s, v0.4s  \n"    // relu6
            "fmax   v21.4s, v21.4s, v0.4s  \n"    // relu6
            "fmax   v22.4s, v22.4s, v0.4s  \n"    // relu6
            "fmax   v23.4s, v23.4s, v0.4s  \n"    // relu6
            "fmax   v24.4s, v24.4s, v0.4s  \n"    // relu6
            "fmax   v25.4s, v25.4s, v0.4s  \n"    // relu6
            "fmax   v26.4s, v26.4s, v0.4s  \n"    // relu6
            "fmax   v27.4s, v27.4s, v0.4s  \n"    // relu6
            "fmax   v28.4s, v28.4s, v0.4s  \n"    // relu6
            "fmax   v29.4s, v29.4s, v0.4s  \n"    // relu6
            "fmax   v30.4s, v30.4s, v0.4s  \n"    // relu6
            "fmax   v31.4s, v31.4s, v0.4s  \n"    // relu6
            "fmin   v8.4s,  v8.4s,  v1.4s  \n"    // relu6
            "fmin   v9.4s,  v9.4s,  v1.4s  \n"    // relu6
            "fmin   v10.4s, v10.4s, v1.4s  \n"    // relu6
            "fmin   v11.4s, v11.4s, v1.4s  \n"    // relu6
            "fmin   v12.4s, v12.4s, v1.4s  \n"    // relu6
            "fmin   v13.4s, v13.4s, v1.4s  \n"    // relu6
            "fmin   v14.4s, v14.4s, v1.4s  \n"    // relu6
            "fmin   v15.4s, v15.4s, v1.4s  \n"    // relu6
            "fmin   v16.4s, v16.4s, v1.4s  \n"    // relu6
            "fmin   v17.4s, v17.4s, v1.4s  \n"    // relu6
            "fmin   v18.4s, v18.4s, v1.4s  \n"    // relu6
            "fmin   v19.4s, v19.4s, v1.4s  \n"    // relu6
            "fmin   v20.4s, v20.4s, v1.4s  \n"    // relu6
            "fmin   v21.4s, v21.4s, v1.4s  \n"    // relu6
            "fmin   v22.4s, v22.4s, v1.4s  \n"    // relu6
            "fmin   v23.4s, v23.4s, v1.4s  \n"    // relu6
            "fmin   v24.4s, v24.4s, v1.4s  \n"    // relu6
            "fmin   v25.4s, v25.4s, v1.4s  \n"    // relu6
            "fmin   v26.4s, v26.4s, v1.4s  \n"    // relu6
            "fmin   v27.4s, v27.4s, v1.4s  \n"    // relu6
            "fmin   v28.4s, v28.4s, v1.4s  \n"    // relu6
            "fmin   v29.4s, v29.4s, v1.4s  \n"    // relu6
            "fmin   v30.4s, v30.4s, v1.4s  \n"    // relu6
            "fmin   v31.4s, v31.4s, v1.4s  \n"    // relu6
            "b      20f                    \n"    // relu6 end
            //! leakey relu
            "13:                                \n" // otherwise is leakey relu
            "movi   v0.4s,    #0                \n" // for leakey relu
            "ld1    {v1.4s},  [%[alpha]]        \n" // leakey relu alpha
            "fcmge  v2.4s,    v8.4s,    v0.4s   \n" // vcgeq_f32
            "fmul   v3.4s,    v8.4s,    v1.4s   \n" // vmulq_f32
            "fcmge  v4.4s,    v9.4s,    v0.4s   \n" // vcgeq_f32
            "fmul   v5.4s,    v9.4s,    v1.4s   \n" // vmulq_f32
            "fcmge  v6.4s,    v10.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v7.4s,    v10.4s,   v1.4s   \n" // vmulq_f32
            "bif    v8.16b,   v3.16b,   v2.16b  \n" // choose
            "bif    v9.16b,   v5.16b,   v4.16b  \n" // choose
            "bif    v10.16b,  v7.16b,   v6.16b  \n" // choose
            "fcmge  v2.4s,    v11.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v3.4s,    v11.4s,   v1.4s   \n" // vmulq_f32
            "bif    v11.16b,  v3.16b,   v2.16b  \n" // choose
            "fcmge  v2.4s,    v12.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v3.4s,    v12.4s,   v1.4s   \n" // vmulq_f32
            "fcmge  v4.4s,    v13.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v5.4s,    v13.4s,   v1.4s   \n" // vmulq_f32
            "fcmge  v6.4s,    v14.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v7.4s,    v14.4s,   v1.4s   \n" // vmulq_f32
            "bif    v12.16b,  v3.16b,   v2.16b  \n" // choose
            "bif    v13.16b,  v5.16b,   v4.16b  \n" // choose
            "bif    v14.16b,  v7.16b,   v6.16b  \n" // choose
            "fcmge  v2.4s,    v15.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v3.4s,    v15.4s,   v1.4s   \n" // vmulq_f32
            "bif    v15.16b,  v3.16b,   v2.16b  \n" // choose
            "fcmge  v2.4s,    v16.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v3.4s,    v16.4s,   v1.4s   \n" // vmulq_f32
            "fcmge  v4.4s,    v17.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v5.4s,    v17.4s,   v1.4s   \n" // vmulq_f32
            "fcmge  v6.4s,    v18.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v7.4s,    v18.4s,   v1.4s   \n" // vmulq_f32
            "bif    v16.16b,  v3.16b,   v2.16b  \n" // choose
            "bif    v17.16b,  v5.16b,   v4.16b  \n" // choose
            "bif    v18.16b,  v7.16b,   v6.16b  \n" // choose
            "fcmge  v2.4s,    v19.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v3.4s,    v19.4s,   v1.4s   \n" // vmulq_f32
            "bif    v19.16b,  v3.16b,   v2.16b  \n" // choose
            "fcmge  v2.4s,    v20.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v3.4s,    v20.4s,   v1.4s   \n" // vmulq_f32
            "fcmge  v4.4s,    v21.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v5.4s,    v21.4s,   v1.4s   \n" // vmulq_f32
            "fcmge  v6.4s,    v22.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v7.4s,    v22.4s,   v1.4s   \n" // vmulq_f32
            "bif    v20.16b,  v3.16b,   v2.16b  \n" // choose
            "bif    v21.16b,  v5.16b,   v4.16b  \n" // choose
            "bif    v22.16b,  v7.16b,   v6.16b  \n" // choose
            "fcmge  v2.4s,    v23.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v3.4s,    v23.4s,   v1.4s   \n" // vmulq_f32
            "bif    v23.16b,  v3.16b,   v2.16b  \n" // choose
            "fcmge  v2.4s,    v24.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v3.4s,    v24.4s,   v1.4s   \n" // vmulq_f32
            "fcmge  v4.4s,    v25.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v5.4s,    v25.4s,   v1.4s   \n" // vmulq_f32
            "fcmge  v6.4s,    v26.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v7.4s,    v26.4s,   v1.4s   \n" // vmulq_f32
            "bif    v24.16b,  v3.16b,   v2.16b  \n" // choose
            "bif    v25.16b,  v5.16b,   v4.16b  \n" // choose
            "bif    v26.16b,  v7.16b,   v6.16b  \n" // choose
            "fcmge  v2.4s,    v27.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v3.4s,    v27.4s,   v1.4s   \n" // vmulq_f32
            "bif    v27.16b,  v3.16b,   v2.16b  \n" // choose
            "fcmge  v2.4s,    v28.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v3.4s,    v28.4s,   v1.4s   \n" // vmulq_f32
            "fcmge  v4.4s,    v29.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v5.4s,    v29.4s,   v1.4s   \n" // vmulq_f32
            "fcmge  v6.4s,    v30.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v7.4s,    v30.4s,   v1.4s   \n" // vmulq_f32
            "bif    v28.16b,  v3.16b,   v2.16b  \n" // choose
            "bif    v29.16b,  v5.16b,   v4.16b  \n" // choose
            "bif    v30.16b,  v7.16b,   v6.16b  \n" // choose
            "fcmge  v2.4s,    v31.4s,   v0.4s   \n" // vcgeq_f32
            "fmul   v3.4s,    v31.4s,   v1.4s   \n" // vmulq_f32
            "bif    v31.16b,  v3.16b,   v2.16b  \n" // choose
            "20:                                \n" // act end

            "st1 {v8.4s, v9.4s, v10.4s},[%[c_ptr0]], #48\n"   // store r0
            "st1 {v11.4s, v12.4s, v13.4s},[%[c_ptr1]], #48\n" // store r1
            "st1 {v14.4s, v15.4s, v16.4s},[%[c_ptr2]], #48\n" // store r2
            "st1 {v17.4s, v18.4s, v19.4s},[%[c_ptr3]], #48\n" // store r3
            "st1 {v20.4s, v21.4s, v22.4s},[%[c_ptr4]], #48\n" // store r4
            "st1 {v23.4s, v24.4s, v25.4s},[%[c_ptr5]], #48\n" // store r5
            "st1 {v26.4s, v27.4s, v28.4s},[%[c_ptr6]], #48\n" // store r6
            "st1 {v29.4s, v30.4s, v31.4s},[%[c_ptr7]], #48\n" // store r7

            : [a_ptr] "+r"(a_ptr),
              [b_ptr] "+r"(b_ptr),
              [k] "+r"(k),
              [tail] "+r"(tail),
              [c_ptr0] "+r"(c_ptr0),
              [c_ptr1] "+r"(c_ptr1),
              [c_ptr2] "+r"(c_ptr2),
              [c_ptr3] "+r"(c_ptr3),
              [c_ptr4] "+r"(c_ptr4),
              [c_ptr5] "+r"(c_ptr5),
              [c_ptr6] "+r"(c_ptr6),
              [c_ptr7] "+r"(c_ptr7)
            : [bias_ptr] "r"(bias_local),
              [has_beta] "r"(has_beta),
              [beta] "r"(beta),
              [alpha] "r"(alpha),
              [flag_act] "r"(flag_act)
            : "cc","memory", "x20",
              "v0","v1","v2","v3","v4","v5","v6","v7",
              "v8","v9","v10","v11","v12","v13",
              "v14","v15","v16","v17","v18","v19",
              "v20","v21","v22","v23","v24","v25",
              "v26","v27","v28","v29","v30","v31");
        // clang-format on
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

void sgemm_prepacked_4x8(bool is_transB,
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
  size_t l2_cache = ctx->llc_size() > 0 ? ctx->llc_size() : 512 * 1024;
  auto *workspace = ctx->workspace_data<float>();
  int threads = ctx->threads();
  auto act_type = act_param.active_type;
  float alpha[4] = {0.f, 0.f, 0.f, 0.f};
  int flag_act = 0x00;  // relu: 1, relu6: 2, leaky: 4
  const int n_block = 8;
  const int m_block = 4;
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    }
  }
  float32x4_t valpha = vld1q_f32(alpha);
  float32x4_t vzero = vdupq_n_f32(0.f);
  //! MBLOCK * x (result) + MBLOCK * k (A) + x * k (B) = l2
  int x_block = (l2_cache - (m_block * K)) / (sizeof(float) * (K + m_block));
  x_block /= n_block;
  x_block *= n_block;
  int x_num = (N + (x_block - 1)) / x_block;
  x_block = (N + x_num - 1) / x_num;
  x_block = (x_block + n_block - 1) / n_block;
  x_block *= n_block;
  x_block = x_block < n_block ? n_block : x_block;

  int k_pre = ((K + KBLOCK - 1) / KBLOCK) - 1;
  int tail_pre = (K & (KBLOCK - 1));
  if (tail_pre == 0) {
    tail_pre = KBLOCK;
  }
  bool flag_p_remain = false;
  int remain = 0;

  int has_beta = fabsf(beta) > 1e-8f ? 1 : 0;
  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    if (xmax > N) {
      xmax = N;
    }
    int bblocks = (xmax - x0 + n_block - 1) / n_block;
    remain = xmax - x0 - (bblocks - 1) * n_block;
    if (remain > 0) {
      flag_p_remain = true;
    }
    //! load bpanel
    auto b_pannel = static_cast<float *>(workspace);
    if (is_transB) {
      loadb_trans_eight(b_pannel, B, ldb, 0, K, x0, xmax);
    } else {
      loadb_eight(b_pannel, B, ldb, 0, K, x0, xmax);
    }
#pragma omp parallel for num_threads(threads)
    for (unsigned int y = 0; y < M; y += m_block) {
      unsigned int ymax = y + m_block;
      if (ymax > M) {
        ymax = M;
      }
      float cout0[n_block];  // NOLINT
      float cout1[n_block];  // NOLINT
      float cout2[n_block];  // NOLINT;
      float cout3[n_block];  // NOLINT

      float bias_local[4] = {0};
      if (has_bias) {
        bias_local[0] = bias[y];
        bias_local[1] = bias[y + 1];
        bias_local[2] = bias[y + 2];
        bias_local[3] = bias[y + 3];
      }

      float *c_ptr0 = C + y * ldc + x0;
      float *c_ptr1 = c_ptr0 + ldc;
      float *c_ptr2 = c_ptr1 + ldc;
      float *c_ptr3 = c_ptr2 + ldc;

      float *pout0 = c_ptr0;
      float *pout1 = c_ptr1;
      float *pout2 = c_ptr2;
      float *pout3 = c_ptr3;

      const float *a_ptr_l = A_packed + y * K;
      const float *b_ptr = b_pannel;
      for (int xb = 0; xb < bblocks; xb++) {
        if ((y + 3) >= ymax) {
          switch ((y + 3) - ymax) {
            case 2:
              c_ptr1 = cout1;
            case 1:
              c_ptr2 = cout1;
            case 0:
              c_ptr3 = cout1;
            default:
              break;
          }
        }
        if (flag_p_remain && (xb == bblocks - 1)) {
          pout0 = c_ptr0;
          pout1 = c_ptr1;
          pout2 = c_ptr2;
          pout3 = c_ptr3;

          c_ptr0 = cout0;
          c_ptr1 = cout1;
          c_ptr2 = cout2;
          c_ptr3 = cout3;

          if (has_beta) {
            for (int i = 0; i < remain; ++i) {
              cout0[i] = pout0[i];
              cout1[i] = pout1[i];
              cout2[i] = pout2[i];
              cout3[i] = pout3[i];
            }
          }
        }
        const float *a_ptr = a_ptr_l;
        int tails = tail_pre;
        int k = k_pre;
        // clang-format off
        asm volatile(
            "ld1 {v2.4s}, [%[bias_ptr]]\n"
            "dup v8.4s, v2.s[0]\n"
            "prfm   pldl1keep, [%[a_ptr]]\n"
            "dup v9.4s, v2.s[0]\n"
            "prfm   pldl1keep, [%[b_ptr]]\n"
            "dup v10.4s, v2.s[1]\n"
            "dup v11.4s, v2.s[1]\n"
            "prfm   pldl1keep, [%[a_ptr], #64]\n"
            "dup v12.4s, v2.s[2]\n"
            "dup v13.4s, v2.s[2]\n"
            "prfm   pldl1keep, [%[b_ptr], #64]\n"
            "dup v14.4s, v2.s[3]\n"
            "dup v15.4s, v2.s[3]\n"
            "prfm   pldl1keep, [%[a_ptr], #128]\n"
            "cmp    %w[beta], #0\n"                // check beta == 0?
            "prfm   pldl1keep, [%[b_ptr], #128]\n"
            "prfm   pldl1keep, [%[b_ptr], #192]\n"
            // process beta
            "beq    11f\n"
            "dup    v7.4s, %w[beta]\n"              // beta to vector
            "ld1 {v0.4s, v1.4s}, [%[c_ptr0]]\n"     // load output r0
            "ld1 {v2.4s, v3.4s}, [%[c_ptr1]]\n"     // load output r1
            "fmla v8.4s, v0.4s, v7.4s\n"            // cr00 += beta * c_r00
            "fmla v9.4s, v1.4s, v7.4s\n"            // cr01 += beta * c_r01
            "ld1 {v0.4s, v1.4s}, [%[c_ptr2]]\n"     // load output r2
            "fmla v10.4s, v2.4s, v7.4s\n"           // cr10 += beta * c_r10
            "fmla v11.4s, v3.4s, v7.4s\n"           // cr11 += beta * c_r11
            "ld1 {v2.4s, v3.4s}, [%[c_ptr3]]\n"     // load output r3
            "fmla v12.4s, v0.4s, v7.4s\n"           // cr20 += beta * c_r20
            "fmla v13.4s, v1.4s, v7.4s\n"           // cr21 += beta * c_r21
            "fmla v14.4s, v2.4s, v7.4s\n"           // cr30 += beta * c_r30
            "fmla v15.4s, v3.4s, v7.4s\n"           // cr31 += beta * c_r31
            "11: \n"                                // check loop count
            "ldp  q0, q1, [%[a_ptr]], #32\n"        // load a0~a3 to q0, q1
            "ldp  q4, q5, [%[b_ptr]], #32\n"        // load b0~b3 to q4, q5
            "cbz        %w[k], 0f\n"                // check loop count > 0
            // main loop
            // Unroll 0
            "1: \n"
            "fmla v8.4s, v4.4s, v0.s[0]\n"          // out0 += b0 * a0[0]
            "fmla v10.4s, v4.4s, v0.s[1]\n"         // out1 += b0 * a0[1]
            "ldp  q6, q7, [%[b_ptr]], #32\n"        // load next b1, b2
            "fmla v12.4s, v4.4s, v0.s[2]\n"         // out2 += b0 * a0[2]
            "fmla v14.4s, v4.4s, v0.s[3]\n"         // out3 += b0 * a0[3]
            "ldp  q2, q3, [%[a_ptr]], #32\n"        // load next 2xa0~a3
            "fmla v9.4s, v5.4s, v0.s[0]\n"          // out4 += b1 * a0[0]
            "fmla v11.4s, v5.4s, v0.s[1]\n"         // out5 += b1 * a0[1]
            "fmla v13.4s, v5.4s, v0.s[2]\n"         // out6 += b1 * a0[2]
            "fmla v15.4s, v5.4s, v0.s[3]\n"         // out7 += b1 * a0[3]
            "ldp  q4, q5, [%[b_ptr]], #32\n"        // load b0~b3 to q4, q5
            // Unroll 1
            "fmla v8.4s, v6.4s, v1.s[0]\n"          // out0 += b0 * a0[0]
            "prfm   pldl1keep, [%[b_ptr], #64]\n"
            "fmla v10.4s, v6.4s, v1.s[1]\n"         // out1 += b0 * a0[1]
            "fmla v12.4s, v6.4s, v1.s[2]\n"         // out1 += b0 * a0[2]
            "fmla v14.4s, v6.4s, v1.s[3]\n"         // out1 += b0 * a0[3]
            "fmla v9.4s, v7.4s, v1.s[0]\n"          // out4 += b1 * a0[0]
            "fmla v11.4s, v7.4s, v1.s[1]\n"         // out5 += b1 * a0[1]
            "fmla v13.4s, v7.4s, v1.s[2]\n"         // out6 += b1 * a0[2]
            "fmla v15.4s, v7.4s, v1.s[3]\n"         // out7 += b1 * a0[3]
            "ldp  q6, q7, [%[b_ptr]], #32\n"        // load next b1, b2
            // Unroll 2
            "fmla v8.4s, v4.4s, v2.s[0]\n"          // out0 += b0 * a0[0]
            "ldp  q0, q1, [%[a_ptr]], #32\n"        // load a0~a3 to q0, q1
            "fmla v10.4s, v4.4s, v2.s[1]\n"         // out1 += b0 * a0[1]
            "fmla v12.4s, v4.4s, v2.s[2]\n"         // out1 += b0 * a0[2]
            "fmla v14.4s, v4.4s, v2.s[3]\n"         // out1 += b0 * a0[3]
            "fmla v9.4s, v5.4s, v2.s[0]\n"          // out4 += b1 * a0[0]
            "fmla v11.4s, v5.4s, v2.s[1]\n"         // out5 += b1 * a0[1]
            "fmla v13.4s, v5.4s, v2.s[2]\n"         // out6 += b1 * a0[2]
            "fmla v15.4s, v5.4s, v2.s[3]\n"         // out7 += b1 * a0[3]
            "ldp  q4, q5, [%[b_ptr]], #32\n"        // load b0~b3 to q4, q5
            // Unroll 3
            "fmla v8.4s, v6.4s, v3.s[0]\n"          // out0 += b0 * a0[0]
            "prfm   pldl1keep, [%[a_ptr], #64]\n"
            "fmla v10.4s, v6.4s, v3.s[1]\n"         // out1 += b0 * a0[1]
            "fmla v12.4s, v6.4s, v3.s[2]\n"         // out1 += b0 * a0[2]
            "fmla v14.4s, v6.4s, v3.s[3]\n"         // out1 += b0 * a0[3]
            "subs       %w[k], %w[k], #1\n"         // loop count - 1
            "fmla v9.4s, v7.4s, v3.s[0]\n"          // out4 += b1 * a0[0]
            "fmla v11.4s, v7.4s, v3.s[1]\n"         // out5 += b1 * a0[1]
            "fmla v13.4s, v7.4s, v3.s[2]\n"         // out6 += b1 * a0[2]
            "fmla v15.4s, v7.4s, v3.s[3]\n"         // out7 += b1 * a0[3]
            "bne        1b\n"
            "0: \n"
            "subs %w[tail], %w[tail], #1\n"         // tail--
            "beq                3f\n"               // jump to tail = 1
            // Unroll 0
            "ldp  q6, q7, [%[b_ptr]], #32\n"        // load next b1, b2
            "fmla v8.4s, v4.4s, v0.s[0]\n"          // out0 += b0 * a0[0]
            "fmla v10.4s, v4.4s, v0.s[1]\n"         // out1 += b0 * a0[1]
            "subs %w[tail], %w[tail], #1\n"         // tail--
            "fmla v12.4s, v4.4s, v0.s[2]\n"         // out2 += b0 * a0[2]
            "fmla v14.4s, v4.4s, v0.s[3]\n"         // out3 += b0 * a0[3]
            "fmla v9.4s, v5.4s, v0.s[0]\n"          // out4 += b1 * a0[0]
            "fmla v11.4s, v5.4s, v0.s[1]\n"         // out5 += b1 * a0[1]
            "fmla v13.4s, v5.4s, v0.s[2]\n"         // out6 += b1 * a0[2]
            "fmla v15.4s, v5.4s, v0.s[3]\n"         // out7 += b1 * a0[3]
            "beq  4f\n"                             // jump to tail = 2
            // Unroll 1
            "ldp  q4, q5, [%[b_ptr]], #32\n"        // load b0~b3 to q4, q5
            "fmla v8.4s, v6.4s, v1.s[0]\n"          // out0 += b0 * a0[0]
            "ldp  q2, q3, [%[a_ptr]], #32\n"        // load next 2xa0~a3
            "fmla v10.4s, v6.4s, v1.s[1]\n"         // out1 += b0 * a0[1]
            "subs %w[tail], %w[tail], #1\n"         // tail--
            "fmla v12.4s, v6.4s, v1.s[2]\n"         // out1 += b0 * a0[2]
            "fmla v14.4s, v6.4s, v1.s[3]\n"         // out1 += b0 * a0[3]
            "fmla v9.4s, v7.4s, v1.s[0]\n"          // out4 += b1 * a0[0]
            "fmla v11.4s, v7.4s, v1.s[1]\n"         // out5 += b1 * a0[1]
            "fmla v13.4s, v7.4s, v1.s[2]\n"         // out6 += b1 * a0[2]
            "fmla v15.4s, v7.4s, v1.s[3]\n"         // out7 += b1 * a0[3]
            "beq  5f\n"                             // jump to tail = 3
            // Unroll 2
            "ldp  q6, q7, [%[b_ptr]], #32\n"        // load next b1, b2
            "fmla v8.4s, v4.4s, v2.s[0]\n"          // out0 += b0 * a0[0]
            "fmla v10.4s, v4.4s, v2.s[1]\n"         // out1 += b0 * a0[1]
            "fmla v12.4s, v4.4s, v2.s[2]\n"         // out2 += b0 * a0[2]
            "fmla v14.4s, v4.4s, v2.s[3]\n"         // out3 += b0 * a0[3]
            "fmla v9.4s, v5.4s, v2.s[0]\n"          // out4 += b1 * a0[0]
            "fmla v11.4s, v5.4s, v2.s[1]\n"         // out5 += b1 * a0[1]
            "fmla v13.4s, v5.4s, v2.s[2]\n"         // out6 += b1 * a0[2]
            "fmla v15.4s, v5.4s, v2.s[3]\n"         // out7 += b1 * a0[3]
            // Unroll 3
            "fmla v8.4s, v6.4s, v3.s[0]\n"          // out0 += b0 * a0[0]
            "fmla v10.4s, v6.4s, v3.s[1]\n"         // out1 += b0 * a0[1]
            "fmla v12.4s, v6.4s, v3.s[2]\n"         // out1 += b0 * a0[2]
            "fmla v14.4s, v6.4s, v3.s[3]\n"         // out1 += b0 * a0[3]
            "fmla v9.4s, v7.4s, v3.s[0]\n"          // out4 += b1 * a0[0]
            "fmla v11.4s, v7.4s, v3.s[1]\n"         // out5 += b1 * a0[1]
            "fmla v13.4s, v7.4s, v3.s[2]\n"         // out6 += b1 * a0[2]
            "fmla v15.4s, v7.4s, v3.s[3]\n"         // out7 += b1 * a0[3]
            "b		2f\n"
            // tails==1 final tail
            "3: \n"
            "fmla v8.4s, v4.4s, v0.s[0]\n"          // out0 += b0 * a0[0]
            "fmla v10.4s, v4.4s, v0.s[1]\n"         // out1 += b0 * a0[1]
            "fmla v12.4s, v4.4s, v0.s[2]\n"         // out2 += b0 * a0[2]
            "fmla v14.4s, v4.4s, v0.s[3]\n"         // out3 += b0 * a0[3]
            "fmla v9.4s, v5.4s, v0.s[0]\n"          // out4 += b1 * a0[0]
            "fmla v11.4s, v5.4s, v0.s[1]\n"         // out5 += b1 * a0[1]
            "fmla v13.4s, v5.4s, v0.s[2]\n"         // out6 += b1 * a0[2]
            "fmla v15.4s, v5.4s, v0.s[3]\n"         // out7 += b1 * a0[3]
            // aptr - 16
            "sub %w[a_ptr], %w[a_ptr], #16\n"
            "b		2f\n"
            "4: \n"
            // tails==2 final tail
            "fmla v8.4s, v6.4s, v1.s[0]\n"          // out0 += b0 * a0[0]
            "fmla v10.4s, v6.4s, v1.s[1]\n"         // out1 += b0 * a0[1]
            "fmla v12.4s, v6.4s, v1.s[2]\n"         // out1 += b0 * a0[2]
            "fmla v14.4s, v6.4s, v1.s[3]\n"         // out1 += b0 * a0[3]
            "fmla v9.4s, v7.4s, v1.s[0]\n"          // out4 += b1 * a0[0]
            "fmla v11.4s, v7.4s, v1.s[1]\n"         // out5 += b1 * a0[1]
            "fmla v13.4s, v7.4s, v1.s[2]\n"         // out6 += b1 * a0[2]
            "fmla v15.4s, v7.4s, v1.s[3]\n"         // out7 += b1 * a0[3]
            "b		2f\n"
            // tails==3 final tail
            "5: \n"
            "fmla v8.4s, v4.4s, v2.s[0]\n"          // out0 += b0 * a0[0]
            "fmla v10.4s, v4.4s, v2.s[1]\n"         // out1 += b0 * a0[1]
            "fmla v12.4s, v4.4s, v2.s[2]\n"         // out2 += b0 * a0[2]
            "fmla v14.4s, v4.4s, v2.s[3]\n"         // out3 += b0 * a0[3]
            "fmla v9.4s, v5.4s, v2.s[0]\n"          // out4 += b1 * a0[0]
            "fmla v11.4s, v5.4s, v2.s[1]\n"         // out5 += b1 * a0[1]
            "fmla v13.4s, v5.4s, v2.s[2]\n"         // out6 += b1 * a0[2]
            "fmla v15.4s, v5.4s, v2.s[3]\n"         // out7 += b1 * a0[3]
            // aptr - 16
            "sub %w[a_ptr], %w[a_ptr], #16\n"
            "2: \n"
            "cmp %w[flag_act], #0\n"                // check if has act
            "beq 10f\n"
            "cmp %w[flag_act], #1\n"                // check if has relu
            "bne 6f\n"
            "fmax   v8.4s,  v8.4s,  %[vzero].4s\n"
            "fmax   v9.4s,  v9.4s,  %[vzero].4s\n"
            "fmax   v10.4s, v10.4s, %[vzero].4s\n"
            "fmax   v11.4s, v11.4s, %[vzero].4s\n"
            "fmax   v12.4s, v12.4s, %[vzero].4s\n"
            "fmax   v13.4s, v13.4s, %[vzero].4s\n"
            "fmax   v14.4s, v14.4s, %[vzero].4s\n"
            "fmax   v15.4s, v15.4s, %[vzero].4s\n"
            "b 10f\n"                               // end
            "6: \n"
            "cmp  %w[flag_act],  #2\n"             // check relu6
            "bne 7f\n"
            "fmax   v8.4s,  v8.4s,  %[vzero].4s\n"
            "fmax   v9.4s,  v9.4s,  %[vzero].4s\n"
            "fmax   v10.4s, v10.4s, %[vzero].4s\n"
            "fmax   v11.4s, v11.4s, %[vzero].4s\n"
            "fmax   v12.4s, v12.4s, %[vzero].4s\n"
            "fmax   v13.4s, v13.4s, %[vzero].4s\n"
            "fmax   v14.4s, v14.4s, %[vzero].4s\n"
            "fmax   v15.4s, v15.4s, %[vzero].4s\n"
            "fmin   v8.4s,  v8.4s,  %[valpha].4s\n"
            "fmin   v9.4s,  v9.4s,  %[valpha].4s\n"
            "fmin   v10.4s, v10.4s, %[valpha].4s\n"
            "fmin   v11.4s, v11.4s, %[valpha].4s\n"
            "fmin   v12.4s, v12.4s, %[valpha].4s\n"
            "fmin   v13.4s, v13.4s, %[valpha].4s\n"
            "fmin   v14.4s, v14.4s, %[valpha].4s\n"
            "fmin   v15.4s, v15.4s, %[valpha].4s\n"
            "b 10f\n"
            "7: \n"
            "fcmge  v2.4s,    v8.4s, %[vzero].4s\n"
            "fmul   v3.4s,    v8.4s, %[valpha].4s\n"
            "fcmge  v4.4s,    v9.4s, %[vzero].4s\n"
            "fmul   v5.4s,    v9.4s, %[valpha].4s\n"
            "fcmge  v6.4s,    v10.4s, %[vzero].4s\n"
            "fmul   v7.4s,    v10.4s, %[valpha].4s\n"
            "fcmge  v0.4s,    v11.4s, %[vzero].4s\n"
            "fmul   v1.4s,    v11.4s, %[valpha].4s\n"
            "bif    v8.16b,   v3.16b,   v2.16b  \n"
            "bif    v9.16b,   v5.16b,   v4.16b  \n"
            "bif    v10.16b,  v7.16b,   v6.16b  \n"
            "bif    v11.16b,  v1.16b,   v0.16b  \n"
            "fcmge  v2.4s,    v12.4s, %[vzero].4s\n"
            "fmul   v3.4s,    v12.4s, %[valpha].4s\n"
            "fcmge  v4.4s,    v13.4s, %[vzero].4s\n"
            "fmul   v5.4s,    v13.4s,   v1.4s   \n"
            "fcmge  v6.4s,    v14.4s, %[vzero].4s\n"
            "fmul   v7.4s,    v14.4s, %[valpha].4s\n"
            "fcmge  v0.4s,    v15.4s, %[vzero].4s\n"
            "fmul   v1.4s,    v15.4s, %[valpha].4s\n"
            "bif    v12.16b,  v3.16b,   v2.16b  \n"
            "bif    v13.16b,  v5.16b,   v4.16b  \n"
            "bif    v14.16b,  v7.16b,   v6.16b  \n"
            "bif    v15.16b,  v1.16b,   v0.16b  \n"
            "10: \n"
            "st1 {v8.4s, v9.4s},[%[c_ptr0]], #32\n"
            "st1 {v10.4s, v11.4s},[%[c_ptr1]], #32\n"
            "st1 {v12.4s, v13.4s},[%[c_ptr2]], #32\n"
            "st1 {v14.4s, v15.4s},[%[c_ptr3]], #32\n"
            : [a_ptr] "+r"(a_ptr),
              [b_ptr] "+r"(b_ptr),
              [c_ptr0] "+r"(c_ptr0),
              [c_ptr1] "+r"(c_ptr1),
              [c_ptr2] "+r"(c_ptr2),
              [c_ptr3] "+r"(c_ptr3),
              [k] "+r"(k),
              [tail] "+r"(tails)
            : [bias_ptr] "r"(bias_local),
              [beta] "r"(beta),
              [alpha] "r"(alpha),
              [flag_act] "r"(flag_act),
              [vzero] "w"(vzero),
              [valpha] "w"(valpha)
            : "cc","memory",
              "v0","v1","v2","v3","v4","v5","v6","v7",
              "v8","v9","v10","v11","v12","v13",
              "v14","v15");
        // clang-format on
        if (flag_p_remain && (xb == bblocks - 1)) {
          for (int i = 0; i < remain; ++i) {
            *pout0++ = cout0[i];
            *pout1++ = cout1[i];
            *pout2++ = cout2[i];
            *pout3++ = cout3[i];
          }
        }
      }
    }
  }
}

void sgemm_prepacked_4x4(bool is_transB,
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
  size_t l2_cache = ctx->llc_size() > 0 ? ctx->llc_size() : 512 * 1024;
  auto workspace = ctx->workspace_data<float>();
  int threads = ctx->threads();

  auto act_type = act_param.active_type;
  float alpha[4] = {0.f, 0.f, 0.f, 0.f};
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    }
  }
  const int n_block = 4;
  const int m_block = 4;
  //! MBLOCK * x (result) + MBLOCK * k (A) + x * k (B) = l2
  int x_block = (l2_cache - (m_block * K)) / (sizeof(float) * (K + m_block));
  x_block /= n_block;
  x_block *= n_block;
  int x_num = (N + (x_block - 1)) / x_block;
  x_block = (N + x_num - 1) / x_num;
  x_block = (x_block + n_block - 1) / n_block;
  x_block *= n_block;
  x_block = x_block < n_block ? n_block : x_block;

  // unroll 2 loop
  int tail_pre = (K & (KBLOCK - 1));
  int k_pre = ((K + KBLOCK - 1) / KBLOCK) - 1;
  if (tail_pre == 0) {
    tail_pre = KBLOCK;
  }

  bool flag_p_remain = false;
  int remain = 0;

  int has_beta = fabsf(beta) > 1e-8f ? 1 : 0;
  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    if (xmax > N) {
      xmax = N;
    }
    int bblocks = (xmax - x0 + n_block - 1) / n_block;
    remain = xmax - x0 - (bblocks - 1) * n_block;
    if (remain > 0) {
      flag_p_remain = true;
    }
    //! load bpanel
    float *b_pannel = workspace;
    if (is_transB) {
      pack_m4(b_pannel, B, 1.0f, ldb, x0, xmax, 0, K);
    } else {
      pack_trans_m4(b_pannel, B, 1.0f, ldb, x0, xmax, 0, K);
    }
#pragma omp parallel for num_threads(threads)
    for (unsigned int y = 0; y < M; y += m_block) {
      unsigned int ymax = y + m_block;
      if (ymax > M) {
        ymax = M;
      }

      float bias_local[4] = {0};
      if (has_bias) {
        bias_local[0] = bias[y];
        bias_local[1] = bias[y + 1];
        bias_local[2] = bias[y + 2];
        bias_local[3] = bias[y + 3];
      }

      float cout0[n_block];  // NOLINT
      float cout1[n_block];  // NOLINT
      float cout2[n_block];  // NOLINT
      float cout3[n_block];  // NOLINT

      float *c_ptr0 = C + y * ldc + x0;
      float *c_ptr1 = c_ptr0 + ldc;
      float *c_ptr2 = c_ptr1 + ldc;
      float *c_ptr3 = c_ptr2 + ldc;

      float *pout0 = c_ptr0;
      float *pout1 = c_ptr1;
      float *pout2 = c_ptr2;
      float *pout3 = c_ptr3;

      const float *a_ptr_l = A_packed + y * K;
      const float *b_ptr_l = b_pannel;
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
          pout0 = c_ptr0;
          pout1 = c_ptr1;
          pout2 = c_ptr2;
          pout3 = c_ptr3;

          c_ptr0 = cout0;
          c_ptr1 = cout1;
          c_ptr2 = cout2;
          c_ptr3 = cout3;
          if (has_beta) {
            for (int i = 0; i < remain; ++i) {
              cout0[i] = pout0[i];
              cout1[i] = pout1[i];
              cout2[i] = pout2[i];
              cout3[i] = pout3[i];
            }
          }
        }
        const float *a_ptr = a_ptr_l;
        const float *b_ptr = b_ptr_l + xb * K * 4;
        int tail = tail_pre;
        int k = k_pre;
        // clang-format off
        asm volatile(
            "prfm   pldl1keep, [%[a_ptr]]\n"       /* preload a*/
            "ld1    {v2.4s}, [%[bias_ptr]]\n"         /* load bias to q2, q3*/
            "dup    v8.4s,  v2.s[0]\n"               /* out0 = 0 */
            "prfm   pldl1keep, [%[b_ptr]]\n"       /* preload b*/
            "dup    v9.4s,  v2.s[1]\n"               /* out1 = 0*/
            "prfm   pldl1keep, [%[a_ptr], #64]\n"  /* preload a*/
            "dup    v10.4s, v2.s[2]\n"               /* out2 = 0*/
            "prfm   pldl1keep, [%[b_ptr], #64]\n"  /* preload b*/
            "dup    v11.4s, v2.s[3]\n"               /* out3 = 0*/
            "cbz    %w[has_beta], 0f\n"            /* check beta == 0? */
            /* process beta */
            "dup    v7.4s, %w[beta]\n"                    /* beta to vector */
            "ld1    {v0.4s}, [%[c_ptr0]]\n" /* load output r0 */
            "ld1    {v1.4s}, [%[c_ptr1]]\n" /* load output r1 */
            "fmla   v8.4s, v0.4s, v7.4s\n"  /* cr00 += beta * c_r00*/
            "fmla   v9.4s, v1.4s, v7.4s\n"  /* cr10 += beta * c_r10*/
            "ld1    {v2.4s}, [%[c_ptr2]]\n"
            "ld1    {v3.4s}, [%[c_ptr3]]\n"
            "fmla   v10.4s, v2.4s, v7.4s\n" /* cr20 += beta * c_r20*/
            "fmla   v11.4s, v3.4s, v7.4s\n" /* cr30 += beta * c_r30*/

            "0: \n"                          /* check loop count */
            "ldp	q0, q1, [%[a_ptr]], #32\n" /* load a00,a10 to q0, q1*/
            "ldp	q4, q5, [%[b_ptr]], #32\n" /* load b0, b1 to q4, q5*/
            "cbz	%w[k], 2f\n"               /* check loop count > 0 */
            /* main loop */
            /* unrool 0*/
            "1:\n"                              /* main loop */
            "fmla 	v8.4s,   v4.4s,  v0.s[0]\n"   /* out0 = b0 * a00[0], b0 =q4 */
            "fmla  	v9.4s,   v4.4s,  v0.s[1]\n"   /* out1 = b0 * a00[1], b0 =q4 */
            "ldp	q6, q7, [%[b_ptr]], #32\n"    /* load b2, b3 to q6, q7 */
            "fmla	v10.4s,  v4.4s,  v0.s[2]\n"   /* out2 = b0 * a00[2], b0 =q4 */
            "fmla	v11.4s,  v4.4s,  v0.s[3]\n"   /* out3 = b0 * a00[3], b0 =q4 */

            "ldp	q2, q3, [%[a_ptr]], #32\n"    /* load a20, a30 to q2, q3 */
            "fmla 	v8.4s,   v5.4s,  v1.s[0]\n"   /* out0 = b1 * a10[0], b1 =q5 */
            "fmla	v9.4s,   v5.4s,  v1.s[1]\n"   /* out1 = b1 * a10[1], b1 =q5 */
            "fmla	v10.4s,  v5.4s,  v1.s[2]\n"   /* out2 = b1 * a10[2], b1 =q5 */
            "fmla	v11.4s,  v5.4s,  v1.s[3]\n"   /* out3 = b1 * a10[3], b1 =q5 */
            "ldp	q4, q5, [%[b_ptr]], #32\n"    /* load b0, b1 to q4, q5*/

            "fmla	v8.4s,   v6.4s,  v2.s[0]\n"   /* out0 = b2 * a20[0], b2 =q6 */
            "fmla	v9.4s,   v6.4s,  v2.s[1]\n"   /* out1 = b2 * a20[1], b2 =q6 */
            "fmla	v10.4s,  v6.4s,  v2.s[2]\n"   /* out2 = b2 * a20[2], b2 =q6*/
            "fmla	v11.4s,  v6.4s,  v2.s[3]\n"   /* out3 = b2 * a20[3], b2 =q6*/
            "ldp	q0, q1, [%[a_ptr]], #32\n"    /* load a00, a10 to q0, q1 */

            "fmla	v8.4s,   v7.4s,  v3.s[0]\n"   /* out0 = b3 * a30[0], b3 =q7*/
            "fmla	v9.4s,   v7.4s,  v3.s[1]\n"   /* out1 = b3 * a30[1], b3 =q7*/
            "subs	%w[k], %w[k], #1\n"         /* loop count - 1*/
            "fmla	v10.4s,  v7.4s,  v3.s[2]\n"   /* out2 = b3 * a30[2], b3 =q7*/
            "fmla	v11.4s,  v7.4s,  v3.s[3]\n"   /* out3 = b3 * a30[3], b3 =q7*/

            "bne	1b\n"
            "2:\n"                            /* process tail*/
            "subs		%w[tail], %w[tail], #1\n" /* tail--*/
            "beq		3f\n"                     /*jump to tail = 1*/
            /* final unrool 0*/
            /* unrool 0, tail > 1*/
            "fmla 	v8.4s,   v4.4s,  v0.s[0]\n"   /* out0 = b0 * a00[0], b0 =q4 */
            "fmla	v9.4s,   v4.4s,  v0.s[1]\n"   /* out1 = b0 * a00[1], b0 =q4 */
            "subs	%w[tail], %w[tail], #1\n"      /* tail--*/
            "fmla	v10.4s,  v4.4s,  v0.s[2]\n"   /* out2 = b0 * a00[2], b0 =q4 */
            "fmla	v11.4s,  v4.4s,  v0.s[3]\n"   /* out3 = b0 * a00[3], b0 =q4 */

            "beq		4f\n"                     /*jump to tail = 2*/
            /* unrool 1, tail > 2*/
            "ldp	q6, q7, [%[b_ptr]], #32\n"    /* load b2, b3 to q6, q7 */

            "fmla 	v8.4s,   v5.4s,  v1.s[0]\n"   /* out0 = b1 * a10[0], b1 =q5 */
            "fmla       v9.4s,   v5.4s,  v1.s[1]\n"   /* out1 = b1 * a10[1], b1 =q5*/
            "subs	%w[tail], %w[tail], #1\n"      /* tail--*/
            "fmla	v10.4s,  v5.4s,  v1.s[2]\n"   /* out2 = b1 * a10[2], b1 =q5 */
            "fmla	v11.4s,  v5.4s,  v1.s[3]\n"   /* out3 = b1 * a10[3], b1 =q5 */
            "ldp	q2, q3, [%[a_ptr]], #32\n"    /* load a20, a30 to q2, q3 */

            "beq		5f\n"                       /*jump to tail = 3*/
            /* unrool 2, tail = 4*/
            "fmla	v8.4s,   v6.4s,  v2.s[0]\n"   /* out0 = b2 * a20[0], b1 =q6 */
            "fmla	v9.4s,   v6.4s,  v2.s[1]\n"   /* out1 = b2 * a20[1], b1 =q6 */
            "fmla	v10.4s,  v6.4s,  v2.s[2]\n"   /* out2 = b2 * a20[2], b1 =q6*/
            "fmla	v11.4s,  v6.4s,  v2.s[3]\n"   /* out3 = b2 * a20[3], b1 =q6*/

            /* unrool 3, tail = 4*/

            "fmla	v8.4s,   v7.4s,  v3.s[0]\n"   /* out0 = b3 * a30[0], b3 =q7*/
            "fmla	v9.4s,   v7.4s,  v3.s[1]\n"   /* out1 = b3 * a30[1], b3 =q7*/
            "fmla	v10.4s,  v7.4s,  v3.s[2]\n"   /* out2 = b3 * a30[2], b3 =q7*/
            "fmla	v11.4s,  v7.4s,  v3.s[3]\n"   /* out3 = b3 * a30[3], b3 =q7*/

            "b		11f\n"
            /* tails==1 final tail*/
            "3: \n"                            /* tail=1*/
            "fmla 	v8.4s,   v4.4s,  v0.s[0]\n"   /* out0 = b0 * a00[0], b0 =q4 */
            "fmla	v9.4s,   v4.4s,  v0.s[1]\n"   /* out1 = b0 * a00[1], b0 =q4 */
            "fmla	v10.4s,  v4.4s,  v0.s[2]\n"   /* out2 = b0 * a00[2], b0 =q4 */
            "fmla	v11.4s,  v4.4s,  v0.s[3]\n"   /* out3 = b0 * a00[3], b0 =q4 */

            "b		11f\n"
            /* tails==2 final tail*/
            "4:\n"                              /* tail = 2*/

            "fmla 	v8.4s,   v5.4s,  v1.s[0]\n"   /* out0 = b1 * a10[0], b1 =q5 */
            "fmla       v9.4s,   v5.4s,  v1.s[1]\n"   /* out1 = b1 * a10[1], b1 =q5*/
            "fmla	v10.4s,  v5.4s,  v1.s[2]\n"   /* out2 = b1 * a10[2], b1 =q5 */
            "fmla	v11.4s,  v5.4s,  v1.s[3]\n"   /* out3 = b1 * a10[3], b1 =q5 */

            "b		11f\n"
            /* tails==3 final tail*/
            "5:\n"                              /* tail = 3*/
            "fmla	v8.4s,   v6.4s,  v2.s[0]\n"   /* out0 = b2 * a20[0], b1 =q6 */
            "fmla	v9.4s,   v6.4s,  v2.s[1]\n"   /* out1 = b2 * a20[1], b1 =q6 */
            "fmla	v10.4s,  v6.4s,  v2.s[2]\n"   /* out2 = b2 * a20[2], b1 =q6*/
            "fmla	v11.4s,  v6.4s,  v2.s[3]\n"   /* out3 = b2 * a20[3], b1 =q6*/

            "11:                           \n"   /* check activation */
            "cmp    %w[flag_act],   #1     \n"   /* check if has relu */
            "bne    12f                    \n"   /* jump if no relu */
            "movi   v0.4s,  #0             \n"   /* for relu*/
            "fmax   v8.4s,  v8.4s,  v0.4s  \n"   /* relu*/
            "fmax   v9.4s,  v9.4s,  v0.4s  \n"   /* relu*/
            "fmax   v10.4s, v10.4s, v0.4s  \n"   /* relu*/
            "fmax   v11.4s, v11.4s, v0.4s  \n"   /* relu*/
            "b      20f                    \n"   /* relu end */
            //! no act
            "12:                           \n"   /* no relu */
            "cmp   %w[flag_act],  #0       \n"   /* check no act */
            "beq   20f                     \n"   /* no act end */
            //! relu6
            "cmp    %w[flag_act],  #2      \n"    /* check if has relu6 */
            "bne    13f                    \n"    /* jump if no relu6 */
            "movi   v0.4s, #0              \n"    /* for relu6 */
            "ld1    {v1.4s}, [%[alpha]]    \n"    /* relu6 alpha */
            "fmax   v8.4s,  v8.4s,  v0.4s  \n"    /* relu6 */
            "fmax   v9.4s,  v9.4s,  v0.4s  \n"    /* relu6 */
            "fmax   v10.4s, v10.4s, v0.4s  \n"    /* relu6 */
            "fmax   v11.4s, v11.4s, v0.4s  \n"    /* relu6 */

            "fmin   v8.4s,  v8.4s,  v1.4s  \n"    /* relu6*/
            "fmin   v9.4s,  v9.4s,  v1.4s  \n"    /* relu6*/
            "fmin   v10.4s, v10.4s, v1.4s  \n"    /* relu6*/
            "fmin   v11.4s, v11.4s, v1.4s  \n"    /* relu6*/
            "b      20f                    \n"    /* relu6 end */
            //! leakey relu
            "13:                                \n" /* otherwise is leakey relu */
            "movi   v0.4s,    #0                \n" /* for leakey relu */
            "ld1    {v1.4s},  [%[alpha]]        \n" /* leakey relu alpha */
            "fcmge  v2.4s,    v8.4s,    v0.4s   \n" /* vcgeq_f32 */
            "fmul   v3.4s,    v8.4s,    v1.4s   \n" /* vmulq_f32 */
            "fcmge  v4.4s,    v9.4s,    v0.4s   \n" /* vcgeq_f32 */
            "fmul   v5.4s,    v9.4s,    v1.4s   \n" /* vmulq_f32 */
            "fcmge  v6.4s,    v10.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v7.4s,    v10.4s,   v1.4s   \n" /* vmulq_f32 */
            "fcmge  v12.4s,   v11.4s,   v0.4s   \n" /* vcgeq_f32 */
            "fmul   v13.4s,   v11.4s,   v1.4s   \n" /* vmulq_f32 */
            "bif    v8.16b,   v3.16b,   v2.16b  \n" /* choose*/
            "bif    v9.16b,   v5.16b,   v4.16b  \n" /* choose*/
            "bif    v10.16b,  v7.16b,   v6.16b  \n" /* choose*/
            "bif    v11.16b,  v13.16b,  v12.16b \n" /* choose*/
            "20:                                \n" /* act end */
            "st1 {v8.4s}, [%[c_ptr0]], #16\n"   /* store r0 */
            "st1 {v9.4s}, [%[c_ptr1]], #16\n" /* store r1 */
            "st1 {v10.4s}, [%[c_ptr2]], #16\n" /* store r2 */
            "st1 {v11.4s}, [%[c_ptr3]], #16\n" /* store r3 */

            : [a_ptr] "+r"(a_ptr),
              [b_ptr] "+r"(b_ptr),
              [k] "+r"(k),
              [tail] "+r"(tail),
              [c_ptr0] "+r"(c_ptr0),
              [c_ptr1] "+r"(c_ptr1),
              [c_ptr2] "+r"(c_ptr2),
              [c_ptr3] "+r"(c_ptr3)
            : [bias_ptr] "r"(bias_local),
              [has_beta] "r"(has_beta),
              [beta] "r"(beta),
              [alpha] "r"(alpha),
              [flag_act] "r"(flag_act)
            : "cc","memory",
              "v0","v1","v2","v3","v4","v5","v6","v7",
              "v8","v9","v10","v11");
        // clang-format on
        if (flag_p_remain && (xb == bblocks - 1)) {
          for (int i = 0; i < remain; ++i) {
            *pout0++ = cout0[i];
            *pout1++ = cout1[i];
            *pout2++ = cout2[i];
            *pout3++ = cout3[i];
          }
        }
      }
    }
  }
}
#else  // __aarch64__
/**
 * \brief gemm with ablock = 6, bblock = 8, output 6x8
 * @param A
 * @param B
 * @param C
 * @param M
 * @param N
 * @param K
 * @param threads
 * @param workspace
 */
void sgemm_prepacked_6x8(bool is_transB,
                         int M,
                         int N,
                         int K,
                         const float* A_packed,
                         const float* B,
                         int ldb,
                         float beta,
                         float* C,
                         int ldc,
                         const float* bias,
                         bool has_bias,
                         const operators::ActivationParam act_param,
                         ARMContext* ctx) {
  size_t l2_cache = ctx->llc_size() > 0 ? ctx->llc_size() : 512 * 1024;
  auto* workspace = ctx->workspace_data<float>();
  int threads = ctx->threads();
  auto act_type = act_param.active_type;
  float alpha[4] = {0.f, 0.f, 0.f, 0.f};
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    }
  }
  //! MBLOCK * x (result) + MBLOCK * k (A) + x * k (B) = l2
  int x_block =
      (l2_cache - (MBLOCK_OTH * K)) / (sizeof(float) * (K + MBLOCK_OTH));
  x_block /= NBLOCK;
  x_block *= NBLOCK;
  int x_num = (N + (x_block - 1)) / x_block;
  x_block = (N + x_num - 1) / x_num;
  x_block = (x_block + NBLOCK - 1) / NBLOCK;
  x_block *= NBLOCK;
  x_block = x_block < NBLOCK ? NBLOCK : x_block;

  int k_pre = ((K + KBLOCK - 1) / KBLOCK) - 1;
  int tail_pre = (K & (KBLOCK - 1));
  if (tail_pre == 0) {
    tail_pre = KBLOCK;
  }

  bool flag_p_remain = false;
  int remain = 0;

  int has_beta = fabsf(beta) > 1e-8f ? 1 : 0;
  //! merge tail_pre and flag_act
  tail_pre = (tail_pre << 2 | flag_act);

  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    if (xmax > N) {
      xmax = N;
    }
    int bblocks = (xmax - x0 + NBLOCK - 1) / NBLOCK;
    remain = xmax - x0 - (bblocks - 1) * NBLOCK;
    if (remain > 0) {
      flag_p_remain = true;
    }
    //! load bpanel
    auto b_pannel = static_cast<float*>(workspace);
    if (is_transB) {
      loadb_trans(b_pannel, B, ldb, 0, K, x0, xmax);
    } else {
      loadb(b_pannel, B, ldb, 0, K, x0, xmax);
    }
#pragma omp parallel for num_threads(threads)
    for (unsigned int y = 0; y < M; y += MBLOCK_OTH) {
      unsigned int ymax = y + MBLOCK_OTH;
      if (ymax > M) {
        ymax = M;
      }
      float* c_ptr0 = C + y * ldc + x0;
      float* c_ptr1 = c_ptr0 + ldc;
      float* c_ptr2 = c_ptr1 + ldc;
      float* c_ptr3 = c_ptr2 + ldc;
      float* c_ptr4 = c_ptr3 + ldc;
      float* c_ptr5 = c_ptr4 + ldc;

      float* pout0 = c_ptr0;
      float* pout1 = c_ptr1;
      float* pout2 = c_ptr2;
      float* pout3 = c_ptr3;
      float* pout4 = c_ptr4;
      float* pout5 = c_ptr5;

      float bias_local[6] = {0};
      if (has_bias) {
        bias_local[0] = bias[y];
        bias_local[1] = bias[y + 1];
        bias_local[2] = bias[y + 2];
        bias_local[3] = bias[y + 3];
        bias_local[4] = bias[y + 4];
        bias_local[5] = bias[y + 5];
      }

      float cout0[NBLOCK];
      float cout1[NBLOCK];
      float cout2[NBLOCK];
      float cout3[NBLOCK];
      float cout4[NBLOCK];
      float cout5[NBLOCK];

      const float* a_ptr_l = A_packed + y * K;
      const float* b_ptr = b_pannel;
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
          pout0 = c_ptr0;
          pout1 = c_ptr1;
          pout2 = c_ptr2;
          pout3 = c_ptr3;
          pout4 = c_ptr4;
          pout5 = c_ptr5;

          c_ptr0 = cout0;
          c_ptr1 = cout1;
          c_ptr2 = cout2;
          c_ptr3 = cout3;
          c_ptr4 = cout4;
          c_ptr5 = cout5;
          if (has_beta) {
            for (int i = 0; i < remain; ++i) {
              cout0[i] = pout0[i];
              cout1[i] = pout1[i];
              cout2[i] = pout2[i];
              cout3[i] = pout3[i];
              cout4[i] = pout4[i];
              cout5[i] = pout5[i];
            }
          }
        }
        const float* a_ptr = a_ptr_l;
        int tails = tail_pre;
        int k = k_pre;
        // clang-format off
        asm volatile(
            // sgemm 6x8
            "vld1.32	{d2-d4}, [%[bias_ptr]]      @ load bias 6 elements\n"
            "pld [%[a_ptr]]                         @ preload a\n"
            "vdup.i32	q12,d4[0]                   @ out40=0\n"
            "pld [%[b_ptr]]                         @ preload b\n"
            "vdup.i32	q13,d4[0]                   @ out41=0\n"
            "pld [%[a_ptr], #64]                    @ preload a\n"
            "vdup.i32	q14,d4[1]                   @ out50=0\n"
            "pld [%[b_ptr], #64]                    @ preload b\n"
            "vdup.i32	q15,d4[1]                   @ out51=0\n"
            "pld [%[a_ptr], #128]                   @ preload a\n"
            "vdup.i32	q4, d2[0]                   @ out00=0\n"
            "pld [%[b_ptr], #128]                   @ preload b\n"
            "vdup.i32	q5, d2[0]                   @ out01=0\n"
            "vdup.i32	q6, d2[1]                   @ out10=0\n"
            "pld [%[a_ptr], #192]                   @ preload a\n"
            "vdup.i32	q7, d2[1]                   @ out11=0\n"
            "pld [%[b_ptr], #192]                   @ preload a\n"
            "vdup.i32	q8, d3[0]                   @ out20=0\n"
            "pld [%[a_ptr], #256]                   @ preload a\n"
            "vdup.i32	q9, d3[0]                   @ out21=0\n"
            "pld [%[b_ptr], #256]                   @ preload a\n"
            "vdup.i32	q10,d3[1]                   @ out30=0\n"
            "pld [%[b_ptr], #320]                   @ preload b\n"
            "vdup.i32	q11,d3[1]                   @ out31=0\n"
            "pld [%[b_ptr], #384]                   @ preload b\n"
            "cmp %[beta], #0\n"
            "beq    11f\n" /* check beta == 0? */
            /* process beta */
            "vdup.32    q3, %[beta]\n"          /* beta to vector */
            "vld1.32    {d0-d3}, [%[c_ptr0]]\n" /* load output r0 */
            "vmla.f32   q4, q0, q3\n"           /* cr00 += beta * c_r00 */
            "vmla.f32   q5, q1, q3\n"           /* cr01 += beta * c_r01 */
            "vld1.32    {d0-d3}, [%[c_ptr1]]\n" /* load output r1 */
            "vmla.f32   q6, q0, q3\n"           /* cr10 += beta * c_r10 */
            "vmla.f32   q7, q1, q3\n"           /* cr11 += beta * c_r11 */
            "vld1.32    {d0-d3}, [%[c_ptr2]]\n" /* load output r2 */
            "vmla.f32   q8, q0, q3\n"           /* cr20 += beta * c_r20 */
            "vmla.f32   q9, q1, q3\n"           /* cr21 += beta * c_r21 */
            "vld1.32    {d0-d3}, [%[c_ptr3]]\n" /* load output r3 */
            "vmla.f32   q10, q0, q3\n"          /* cr30 += beta * c_r30 */
            "vmla.f32   q11, q1, q3\n"          /* cr31 += beta * c_r31 */
            "vld1.32    {d0-d3}, [%[c_ptr4]]\n" /* load output r4 */
            "vmla.f32   q12, q0, q3\n"          /* cr40 += beta * c_r40 */
            "vmla.f32   q13, q1, q3\n"          /* cr41 += beta * c_r41 */
            "vld1.32    {d0-d3}, [%[c_ptr5]]\n" /* load output r5 */
            "vmla.f32   q14, q0, q3\n"          /* cr50 += beta * c_r50 */
            "vmla.f32   q15, q1, q3\n"          /* cr51 += beta * c_r51 */
            "11: \n"                            /* check loop count */
            "vld1.32	{d0-d1}, [%[a_ptr] :64]!    @ load a0~a3\n"
            "vld1.32	{d4-d5}, [%[b_ptr] :128]!   @ load b1\n"
            "cmp %[k], #0                     @ check weather k is bigger than "
            "0\n"
            "beq 0f                                 @ jump to tail\n"
            "1:                                     @ main loop for k\n"
            /* Unroll 0*/
            "vld1.32	{d2-d3}, [%[a_ptr] :64]!    @ load a4, a5, and next a0, "
            "a1\n"
            "vmla.f32	q4, q2, d0[0]               @ out0 += b1 * a0\n"
            "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"
            "vmla.f32	q6, q2, d0[1]               @ out1 += b1 * a1\n"
            "vmla.f32	q8, q2, d1[0]               @ out2 += b1 * a2\n"
            "vmla.f32	q10, q2, d1[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q12, q2, d2[0]              @ out4 += b1 * a4\n"
            "vmla.f32	q14, q2, d2[1]              @ out5 += b1 * a5\n"
            "vld1.32	{d4-d5}, [%[b_ptr] :128]!   @ load b1\n"
            "vmla.f32	q5, q3, d0[0]               @ out6 += b2 * a0\n"
            "vmla.f32	q7, q3, d0[1]               @ out7 += b2 * a1\n"
            "vmla.f32	q9, q3, d1[0]               @ out8 += b2 * a2\n"
            "vmla.f32	q11, q3, d1[1]              @ out9 += b2 * a3\n"
            "vld1.32	{d0-d1}, [%[a_ptr] :64]!    @ load a2~a5\n"
            "vmla.f32	q13, q3, d2[0]              @ out10 += b2 * a4\n"
            "vmla.f32	q15, q3, d2[1]              @ out11 += b2 * a5\n"
            "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"
            /* Unroll 1 */
            "vmla.f32	q4, q2, d3[0]               @ out0 += b1 * a0\n"
            "vmla.f32	q6, q2, d3[1]               @ out1 += b1 * a1\n"
            /*"pld [%[a_ptr], #64]                    @ preload a\n"*/
            "vmla.f32	q8, q2, d0[0]               @ out2 += b1 * a2\n"
            "vmla.f32	q10, q2, d0[1]              @ out3 += b1 * a3\n"
            /*"pld [%[b_ptr], #192]\n"*/
            "vmla.f32	q12, q2, d1[0]              @ out4 += b1 * a4\n"
            "vmla.f32	q14, q2, d1[1]              @ out5 += b1 * a5\n"
            "vld1.32	{d4-d5}, [%[b_ptr] :128]!   @ load b1\n"
            "vmla.f32	q5, q3, d3[0]               @ out6 += b2 * a0\n"
            "vmla.f32	q7, q3, d3[1]               @ out7 += b2 * a1\n"
            "vld1.32	{d2-d3}, [%[a_ptr] :64]!    @ load a0~a3\n"
            "vmla.f32	q9, q3, d0[0]               @ out8 += b2 * a2\n"
            "vmla.f32	q11, q3, d0[1]              @ out9 += b2 * a3\n"
            "vmla.f32	q13, q3, d1[0]              @ out10 += b2 * a4\n"
            "vmla.f32	q15, q3, d1[1]              @ out11 += b2 * a5\n"
            "vld1.32	{d0-d1}, [%[a_ptr] :64]!    @ load a4, a5, a0, a1\n"
            /* Unroll 2 */
            "vmla.f32	q4, q2, d2[0]               @ out0 += b1 * a0\n"
            "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"
            "vmla.f32	q6, q2, d2[1]               @ out1 += b1 * a1\n"
            "vmla.f32	q8, q2, d3[0]               @ out2 += b1 * a2\n"
            "vmla.f32	q10, q2, d3[1]              @ out3 += b1 * a3\n"
            /*"pld [%[a_ptr], #240]                   @ preload\n"*/
            "vmla.f32	q12, q2, d0[0]              @ out4 += b1 * a4\n"
            "vmla.f32	q14, q2, d0[1]              @ out5 += b1 * a5\n"
            "vld1.32	{d4-d5}, [%[b_ptr] :128]!   @ load b1\n"
            "vmla.f32	q5, q3, d2[0]               @ out6 += b2 * a0\n"
            "vmla.f32	q7, q3, d2[1]               @ out7 += b2 * a1\n"
            /*"pld [%[b_ptr], #208]\n"*/
            "vmla.f32	q9, q3, d3[0]               @ out8 += b2 * a2\n"
            "vmla.f32	q11, q3, d3[1]              @ out9 += b2 * a3\n"
            "vld1.32	{d2-d3}, [%[a_ptr] :64]!    @ load a2~a5\n"
            "vmla.f32	q13, q3, d0[0]              @ out10 += b2 * a4\n"
            "vmla.f32	q15, q3, d0[1]              @ out11 += b2 * a5\n"
            "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"
            /* Unroll 3 */
            "vmla.f32	q4, q2, d1[0]               @ out0 += b1 * a0\n"
            "vmla.f32	q6, q2, d1[1]               @ out1 += b1 * a1\n"
            "vmla.f32	q8, q2, d2[0]               @ out2 += b1 * a2\n"
            "vmla.f32	q10, q2, d2[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q12, q2, d3[0]              @ out4 += b1 * a4\n"
            "vmla.f32	q14, q2, d3[1]              @ out5 += b1 * a5\n"
            "vld1.32	{d4-d5}, [%[b_ptr] :128]!   @ load b1\n"
            "vmla.f32	q5, q3, d1[0]               @ out6 += b2 * a0\n"
            "vmla.f32	q7, q3, d1[1]               @ out7 += b2 * a1\n"
            "vld1.32	{d0-d1}, [%[a_ptr] :64]!    @ load a0~a3\n"
            "vmla.f32	q9, q3, d2[0]               @ out8 += b2 * a2\n"
            "vmla.f32	q11, q3, d2[1]              @ out9 += b2 * a3\n"
            "subs		%[k], %[k], #1                @ k--\n"
            "vmla.f32	q13, q3, d3[0]              @ out10 += b2 * a4\n"
            "vmla.f32	q15, q3, d3[1]              @ out11 += b2 * a5\n"
            "bne		1b                            @ jump to main loop\n"
            "0:                                   @ process tail\n"
            "sub		%[tails], %[tails], #4        @ tail--\n"
            "cmp    %[tails], #4                  @ cmp with act bits\n"
            "blt		3f                            @ jump to tail = 1\n"
            /* Unroll 0*/
            "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"
            "vmla.f32	q4, q2, d0[0]               @ out0 += b1 * a0\n"
            "vld1.32	{d2-d3}, [%[a_ptr] :64]!    @ load a4,5, a0, a1\n"
            "vmla.f32	q6, q2, d0[1]               @ out1 += b1 * a1\n"
            "vmla.f32	q8, q2, d1[0]               @ out2 += b1 * a2\n"
            "vmla.f32	q10, q2, d1[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q12, q2, d2[0]              @ out4 += b1 * a4\n"
            "sub		%[tails], %[tails], #4        @ tail--\n"
            "vmla.f32	q14, q2, d2[1]              @ out5 += b1 * a5\n"
            "vld1.32	{d4-d5}, [%[b_ptr] :128]!   @ load b1\n"
            "cmp    %[tails], #4                  @ cmp with act bits\n"
            "vmla.f32	q5, q3, d0[0]               @ out6 += b2 * a0\n"
            "vmla.f32	q7, q3, d0[1]               @ out7 += b2 * a1\n"
            "vmla.f32	q9, q3, d1[0]               @ out8 += b2 * a2\n"
            "vmla.f32	q11, q3, d1[1]              @ out9 += b2 * a3\n"
            "vld1.32	{d0-d1}, [%[a_ptr] :64]!    @ load a2~a5\n"
            "vmla.f32	q13, q3, d2[0]              @ out10 += b2 * a4\n"
            "vmla.f32	q15, q3, d2[1]              @ out11 += b2 * a5\n"
            "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"
            "blt		4f                            @ jump to tail==2\n"
            /* Unroll 1*/
            "vmla.f32	q4, q2, d3[0]               @ out0 += b1 * a0\n"
            "vmla.f32	q6, q2, d3[1]               @ out1 += b1 * a1\n"
            "sub		%[tails], %[tails], #4        @ tail--\n"
            "vmla.f32	q8, q2, d0[0]               @ out2 += b1 * a2\n"
            "vmla.f32	q10, q2, d0[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q12, q2, d1[0]              @ out4 += b1 * a4\n"
            "vmla.f32	q14, q2, d1[1]              @ out5 += b1 * a5\n"
            "vld1.32	{d4-d5}, [%[b_ptr] :128]!   @ load b1\n"
            "cmp    %[tails],  #4                 @ cmp with act bits\n"
            "vmla.f32	q5, q3, d3[0]               @ out6 += b2 * a0\n"
            "vmla.f32	q7, q3, d3[1]               @ out7 += b2 * a1\n"
            "vld1.32	{d2-d3}, [%[a_ptr] :64]!    @ load a0~a3\n"
            "vmla.f32	q9, q3, d0[0]               @ out8 += b2 * a2\n"
            "vmla.f32	q11, q3, d0[1]              @ out9 += b2 * a3\n"
            "vmla.f32	q13, q3, d1[0]              @ out10 += b2 * a4\n"
            "vmla.f32	q15, q3, d1[1]              @ out11 += b2 * a5\n"
            "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"
            "blt		5f                            @ jump to tail==3\n"
            /* Unroll 2 */
            "sub		%[tails], %[tails], #4        @ tail--\n"
            "vld1.32	{d0-d1}, [%[a_ptr] :64]!    @ load a4,a5, a0,a1\n"
            "vmla.f32	q4, q2, d2[0]               @ out0 += b1 * a0\n"
            "vmla.f32	q6, q2, d2[1]               @ out1 += b1 * a1\n"
            "vmla.f32	q8, q2, d3[0]               @ out2 += b1 * a2\n"
            "vmla.f32	q10, q2, d3[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q12, q2, d0[0]              @ out4 += b1 * a4\n"
            "vmla.f32	q14, q2, d0[1]              @ out5 += b1 * a5\n"
            "vld1.32	{d4-d5}, [%[b_ptr] :128]!   @ load b1\n"
            "vmla.f32	q5, q3, d2[0]               @ out6 += b2 * a0\n"
            "vmla.f32	q7, q3, d2[1]               @ out7 += b2 * a1\n"
            "vmla.f32	q9, q3, d3[0]               @ out8 += b2 * a2\n"
            "vmla.f32	q11, q3, d3[1]              @ out9 += b2 * a3\n"
            "vld1.32	{d2-d3}, [%[a_ptr] :64]!    @ load a2~a5\n"
            "vmla.f32	q13, q3, d0[0]              @ out10 += b2 * a4\n"
            "vmla.f32	q15, q3, d0[1]              @ out11 += b2 * a5\n"
            "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"
            /* Unroll 3*/
            "vmla.f32	q4, q2, d1[0]               @ out0 += b1 * a0\n"
            "vmla.f32	q6, q2, d1[1]               @ out1 += b1 * a1\n"
            "vmla.f32	q8, q2, d2[0]               @ out2 += b1 * a2\n"
            "vmla.f32	q10, q2, d2[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q12, q2, d3[0]              @ out4 += b1 * a4\n"
            "vmla.f32	q14, q2, d3[1]              @ out5  += b1 * a5\n"
            "vmla.f32	q5, q3, d1[0]               @ out6 += b2 * a0\n"
            "vmla.f32	q7, q3, d1[1]               @ out7 += b2 * a1\n"
            "vmla.f32	q9, q3, d2[0]               @ out8 += b2 * a2\n"
            "vmla.f32	q11, q3, d2[1]              @ out9 += b2 * a3\n"
            "vmla.f32	q13, q3, d3[0]              @ out10 += b2 * a4\n"
            "vmla.f32	q15, q3, d3[1]              @ out11 += b2 * a5\n"
            "b		2f\n"
            /* tails==1 final tail*/
            "3:                                     @ tail=1\n"
            "vmla.f32	q4, q2, d0[0]               @ out0 += b1 * a0\n"
            "vld1.32	{d2}, [%[a_ptr] :64]!       @ load a4,a5\n"
            "vmla.f32	q6, q2, d0[1]               @ out1 += b1 * a1\n"
            "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"
            "vmla.f32	q8, q2, d1[0]               @ out2 += b1 * a2\n"
            "vmla.f32	q10, q2, d1[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q12, q2, d2[0]              @ out4 += b1 * a4\n"
            "vmla.f32	q14, q2, d2[1]              @ out5 += b1 * a5\n"
            "vmla.f32	q5, q3, d0[0]               @ out6 += b2 * a0\n"
            "vmla.f32	q7, q3, d0[1]               @ out7 += b2 * a1\n"
            "vmla.f32	q9, q3, d1[0]               @ out8 += b2 * a2\n"
            "vmla.f32	q11, q3, d1[1]              @ out9 += b2 * a3\n"
            "vmla.f32	q13, q3, d2[0]              @ out10 += b2 * a4\n"
            "vmla.f32	q15, q3, d2[1]              @ out11 += b2 * a5\n"
            "b		2f                              @ jump to end\n"
            /* tails==2 final tail*/
            "4:                                     @ tail == 2\n"
            "vmla.f32	q4, q2, d3[0]               @ out0 += b1 * a0\n"
            "vmla.f32	q6, q2, d3[1]               @ out1 += b1 * a1\n"
            "vmla.f32	q8, q2, d0[0]               @ out2 += b1 * a2\n"
            "vmla.f32	q10, q2, d0[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q12, q2, d1[0]              @ out4 += b1 * a4\n"
            "vmla.f32	q14, q2, d1[1]              @ out5 += b1 * a5\n"
            "vmla.f32	q5, q3, d3[0]               @ out6 += b2 * a0\n"
            "vmla.f32	q7, q3, d3[1]               @ out7 += b2 * a1\n"
            "vmla.f32	q9, q3, d0[0]               @ out8 += b2 * a2\n"
            "vmla.f32	q11, q3, d0[1]              @ out9 += b2 * a3\n"
            "vmla.f32	q13, q3, d1[0]              @ out10 += b2 * a4\n"
            "vmla.f32	q15, q3, d1[1]              @ out11 += b2 * a5\n"
            "b		2f                              @ jump to end\n"
            /* tails==3 final tail*/
            "5:                                     @ tail=3\n"
            "vmla.f32	q4, q2, d2[0]               @ out0 += b1 * a0\n"
            "vld1.32	{d0}, [%[a_ptr] :64]!       @ load a4,a5\n"
            "vmla.f32	q6, q2, d2[1]               @ out1 += b1 * a1\n"
            "vmla.f32	q8, q2, d3[0]               @ out2 += b1 * a2\n"
            "vmla.f32	q10, q2, d3[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q12, q2, d0[0]              @ out4 += b1 * a4\n"
            "vmla.f32	q14, q2, d0[1]              @ out5 += b1 * a5\n"
            "vmla.f32	q5, q3, d2[0]               @ out6 += b2 * a0\n"
            "vmla.f32	q7, q3, d2[1]               @ out7 += b2 * a1\n"
            "vmla.f32	q9, q3, d3[0]               @ out8 += b2 * a2\n"
            "vmla.f32	q11, q3, d3[1]              @ out9 += b2 * a3\n"
            "vmla.f32	q13, q3, d0[0]              @ out10 += b2 * a4\n"
            "vmla.f32	q15, q3, d0[1]              @ out11 += b2 * a5\n"
            "2:                                   @ check activation\n"
            //!   relu
            "cmp        %[tails], #1              @ check if has relu\n"
            "bne        6f                        @ jump if not relu \n"
            "vmov.u32   q0, #0                    @ for relu\n"
            "vmax.f32   q4, q4, q0                @ for relu\n"
            "vmax.f32   q5, q5, q0                @ for relu\n"
            "vmax.f32   q6, q6, q0                @ for relu\n"
            "vmax.f32   q7, q7, q0                @ for relu\n"
            "vmax.f32   q8, q8, q0                @ for relu\n"
            "vmax.f32   q9, q9, q0                @ for relu\n"
            "vmax.f32   q10, q10, q0              @ for relu\n"
            "vmax.f32   q11, q11, q0              @ for relu\n"
            "vmax.f32   q12, q12, q0              @ for relu\n"
            "vmax.f32   q13, q13, q0              @ for relu\n"
            "vmax.f32   q14, q14, q0              @ for relu\n"
            "vmax.f32   q15, q15, q0              @ for relu\n"
            "b          10f                       @ relu end\n"
            "6:                                   @ no relu \n"
            "cmp        %[tails], #0              @ check no act\n"
            "beq        10f                       @ no act end  \n"
            //!   relu6
            "cmp        %[tails], #2              @ check if has relu6\n"
            "bne        7f                        @ jump if no relu6 \n"
            "vmov.u32   q0, #0                    @ for relu6\n"
            "vmax.f32   q4, q4, q0                @ for relu6\n"
            "vmax.f32   q5, q5, q0                @ for relu6\n"
            "vmax.f32   q6, q6, q0                @ for relu6\n"
            "vmax.f32   q7, q7, q0                @ for relu6\n"
            "vmax.f32   q8, q8, q0                @ for relu6\n"
            "vmax.f32   q9, q9, q0                @ for relu6\n"
            "vld1.f32   {d2-d3}, [%[alpha]]       @ load relu6 alpha\n"
            "vmax.f32   q10, q10, q0              @ for relu6\n"
            "vmax.f32   q11, q11, q0              @ for relu6\n"
            "vmax.f32   q12, q12, q0              @ for relu6\n"
            "vmax.f32   q13, q13, q0              @ for relu6\n"
            "vmax.f32   q14, q14, q0              @ for relu6\n"
            "vmax.f32   q15, q15, q0              @ for relu6\n"

            "vmin.f32   q4, q4, q1                @ for relu6\n"
            "vmin.f32   q5, q5, q1                @ for relu6\n"
            "vmin.f32   q6, q6, q1                @ for relu6\n"
            "vmin.f32   q7, q7, q1                @ for relu6\n"
            "vmin.f32   q8, q8, q1                @ for relu6\n"
            "vmin.f32   q9, q9, q1                @ for relu6\n"
            "vmin.f32   q10, q10, q1              @ for relu6\n"
            "vmin.f32   q11, q11, q1              @ for relu6\n"
            "vmin.f32   q12, q12, q1              @ for relu6\n"
            "vmin.f32   q13, q13, q1              @ for relu6\n"
            "vmin.f32   q14, q14, q1              @ for relu6\n"
            "vmin.f32   q15, q15, q1              @ for relu6\n"
            "b          10f                       @ relu6 end \n"
            //! leakey relu
            "7:                                   @ otherwise is leakey relu\n"
            "vmov.u32   q0,   #0                  @ for leakey relu \n"
            "vld1.f32   {d2-d3}, [%[alpha]]       @ load leakey relu alpha\n"
            "vcge.f32   q2, q4, q0                @ vcgeq_u32 \n"
            "vmul.f32   q3, q4, q1                @ vmulq_f32 \n"
            "vbif       q4, q3, q2                @ choose    \n"
            "vcge.f32   q2, q5, q0                @ vcgeq_u32 \n"
            "vmul.f32   q3, q5, q1                @ vmulq_f32 \n"
            "vbif       q5, q3, q2                @ choose    \n"
            "vcge.f32   q2, q6, q0                @ vcgeq_u32 \n"
            "vmul.f32   q3, q6, q1                @ vmulq_f32 \n"
            "vbif       q6, q3, q2                @ choose    \n"
            "vcge.f32   q2, q7, q0                @ vcgeq_u32 \n"
            "vmul.f32   q3, q7, q1                @ vmulq_f32 \n"
            "vbif       q7, q3, q2                @ choose    \n"
            "vcge.f32   q2, q8, q0                @ vcgeq_u32 \n"
            "vmul.f32   q3, q8, q1                @ vmulq_f32 \n"
            "vbif       q8, q3, q2                @ choose    \n"
            "vcge.f32   q2, q9, q0                @ vcgeq_u32 \n"
            "vmul.f32   q3, q9, q1                @ vmulq_f32 \n"
            "vbif       q9, q3, q2                @ choose    \n"
            "vcge.f32   q2, q10, q0               @ vcgeq_u32 \n"
            "vmul.f32   q3, q10, q1               @ vmulq_f32 \n"
            "vbif       q10, q3, q2               @ choose    \n"
            "vcge.f32   q2, q11, q0               @ vcgeq_u32 \n"
            "vmul.f32   q3, q11, q1               @ vmulq_f32 \n"
            "vbif       q11, q3, q2               @ choose    \n"
            "vcge.f32   q2, q12, q0               @ vcgeq_u32 \n"
            "vmul.f32   q3, q12, q1               @ vmulq_f32 \n"
            "vbif       q12, q3, q2               @ choose    \n"
            "vcge.f32   q2, q13, q0               @ vcgeq_u32 \n"
            "vmul.f32   q3, q13, q1               @ vmulq_f32 \n"
            "vbif       q13, q3, q2               @ choose    \n"
            "vcge.f32   q2, q14, q0               @ vcgeq_u32 \n"
            "vmul.f32   q3, q14, q1               @ vmulq_f32 \n"
            "vbif       q14, q3, q2               @ choose    \n"
            "vcge.f32   q2, q15, q0               @ vcgeq_u32 \n"
            "vmul.f32   q3, q15, q1               @ vmulq_f32 \n"
            "vbif       q15, q3, q2               @ choose    \n"
            "10:                                  @ act end  \n"
            "vst1.32    {d8-d11},   [%[c_ptr0]]!    @ store r0\n"
            "vst1.32    {d12-d15},  [%[c_ptr1]]!    @ store r1\n"
            "vst1.32    {d16-d19},  [%[c_ptr2]]!    @ store r2\n"
            "vst1.32    {d20-d23},  [%[c_ptr3]]!    @ store r3\n"
            "vst1.32    {d24-d27},  [%[c_ptr4]]!    @ store r4\n"
            "vst1.32    {d28-d31},  [%[c_ptr5]]!    @ store r5\n"
            : [a_ptr] "+r"(a_ptr),
              [b_ptr] "+r"(b_ptr),
              [c_ptr0] "+r"(c_ptr0),
              [c_ptr1] "+r"(c_ptr1),
              [c_ptr2] "+r"(c_ptr2),
              [c_ptr3] "+r"(c_ptr3),
              [c_ptr4] "+r"(c_ptr4),
              [c_ptr5] "+r"(c_ptr5),
              [k] "+r"(k),
              [tails] "+r"(tails)
            : [bias_ptr] "r"(bias_local),
              [beta] "r"(beta),
              [alpha] "r" (alpha)
            : "q0","q1","q2","q3","q4",
              "q5","q6","q7","q8","q9","q10","q11",
              "q12","q13","q14","q15","cc","memory");
        // clang-format on

        if (flag_p_remain && (xb == bblocks - 1)) {
          for (int i = 0; i < remain; ++i) {
            *pout0++ = cout0[i];
            *pout1++ = cout1[i];
            *pout2++ = cout2[i];
            *pout3++ = cout3[i];
            *pout4++ = cout4[i];
            *pout5++ = cout5[i];
          }
        }
      }
    }
  }
}

/**
 * \brief gemm with ablock = 6, bblock = 8, output 6x8, optimize for a53 arch
 * @param A
 * @param B
 * @param C
 * @param M
 * @param N
 * @param K
 * @param threads
 * @param workspace
 */
void sgemm_prepacked_6x8_a53(bool is_transB,
                             int M,
                             int N,
                             int K,
                             const float* A_packed,
                             const float* B,
                             int ldb,
                             float* C,
                             int ldc,
                             const float* bias,
                             bool has_bias,
                             int is_relu,
                             ARMContext* ctx) {
  size_t l2_cache = ctx->llc_size() > 0 ? ctx->llc_size() : 512 * 1024;
  auto* workspace = ctx->workspace_data<float>();
  int threads = ctx->threads();
  //! MBLOCK * x (result) + MBLOCK * k (A) + x * k (B) = l2
  int x_block =
      (l2_cache - (MBLOCK_OTH * K)) / (sizeof(float) * (K + MBLOCK_OTH));
  x_block /= NBLOCK;
  x_block *= NBLOCK;
  int x_num = (N + (x_block - 1)) / x_block;
  x_block = (N + x_num - 1) / x_num;
  x_block = (x_block + NBLOCK - 1) / NBLOCK;
  x_block *= NBLOCK;
  x_block = x_block < NBLOCK ? NBLOCK : x_block;

  int k_pre = ((K + KBLOCK - 1) / KBLOCK) - 1;
  int tail_pre = (K & (KBLOCK - 1));
  if (tail_pre == 0) {
    tail_pre = KBLOCK;
  }

  //! merge tail_pre and flag_act
  tail_pre = (tail_pre << 2 | is_relu);
  bool flag_p_remain = false;
  int remain = 0;

  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    if (xmax > N) {
      xmax = N;
    }
    int bblocks = (xmax - x0 + NBLOCK - 1) / NBLOCK;
    remain = xmax - x0 - (bblocks - 1) * NBLOCK;
    if (remain > 0) {
      flag_p_remain = true;
    }
    //! load bpanel
    auto b_pannel = static_cast<float*>(workspace);
    if (is_transB) {
      loadb_trans(b_pannel, B, ldb, 0, K, x0, xmax);
    } else {
      loadb(b_pannel, B, ldb, 0, K, x0, xmax);
    }
#pragma omp parallel for num_threads(threads)
    for (unsigned int y = 0; y < M; y += MBLOCK_OTH) {
      unsigned int ymax = y + MBLOCK_OTH;
      if (ymax > M) {
        ymax = M;
      }
      float* c_ptr0 = C + y * ldc + x0;
      float* c_ptr1 = c_ptr0 + ldc;
      float* c_ptr2 = c_ptr1 + ldc;
      float* c_ptr3 = c_ptr2 + ldc;
      float* c_ptr4 = c_ptr3 + ldc;
      float* c_ptr5 = c_ptr4 + ldc;

      float* pout0 = c_ptr0;
      float* pout1 = c_ptr1;
      float* pout2 = c_ptr2;
      float* pout3 = c_ptr3;
      float* pout4 = c_ptr4;
      float* pout5 = c_ptr5;

      float bias_local[6] = {0};
      if (has_bias) {
        bias_local[0] = bias[y];
        bias_local[1] = bias[y + 1];
        bias_local[2] = bias[y + 2];
        bias_local[3] = bias[y + 3];
        bias_local[4] = bias[y + 4];
        bias_local[5] = bias[y + 5];
      }

      float cout0[NBLOCK];
      float cout1[NBLOCK];
      float cout2[NBLOCK];
      float cout3[NBLOCK];
      float cout4[NBLOCK];
      float cout5[NBLOCK];

      const float* a_ptr_l = A_packed + y * K;
      const float* b_ptr = b_pannel;
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
          pout0 = c_ptr0;
          pout1 = c_ptr1;
          pout2 = c_ptr2;
          pout3 = c_ptr3;
          pout4 = c_ptr4;
          pout5 = c_ptr5;

          c_ptr0 = cout0;
          c_ptr1 = cout1;
          c_ptr2 = cout2;
          c_ptr3 = cout3;
          c_ptr4 = cout4;
          c_ptr5 = cout5;
        }
        const float* a_ptr = a_ptr_l;
        int tails = tail_pre;
        int k = k_pre;

        // clang-format off
        asm volatile(
            // sgemm 6x8 for a53
            "vld1.32  {d2-d4},  [%[bias_ptr]]   \n"   /* load bias0-5 to d2-d4 */
            "pld [%[a_ptr]]                         @ preload a\n"
            "vdup.i32   q12,d4[0]                   @ out40=0\n"
            "pld [%[b_ptr]]                         @ preload b\n"
            "vdup.i32   q13,d4[0]                   @ out41=0\n"
            "pld [%[a_ptr], #64]                    @ preload a\n"
            "vdup.i32   q14,d4[1]                   @ out50=0\n"
            "pld [%[b_ptr], #64]                    @ preload b\n"
            "vdup.i32   q15,d4[1]                   @ out51=0\n"
            "pld [%[a_ptr], #128]                   @ preload a\n"
            "vdup.i32 q4, d2[0]                 \n"   /*  set out00 to bias0   */
            "vld1.32	{d0-d1},  [%[a_ptr] :64]  \n"   /* load a00-a30 to d0,d1 */
            "vdup.i32	q5, d2[0]                 \n"   /*  set out01 to bias0   */
            "vld1.32	{d4-d5},  [%[b_ptr] :128] \n"   /* load b00-b03 to d4,d5 */
            "vdup.i32	q6, d2[1]                 \n"   /*  set out10 to bias1   */
            "ldr  r0, [%[a_ptr], #0x10]         \n"   /*    load a40 to r0     */
            "vdup.i32	q7, d2[1]                 \n"   /*  set out11 to bias1   */
            "ldr  r1, [%[a_ptr], #0x14]         \n"   /*    load a50 to r1     */
            "vdup.i32	q8, d3[0]                 \n"   /*  set out20 to bias2   */
            "pld [%[a_ptr], #0x80]              \n"   /*    pre load apanel    */
            "vdup.i32	q9, d3[0]                 \n"   /*  set out21 to bias2   */
            "pld [%[b_ptr], #0x80]              \n"   /*    pre load bpanel    */
            "vdup.i32	q10, d3[1]                \n"   /*  set out30 to bias3   */
            "pld [%[a_ptr], #0xc0]              \n"   /*    pre load apanel    */
            "vdup.i32	q11, d3[1]                \n"   /*  set out31 to bias3   */
            "pld [%[b_ptr], #0xc0]              \n"   /*    pre load bpanel    */
            "cmp  %[k], #0                      \n"   /*      check k loop     */
            "beq  6f                            \n"   /*   k==0, branch to 6   */
            "1:\n"
            /* Unroll 0 */
            "vldr d6, [%[b_ptr], #0x10]         \n"   /*  load b04, b05 to d6  */
            "vmov d2, r0, r1                    \n"   /*   mov a40, a50 to d2  */
            "vmla.f32	q4, q2, d0[0]             \n"   /*   out00 += a00 * b0l  */
            "ldr  r0, [%[b_ptr], #0x18]         \n"   /*    load b06 to r0     */
            "vmla.f32	q6, q2, d0[1]             \n"   /*   out10 += a10 * b0l  */
            "ldr  r1, [%[b_ptr], #0x1C]         \n"   /*    load b07 to r1     */
            "vmla.f32	q8, q2, d1[0]             \n"   /*   out20 += a20 * b0l  */
            "vldr d3, [%[a_ptr], #0x18]         \n"   /*  load a01, a11 to d3  */
            "vmov d7, r0, r1                    \n"   /*   mov b06, b07 to d7  */
            "vmla.f32	q10, q2, d1[1]            \n"   /*   out30 += a30 * b0l  */
            "pld [%[a_ptr], #0x100]             \n"   /*    pre load apanel    */
            "vmla.f32	q12, q2, d2[0]            \n"   /*   out40 += a40 * b0l  */
            "vmla.f32	q14, q2, d2[1]            \n"   /*   out50 += a50 * b0l  */
            "vldr d4, [%[b_ptr], #0x20]         \n"   /*  load b10, b11 to d4  */
            "vmla.f32	q5, q3, d0[0]             \n"   /*   out01 += a00 * b0h  */
            "ldr  r0, [%[b_ptr], #0x28]         \n"   /*    load b12 to r0     */
            "vmla.f32	q7, q3, d0[1]             \n"   /*   out11 += a10 * b0h  */
            "ldr  r1, [%[b_ptr], #0x2C]         \n"   /*    load b13 to r1     */
            "vmla.f32	q9, q3, d1[0]             \n"   /*   out21 += a20 * b0h  */
            "vldr d0, [%[a_ptr], #0x20]         \n"   /*  load a21, a31 to d0  */
            "vmov d5, r0, r1                    \n"   /*   mov b12, b13 to d5  */
            "vmla.f32	q11, q3, d1[1]            \n"   /*   out31 += a30 * b0h  */
            "ldr  r0, [%[a_ptr], #0x28]         \n"   /*    load a41 to r0     */
            "vmla.f32	q13, q3, d2[0]            \n"   /*   out41 += a40 * b0h  */
            "ldr  r1, [%[a_ptr], #0x2C]         \n"   /*    load a51 to r1     */
            "vmla.f32	q15, q3, d2[1]            \n"   /*   out51 += a50 * b0h  */
            /* Unroll 1 */
            "vldr d6, [%[b_ptr], #0x30]         \n"   /*  load b14, b15 to d6  */
            "vmov d1, r0, r1                    \n"   /*   mov a41, a51 to d1  */
            "vmla.f32	q4, q2, d3[0]             \n"   /*   out00 += a01 * b1l  */
            "ldr  r0, [%[b_ptr], #0x38]         \n"   /*    load b16 to r0     */
            "vmla.f32	q6, q2, d3[1]             \n"   /*   out10 += a11 * b1l  */
            "ldr  r1, [%[b_ptr], #0x3C]         \n"   /*    load b17 to r1     */
            "vmla.f32	q8, q2, d0[0]             \n"   /*   out20 += a21 * b1l  */
            "vldr d2, [%[a_ptr], #0x30]         \n"   /*  load a02, a12 to d0  */
            "vmov d7, r0, r1                    \n"   /*   mov b16, b17 to d7  */
            "vmla.f32	q10, q2, d0[1]            \n"   /*   out30 += a31 * b1l  */
            "pld [%[b_ptr], #0x100]             \n"   /*    pre load apanel    */
            "vmla.f32	q12, q2, d1[0]            \n"   /*   out40 += a41 * b1l  */
            "vmla.f32	q14, q2, d1[1]            \n"   /*   out50 += a51 * b1l  */
            "vldr d4, [%[b_ptr], #0x40]         \n"   /*  load b20, b21 to d4  */
            "vmla.f32	q5, q3, d3[0]             \n"   /*   out01 += a01 * b1h  */
            "ldr  r0, [%[b_ptr], #0x48]         \n"   /*    load b22 to r0     */
            "vmla.f32	q7, q3, d3[1]             \n"   /*   out11 += a11 * b1h  */
            "ldr  r1, [%[b_ptr], #0x4C]         \n"   /*    load b23 to r1     */
            "vmla.f32	q9, q3, d0[0]             \n"   /*   out21 += a21 * b1h  */
            "vldr d3, [%[a_ptr], #0x38]         \n"   /*  load a22, a32 to d3  */
            "vmov d5, r0, r1                    \n"   /*   mov b22, b23 to d5  */
            "vmla.f32	q11, q3, d0[1]            \n"   /*   out31 += a31 * b1h  */
            "ldr  r0, [%[a_ptr], #0x40]         \n"   /*    load a42 to r0     */
            "vmla.f32	q13, q3, d1[0]            \n"   /*   out41 += a41 * b1h  */
            "ldr  r1, [%[a_ptr], #0x44]         \n"   /*    load a52 to r1     */
            "vmla.f32	q15, q3, d1[1]            \n"   /*   out51 += a51 * b1h  */
            /* Unroll 2 */
            "vldr d6, [%[b_ptr], #0x50]         \n"   /*  load b24, b25 to d6  */
            "vmov d0, r0, r1                    \n"   /*   mov a42, a52 to d0  */
            "vmla.f32	q4, q2, d2[0]             \n"   /*   out00 += a02 * b2l  */
            "ldr  r0, [%[b_ptr], #0x58]         \n"   /*    load b26 to r0     */
            "vmla.f32	q6, q2, d2[1]             \n"   /*   out10 += a12 * b2l  */
            "ldr  r1, [%[b_ptr], #0x5C]         \n"   /*    load b27 to r1     */
            "vmla.f32	q8, q2, d3[0]             \n"   /*   out20 += a22 * b2l  */
            "vldr d1, [%[a_ptr], #0x48]         \n"   /*  load a03, a13 to d1  */
            "vmov d7, r0, r1                    \n"   /*   mov b26, b27 to d7  */
            "vmla.f32	q10, q2, d3[1]            \n"   /*   out30 += a32 * b2l  */
            "pld [%[a_ptr], #0x140]             \n"   /*    pre load apanel    */
            "vmla.f32	q12, q2, d0[0]            \n"   /*   out40 += a42 * b2l  */
            "vmla.f32	q14, q2, d0[1]            \n"   /*   out50 += a52 * b2l  */
            "vldr d4, [%[b_ptr], #0x60]         \n"   /*  load b30, b31 to d4  */
            "vmla.f32	q5, q3, d2[0]             \n"   /*   out01 += a02 * b2h  */
            "ldr  r0, [%[b_ptr], #0x68]         \n"   /*    load b32 to r0     */
            "vmla.f32	q7, q3, d2[1]             \n"   /*   out11 += a12 * b2h  */
            "ldr  r1, [%[b_ptr], #0x6C]         \n"   /*    load b33 to r1     */
            "vmla.f32	q9, q3, d3[0]             \n"   /*   out21 += a22 * b2h  */
            "vldr d2, [%[a_ptr], #0x50]         \n"   /*  load a23, a33 to d2  */
            "vmov d5, r0, r1                    \n"   /*   mov b32, b33 to d5  */
            "vmla.f32	q11, q3, d3[1]            \n"   /*   out31 += a32 * b2h  */
            "ldr  r0, [%[a_ptr], #0x58]         \n"   /*    load a43 to r0     */
            "vmla.f32	q13, q3, d0[0]            \n"   /*   out41 += a42 * b2h  */
            "ldr  r1, [%[a_ptr], #0x5C]         \n"   /*    load a53 to r1     */
            "vmla.f32	q15, q3, d0[1]            \n"   /*   out51 += a52 * b2h  */
            "add  %[a_ptr], %[a_ptr], #0x60     \n"   /*      aptr += 96       */
            /* Unroll 3 */
            "vldr d6, [%[b_ptr], #0x70]         \n"   /*  load b34, b35 to d6  */
            "vmov d3, r0, r1                    \n"   /*   mov a43, a53 to d3  */
            "vmla.f32	q4, q2, d1[0]             \n"   /*   out00 += a03 * b3l  */
            "ldr  r0, [%[b_ptr], #0x78]         \n"   /*    load b36 to r0     */
            "vmla.f32	q6, q2, d1[1]             \n"   /*   out10 += a13 * b3l  */
            "ldr  r1, [%[b_ptr], #0x7C]         \n"   /*    load b37 to r1     */
            "vmla.f32	q8, q2, d2[0]             \n"   /*   out20 += a23 * b3l  */
            "add  %[b_ptr], %[b_ptr], #0x80     \n"   /*      bptr += 108      */
            "vldr d0, [%[a_ptr], #0x00]         \n"   /*  load a00, a10 to d0  */
            "vmov d7, r0, r1                    \n"   /*   mov b36, b37 to d7  */
            "vmla.f32	q10, q2, d2[1]            \n"   /*   out30 += a33 * b3l  */
            "pld [%[b_ptr], #0xC0]              \n"   /*    pre load bpanel    */
            "vmla.f32	q12, q2, d3[0]            \n"   /*   out40 += a43 * b3l  */
            "vmla.f32	q14, q2, d3[1]            \n"   /*   out50 += a53 * b3l  */
            "vldr d4, [%[b_ptr], #0x00]         \n"   /*  load b00, b01 to d4  */
            "vmla.f32	q5, q3, d1[0]             \n"   /*   out01 += a03 * b3h  */
            "ldr  r0, [%[b_ptr], #0x08]         \n"   /*    load b02 to r0     */
            "vmla.f32	q7, q3, d1[1]             \n"   /*   out11 += a13 * b3h  */
            "ldr  r1, [%[b_ptr], #0x0C]         \n"   /*    load b03 to r1     */
            "vmla.f32	q9, q3, d2[0]             \n"   /*   out21 += a23 * b3h  */
            "subs %[k], %[k], #1                \n"   /*      loop k -= 1      */
            "vldr d1, [%[a_ptr], #0x08]         \n"   /*  load a20, a30 to d1  */
            "vmov d5, r0, r1                    \n"   /*   mov b02, b03 to d5  */
            "vmla.f32	q11, q3, d2[1]            \n"   /*   out31 += a33 * b3h  */
            "ldr  r0, [%[a_ptr], #0x10]         \n"   /*    load a40 to r0     */
            "vmla.f32	q13, q3, d3[0]            \n"   /*   out41 += a43 * b3h  */
            "ldr  r1, [%[a_ptr], #0x14]         \n"   /*    load a50 to r1     */
            "vmla.f32	q15, q3, d3[1]            \n"   /*   out51 += a53 * b3h  */
            "bne  1b                            \n"   /*    branch to k loop   */
            "6:\n"
            "sub  %[tails], %[tails], #4        \n"   /*       tail -= 4       */
            "cmp  %[tails], #4                  \n"   /*     cmp tail with 4   */
            "blt  3f                            \n"   /*   branch to tail == 1 */
            /* Tail Unroll 0 */
            "vmov d2, r0, r1                    \n"   /*   mov b02, b03 to d2  */
            "add  %[a_ptr], %[a_ptr], #0x18     \n"   /*      aptr +=  24      */
            "vmla.f32	q4, q2, d0[0]             \n"   /*   out00 += a00 * b0l  */
            "vld1.32  {d3}, [%[a_ptr] :64]!     \n"   /*  load a01, a11 to d3  */
            "vmla.f32	q6, q2, d0[1]             \n"   /*   out10 += a10 * b0l  */
            "add  %[b_ptr], %[b_ptr], #0x10     \n"   /*      bptr +=  16      */
            "vmla.f32	q8, q2, d1[0]             \n"   /*   out20 += a20 * b0l  */
            "vld1.32	{d6-d7}, [%[b_ptr] :128]! \n"   /* load b04-b07 to d6,d7 */
            "vmla.f32	q10, q2, d1[1]            \n"   /*   out30 += a30 * b0l  */
            "vmla.f32	q12, q2, d2[0]            \n"   /*   out40 += a40 * b0l  */
            "sub  %[tails], %[tails], #4        \n"   /*       tail -= 4       */
            "vmla.f32	q14, q2, d2[1]            \n"   /*   out50 += a50 * b0l  */
            "vld1.32	{d4-d5}, [%[b_ptr] :128]! \n"   /* load b10-b13 to d4,d5 */
            "vmla.f32	q5, q3, d0[0]             \n"   /*   out01 += a00 * b0h  */
            "vmla.f32	q7, q3, d0[1]             \n"   /*   out11 += a10 * b0h  */
            "vmla.f32	q9, q3, d1[0]             \n"   /*   out21 += a20 * b0h  */
            "vmla.f32	q11, q3, d1[1]            \n"   /*   out31 += a30 * b0h  */
            "vld1.32	{d0-d1}, [%[a_ptr] :64]!  \n"   /* load a21-a51 to d0,d1 */
            "cmp  %[tails], #4                  \n"   /*    cmp tail with 4    */
            "vmla.f32	q13, q3, d2[0]            \n"   /*   out41 += a40 * b0h  */
            "vmla.f32	q15, q3, d2[1]            \n"   /*   out51 += a50 * b0h  */
            "vld1.32	{d6-d7}, [%[b_ptr] :128]! \n"   /* load b14-b17 to d6,d7 */
            "blt  4f                            \n"   /*   branch to tail == 2 */
            /* Tail Unroll 1 */
            "vmla.f32	q4, q2, d3[0]             \n"   /*   out00 += a01 * b1l  */
            "vmla.f32	q6, q2, d3[1]             \n"   /*   out10 += a11 * b1l  */
            "sub  %[tails], %[tails], #4        \n"   /*       tail -= 4       */
            "vmla.f32	q8, q2, d0[0]             \n"   /*   out20 += a21 * b1l  */
            "vmla.f32	q10, q2, d0[1]            \n"   /*   out30 += a31 * b1l  */
            "vmla.f32	q12, q2, d1[0]            \n"   /*   out40 += a41 * b1l  */
            "vmla.f32	q14, q2, d1[1]            \n"   /*   out50 += a51 * b1l  */
            "vld1.32	{d4-d5}, [%[b_ptr] :128]! \n"   /* load b20-b23 to d4,d5 */
            "vmla.f32	q5, q3, d3[0]             \n"   /*   out01 += a01 * b1h  */
            "vmla.f32	q7, q3, d3[1]             \n"   /*   out11 += a11 * b1h  */
            "cmp  %[tails], #4                  \n"   /*    cmp tail with 4    */
            "vld1.32	{d2-d3}, [%[a_ptr] :64]!  \n"   /* load a02-a32 to d2,d3 */
            "vmla.f32	q9, q3, d0[0]             \n"   /*   out21 += a21 * b1h  */
            "vmla.f32	q11, q3, d0[1]            \n"   /*   out31 += a31 * b1h  */
            "vmla.f32	q13, q3, d1[0]            \n"   /*   out41 += a41 * b1h  */
            "vmla.f32	q15, q3, d1[1]            \n"   /*   out51 += a51 * b1h  */
            "vld1.32	{d6-d7}, [%[b_ptr] :128]! \n"   /* load b24-b27 to d6,d7 */
            "blt  5f                            \n"   /*   branch to tail == 3 */
            /* Tail Unroll 2 */
            "sub  %[tails], %[tails], #4        \n"   /*       tail -= 4       */
            "vld1.32	{d0-d1}, [%[a_ptr] :64]!  \n"   /* a42a52a03a13 to d0,d1 */
            "vmla.f32	q4, q2, d2[0]             \n"   /*   out00 += a02 * b2l  */
            "vmla.f32	q6, q2, d2[1]             \n"   /*   out10 += a12 * b2l  */
            "vmla.f32	q8, q2, d3[0]             \n"   /*   out20 += a22 * b2l  */
            "vmla.f32	q10, q2, d3[1]            \n"   /*   out30 += a32 * b2l  */
            "vmla.f32	q12, q2, d0[0]            \n"   /*   out40 += a42 * b2l  */
            "vmla.f32	q14, q2, d0[1]            \n"   /*   out50 += a52 * b2l  */
            "vld1.32	{d4-d5}, [%[b_ptr] :128]! \n"   /* load b30-b33 to d4,d5 */
            "vmla.f32	q5, q3, d2[0]             \n"   /*   out01 += a02 * b2h  */
            "vmla.f32	q7, q3, d2[1]             \n"   /*   out11 += a12 * b2h  */
            "vmla.f32	q9, q3, d3[0]             \n"   /*   out21 += a22 * b2h  */
            "vmla.f32	q11, q3, d3[1]            \n"   /*   out31 += a32 * b2h  */
            "vld1.32	{d2-d3}, [%[a_ptr] :64]!  \n"   /* load a23-a53 to d2,d3 */
            "vmla.f32	q13, q3, d0[0]            \n"   /*   out41 += a42 * b2h  */
            "vmla.f32	q15, q3, d0[1]            \n"   /*   out51 += a52 * b2h  */
            "vld1.32	{d6-d7}, [%[b_ptr] :128]! \n"   /* load b34-b37 to d6,d7 */
            /* Tail Unroll 3 */
            "vmla.f32	q4,  q2, d1[0]            \n"   /*   out00 += a03 * b3l  */
            "vmla.f32	q5,  q3, d1[0]            \n"   /*   out01 += a03 * b3h  */
            "vmla.f32	q6,  q2, d1[1]            \n"   /*   out10 += a13 * b3l  */
            "vmla.f32	q7,  q3, d1[1]            \n"   /*   out11 += a13 * b3h  */
            "vmla.f32	q8,  q2, d2[0]            \n"   /*   out20 += a23 * b3l  */
            "vmla.f32	q9,  q3, d2[0]            \n"   /*   out21 += a23 * b3h  */
            "vmla.f32	q10, q2, d2[1]            \n"   /*   out30 += a33 * b3l  */
            "vmla.f32	q11, q3, d2[1]            \n"   /*   out31 += a33 * b3h  */
            "vmla.f32	q12, q2, d3[0]            \n"   /*   out40 += a43 * b3l  */
            "vmla.f32	q13, q3, d3[0]            \n"   /*   out41 += a43 * b3h  */
            "vmla.f32	q14, q2, d3[1]            \n"   /*   out50 += a53 * b3l  */
            "vmla.f32	q15, q3, d3[1]            \n"   /*   out51 += a53 * b3h  */
            "b  2f                              \n"   /*  branch to check relu */
            /* tails==1 final tail */
            "3:\n"
            "vmov d2, r0, r1                    \n"   /*   mov b02, b03 to d2  */
            "add  %[b_ptr], %[b_ptr], #0x10     \n"   /*      bptr +=  16      */
            "vmla.f32	q4, q2, d0[0]             \n"   /*   out00 += a00 * b0l  */
            "add  %[a_ptr], %[a_ptr], #0x18     \n"   /*      aptr +=  24      */
            "vmla.f32	q6, q2, d0[1]             \n"   /*   out10 += a10 * b0l  */
            "vld1.32	{d6-d7}, [%[b_ptr] :128]! \n"   /* load b04-b07 to d6,d7 */
            "vmla.f32	q8,  q2, d1[0]            \n"   /*   out20 += a20 * b0l  */
            "vmla.f32	q10, q2, d1[1]            \n"   /*   out30 += a30 * b0l  */
            "vmla.f32	q12, q2, d2[0]            \n"   /*   out40 += a40 * b0l  */
            "vmla.f32	q14, q2, d2[1]            \n"   /*   out50 += a50 * b0l  */
            "vmla.f32	q5,  q3, d0[0]            \n"   /*   out01 += a00 * b0h  */
            "vmla.f32	q7,  q3, d0[1]            \n"   /*   out11 += a10 * b0h  */
            "vmla.f32	q9,  q3, d1[0]            \n"   /*   out21 += a20 * b0h  */
            "vmla.f32	q11, q3, d1[1]            \n"   /*   out31 += a30 * b0h  */
            "vmla.f32	q13, q3, d2[0]            \n"   /*   out41 += a40 * b0h  */
            "vmla.f32	q15, q3, d2[1]            \n"   /*   out51 += a50 * b0h  */
            "b  2f                              \n"   /*  branch to check relu */
            /* tails==2 final tail */
            "4:\n"
            "vmla.f32	q4,  q2, d3[0]            \n"   /*   out00 += a01 * b1l  */
            "vmla.f32	q5,  q3, d3[0]            \n"   /*   out01 += a01 * b1h  */
            "vmla.f32	q6,  q2, d3[1]            \n"   /*   out10 += a11 * b1l  */
            "vmla.f32	q7,  q3, d3[1]            \n"   /*   out11 += a11 * b1h  */
            "vmla.f32	q8,  q2, d0[0]            \n"   /*   out20 += a21 * b1l  */
            "vmla.f32	q9,  q3, d0[0]            \n"   /*   out21 += a21 * b1h  */
            "vmla.f32	q10, q2, d0[1]            \n"   /*   out30 += a31 * b1l  */
            "vmla.f32	q11, q3, d0[1]            \n"   /*   out31 += a31 * b1h  */
            "vmla.f32	q12, q2, d1[0]            \n"   /*   out40 += a41 * b1l  */
            "vmla.f32	q13, q3, d1[0]            \n"   /*   out41 += a41 * b1h  */
            "vmla.f32	q14, q2, d1[1]            \n"   /*   out50 += a51 * b1l  */
            "vmla.f32	q15, q3, d1[1]            \n"   /*   out51 += a51 * b1h  */
            "b  2f                              \n"   /*  branch to check relu */
            /* tails==3 final tail */
            "5:\n"
            "vmla.f32	q4,  q2, d2[0]            \n"   /*   out00 += a02 * b2l  */
            "vld1.32	{d0}, [%[a_ptr] :64]!     \n"   /*  load a42, a52 to d0  */
            "vmla.f32	q6,  q2, d2[1]            \n"   /*   out10 += a12 * b2l  */
            "vmla.f32	q8,  q2, d3[0]            \n"   /*   out20 += a22 * b2l  */
            "vmla.f32	q5,  q3, d2[0]            \n"   /*   out01 += a02 * b2h  */
            "vmla.f32	q7,  q3, d2[1]            \n"   /*   out11 += a12 * b2h  */
            "vmla.f32	q9,  q3, d3[0]            \n"   /*   out21 += a22 * b2h  */
            "vmla.f32	q10, q2, d3[1]            \n"   /*   out30 += a32 * b2l  */
            "vmla.f32	q11, q3, d3[1]            \n"   /*   out31 += a32 * b2h  */
            "vmla.f32	q12, q2, d0[0]            \n"   /*   out40 += a42 * b2l  */
            "vmla.f32	q13, q3, d0[0]            \n"   /*   out41 += a42 * b2h  */
            "vmla.f32	q14, q2, d0[1]            \n"   /*   out50 += a52 * b2l  */
            "vmla.f32	q15, q3, d0[1]            \n"   /*   out51 += a52 * b2h  */
            /* relu */
            "2:\n"
            "cmp %[tails], #1                   \n"   /*    cmp tail is relu   */
            "bne 0f                             \n"   /* no relu branch to end */
            "vmov.i32 q0,  #0                   \n"   /*     mov 0.f to q0     */
            "vmax.f32 q4,  q4,  q0              \n"   /*      out00 relu       */
            "vmax.f32 q5,  q5,  q0              \n"   /*      out01 relu       */
            "vmax.f32 q6,  q6,  q0              \n"   /*      out10 relu       */
            "vmax.f32 q7,  q7,  q0              \n"   /*      out11 relu       */
            "vmax.f32 q8,  q8,  q0              \n"   /*      out20 relu       */
            "vmax.f32 q9,  q9,  q0              \n"   /*      out21 relu       */
            "vmax.f32 q10, q10, q0              \n"   /*      out30 relu       */
            "vmax.f32 q11, q11, q0              \n"   /*      out31 relu       */
            "vmax.f32 q12, q12, q0              \n"   /*      out40 relu       */
            "vmax.f32 q13, q13, q0              \n"   /*      out41 relu       */
            "vmax.f32 q14, q14, q0              \n"   /*      out50 relu       */
            "vmax.f32 q15, q15, q0              \n"   /*      out51 relu       */
            "0:\n"
            "vst1.32  {d8-d11},   [%[c_ptr0]]!  \n"   /*  store out0 to cptr0  */
            "vst1.32  {d12-d15},  [%[c_ptr1]]!  \n"   /*  store out1 to cptr1  */
            "vst1.32  {d16-d19},  [%[c_ptr2]]!  \n"   /*  store out2 to cptr2  */
            "vst1.32  {d20-d23},  [%[c_ptr3]]!  \n"   /*  store out3 to cptr3  */
            "vst1.32  {d24-d27},  [%[c_ptr4]]!  \n"   /*  store out4 to cptr4  */
            "vst1.32  {d28-d31},  [%[c_ptr5]]!  \n"   /*  store out5 to cptr5  */
            : [a_ptr] "+r"(a_ptr),
              [b_ptr] "+r"(b_ptr),
              [c_ptr0] "+r"(c_ptr0),
              [c_ptr1] "+r"(c_ptr1),
              [c_ptr2] "+r"(c_ptr2),
              [c_ptr3] "+r"(c_ptr3),
              [c_ptr4] "+r"(c_ptr4),
              [c_ptr5] "+r"(c_ptr5),
              [k] "+r"(k),
              [tails] "+r"(tails)
            : [bias_ptr] "r"(bias_local)
            : "r0", "r1", "q0","q1","q2","q3","q4",
              "q5","q6","q7","q8","q9","q10","q11",
              "q12","q13","q14","q15","cc","memory");
        // clang-format on
        if (flag_p_remain && (xb == bblocks - 1)) {
          for (int i = 0; i < remain; ++i) {
            *pout0++ = cout0[i];
            *pout1++ = cout1[i];
            *pout2++ = cout2[i];
            *pout3++ = cout3[i];
            *pout4++ = cout4[i];
            *pout5++ = cout5[i];
          }
        }
      }
    }
  }
}

void sgemm_prepacked_4x8(bool is_transB,
                         int M,
                         int N,
                         int K,
                         const float* A_packed,
                         const float* B,
                         int ldb,
                         float beta,
                         float* C,
                         int ldc,
                         const float* bias,
                         bool has_bias,
                         const operators::ActivationParam act_param,
                         ARMContext* ctx) {
  size_t l2_cache = ctx->llc_size() > 0 ? ctx->llc_size() : 512 * 1024;
  auto* workspace = ctx->workspace_data<float>();
  int threads = ctx->threads();
  auto act_type = act_param.active_type;
  float alpha[4] = {0.f, 0.f, 0.f, 0.f};
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    }
  }
  //! MBLOCK * x (result) + MBLOCK * k (A) + x * k (B) = l2
  int x_block =
      (l2_cache - (MBLOCK_A73 * K)) / (sizeof(float) * (K + MBLOCK_A73));
  x_block /= NBLOCK;
  x_block *= NBLOCK;
  int x_num = (N + (x_block - 1)) / x_block;
  x_block = (N + x_num - 1) / x_num;
  x_block = (x_block + NBLOCK - 1) / NBLOCK;
  x_block *= NBLOCK;
  x_block = x_block < NBLOCK ? NBLOCK : x_block;

  int k_pre = ((K + KBLOCK - 1) / KBLOCK) - 1;
  int tail_pre = (K & (KBLOCK - 1));
  if (tail_pre == 0) {
    tail_pre = KBLOCK;
  }

  bool flag_p_remain = false;
  int remain = 0;

  int has_beta = fabsf(beta) > 1e-8f ? 1 : 0;

  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    if (xmax > N) {
      xmax = N;
    }
    int bblocks = (xmax - x0 + NBLOCK - 1) / NBLOCK;
    remain = xmax - x0 - (bblocks - 1) * NBLOCK;
    if (remain > 0) {
      flag_p_remain = true;
    }
    //! load bpanel
    auto b_pannel = static_cast<float*>(workspace);
    if (is_transB) {
      loadb_trans(b_pannel, B, ldb, 0, K, x0, xmax);
    } else {
      loadb(b_pannel, B, ldb, 0, K, x0, xmax);
    }
#pragma omp parallel for num_threads(threads)
    for (unsigned int y = 0; y < M; y += MBLOCK_A73) {
      unsigned int ymax = y + MBLOCK_A73;
      if (ymax > M) {
        ymax = M;
      }

      float cout0[NBLOCK];
      float cout1[NBLOCK];
      float cout2[NBLOCK];
      float cout3[NBLOCK];

      float bias_local[4] = {0};
      if (has_bias) {
        bias_local[0] = bias[y];
        bias_local[1] = bias[y + 1];
        bias_local[2] = bias[y + 2];
        bias_local[3] = bias[y + 3];
      }

      float* c_ptr0 = C + y * ldc + x0;
      float* c_ptr1 = c_ptr0 + ldc;
      float* c_ptr2 = c_ptr1 + ldc;
      float* c_ptr3 = c_ptr2 + ldc;

      float* pout0 = c_ptr0;
      float* pout1 = c_ptr1;
      float* pout2 = c_ptr2;
      float* pout3 = c_ptr3;

      const float* a_ptr_l = A_packed + y * K;
      const float* b_ptr = b_pannel;
      for (int xb = 0; xb < bblocks; xb++) {
        if ((y + 3) >= ymax) {
          switch ((y + 3) - ymax) {
            case 2:
              c_ptr1 = cout1;
            case 1:
              c_ptr2 = cout1;
            case 0:
              c_ptr3 = cout1;
            default:
              break;
          }
        }
        if (flag_p_remain && (xb == bblocks - 1)) {
          pout0 = c_ptr0;
          pout1 = c_ptr1;
          pout2 = c_ptr2;
          pout3 = c_ptr3;

          c_ptr0 = cout0;
          c_ptr1 = cout1;
          c_ptr2 = cout2;
          c_ptr3 = cout3;

          if (has_beta) {
            for (int i = 0; i < remain; ++i) {
              cout0[i] = pout0[i];
              cout1[i] = pout1[i];
              cout2[i] = pout2[i];
              cout3[i] = pout3[i];
            }
          }
        }
        const float* a_ptr = a_ptr_l;
        int tails = tail_pre;
        int k = k_pre;
        // clang-format off
        asm volatile(
            "vld1.32    {d4-d5}, [%[bias_ptr]]      @ load bias\n"
            "vdup.32    q8, d4[0]                   @ add bias to out00\n"
            "pld [%[a_ptr]]                         @ preload a, 64byte\n"
            "vdup.32    q9, d4[0]                   @ add bias to out01\n"
            "pld [%[b_ptr]]                         @ preload b\n"
            "vdup.32    q10, d4[1]                  @ add bias to out10\n"
            "pld [%[a_ptr], #64]                    @ preload a\n"
            "vdup.32    q11, d4[1]                  @ add bias to out11\n"
            "vdup.32    q12, d5[0]                  @ add bias to out20\n"
            "pld [%[b_ptr], #64]                    @ preload b\n"
            "vdup.32    q13, d5[0]                  @ add bias to out21\n"
            "pld [%[a_ptr], #128]                   @ preload a\n"
            "vdup.32    q14, d5[1]                  @ add bias to out30\n"
            "pld [%[b_ptr], #128]                   @ preload b\n"
            "vdup.32    q15, d5[1]                  @ add bias to out31\n"
            "pld [%[b_ptr], #192]                   @ preload b\n"
            "cmp %[beta], #0\n"
            "beq    11f\n" /* check beta == 0? */
            /* process beta */
            "vdup.32    q4, %[beta]\n"          /* beta to vector */
            "vld1.32    {d0-d3}, [%[c_ptr0]]\n" /* load output r0 */
            "vld1.32    {d4-d7}, [%[c_ptr1]]\n" /* load output r1 */
            "vmla.f32   q8, q0, q4\n"           /* cr00 += beta * c_r00 */
            "vmla.f32   q9, q1, q4\n"           /* cr01 += beta * c_r01 */
            "vld1.32    {d0-d3}, [%[c_ptr2]]\n" /* load output r2 */
            "vmla.f32   q10, q2, q4\n"          /* cr10 += beta * c_r10 */
            "vmla.f32   q11, q3, q4\n"          /* cr11 += beta * c_r11 */
            "vld1.32    {d4-d7}, [%[c_ptr3]]\n" /* load output r3 */
            "vmla.f32   q12, q0, q4\n"          /* cr20 += beta * c_r20 */
            "vmla.f32   q13, q1, q4\n"          /* cr21 += beta * c_r21 */
            "vmla.f32   q14, q2, q4\n"          /* cr30 += beta * c_r30 */
            "vmla.f32   q15, q3, q4\n"          /* cr31 += beta * c_r31 */
            "11: \n"                            /* check loop count */
            "vld1.32	{d0-d3}, [%[a_ptr] :128]!   @ load a0~a3\n"
            "vld1.32   {d8-d11}, [%[b_ptr] :128]! @ load b1\n"
            "cmp %[k], #0                         @ check weather k is bigger than "
            "0\n"
            "beq 0f                               @ jump to tail\n"
            "1:                                   @ main loop for k\n"
            /* Unroll 0*/
            "vld1.32  {d12-d15}, [%[b_ptr] :128]! @ load next b1, b2\n"
            "vmla.f32	q8, q4, d0[0]               @ out0 += b1 * a0\n"
            "vld1.32	{d4-d7}, [%[a_ptr] :128]!   @ load next 2xa0~a3\n"
            "vmla.f32	q10, q4, d0[1]              @ out1 += b1 * a1\n"
            "vmla.f32	q12, q4, d1[0]              @ out2 += b1 * a2\n"
            "vmla.f32	q14, q4, d1[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q9, q5, d0[0]               @ out4 += b2 * a0\n"
            "vmla.f32	q11, q5, d0[1]              @ out5 += b2 * a1\n"
            "vmla.f32	q13, q5, d1[0]              @ out6 += b2 * a2\n"
            "vmla.f32	q15, q5, d1[1]              @ out7 += b2 * a3\n"
            "vld1.32	{d8-d11}, [%[b_ptr] :128]!  @ load next b1, b2\n"
            /* Unroll 1 */
            "vmla.f32	q8, q6, d2[0]               @ out0 += b1 * a0\n"
            "pld [%[b_ptr], #64]                    @ preload b\n"
            "vmla.f32	q10, q6, d2[1]              @ out1 += b1 * a1\n"
            "vmla.f32	q12, q6, d3[0]              @ out2 += b1 * a2\n"
            "vmla.f32	q14, q6, d3[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q9, q7, d2[0]               @ out6 += b2 * a0\n"
            "vmla.f32	q11, q7, d2[1]              @ out7 += b2 * a1\n"
            "vmla.f32	q13, q7, d3[0]              @ out8 += b2 * a2\n"
            "vmla.f32	q15, q7, d3[1]              @ out9 += b2 * a3\n"
            "vld1.32	{d12-d15}, [%[b_ptr] :128]! @ load next b1,b2\n"
            /* Unroll 2 */
            "vmla.f32	q8, q4, d4[0]               @ out0 += b1 * a0\n"
            "vld1.32	{d0-d3}, [%[a_ptr] :128]!   @ load next a0~a3\n"
            "vmla.f32	q10, q4, d4[1]              @ out1 += b1 * a1\n"
            "vmla.f32	q12, q4, d5[0]              @ out2 += b1 * a2\n"
            "vmla.f32	q14, q4, d5[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q9, q5, d4[0]               @ out4 += b2 * a0\n"
            "vmla.f32	q11, q5, d4[1]              @ out5 += b2 * a1\n"
            "vmla.f32	q13, q5, d5[0]              @ out6 += b2 * a2\n"
            "vmla.f32	q15, q5, d5[1]              @ out7 += b2 * a3\n"
            "vld1.32	{d8-d11}, [%[b_ptr] :128]!  @ load next b1, b2\n"
            /* Unroll 3 */
            "vmla.f32	q8, q6, d6[0]               @ out0 += b1 * a0\n"
            "pld [%[a_ptr], #64]                    @ preload a\n"
            "vmla.f32	q10, q6, d6[1]              @ out1 += b1 * a1\n"
            "vmla.f32	q12, q6, d7[0]              @ out2 += b1 * a2\n"
            "vmla.f32	q14, q6, d7[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q9, q7, d6[0]               @ out4 += b2 * a0\n"
            "vmla.f32	q11, q7, d6[1]              @ out5 += b2 * a1\n"
            "vmla.f32	q13, q7, d7[0]              @ out6 += b2 * a2\n"
            "vmla.f32	q15, q7, d7[1]              @ out7 += b2 * a3\n"
            "subs		%[k], %[k], #1              @ k--\n"
            "bne		1b                          @ jump to main loop\n"
            "0:                                     @ process tail\n"
            "subs		%[tails], %[tails], #1      @ tail--\n"
            "beq		3f                          @ jump to tail = 1\n"
            /* Unroll 0*/
            "vld1.32  {d12-d15}, [%[b_ptr] :128]!   @ load next b1, b2\n"
            "vmla.f32	q8, q4, d0[0]               @ out0 += b1 * a0\n"
            "vmla.f32	q10, q4, d0[1]              @ out1 += b1 * a1\n"
            "subs		%[tails], %[tails], #1      @ tail--\n"
            "vmla.f32	q12, q4, d1[0]              @ out2 += b1 * a2\n"
            "vmla.f32	q14, q4, d1[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q9, q5, d0[0]               @ out4 += b2 * a0\n"
            "vmla.f32	q11, q5, d0[1]              @ out5 += b2 * a1\n"
            "vmla.f32	q13, q5, d1[0]              @ out6 += b2 * a2\n"
            "vmla.f32	q15, q5, d1[1]              @ out7 += b2 * a3\n"
            "beq		4f                          @ jump to tail==2\n"
            /* Unroll 1 */
            "vld1.32	{d8-d11}, [%[b_ptr] :128]!  @ load next b1, b2\n"
            "vmla.f32	q8, q6, d2[0]               @ out0 += b1 * a0\n"
            "vld1.32	{d4-d7}, [%[a_ptr] :128]!   @ load next 2xa0~a3\n"
            "vmla.f32	q10, q6, d2[1]              @ out1 += b1 * a1\n"
            "subs		%[tails], %[tails], #1      @ tail--\n"
            "vmla.f32	q12, q6, d3[0]              @ out2 += b1 * a2\n"
            "vmla.f32	q14, q6, d3[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q9, q7, d2[0]               @ out6 += b2 * a0\n"
            "vmla.f32	q11, q7, d2[1]              @ out7 += b2 * a1\n"
            "vmla.f32	q13, q7, d3[0]              @ out8 += b2 * a2\n"
            "vmla.f32	q15, q7, d3[1]              @ out9 += b2 * a3\n"
            "beq		5f                          @ jump to tail==3\n"
            /* Unroll 2 */
            "vld1.32	{d12-d15}, [%[b_ptr] :128]! @ load next b1,b2\n"
            "vmla.f32	q8, q4, d4[0]               @ out0 += b1 * a0\n"
            "vmla.f32	q10, q4, d4[1]              @ out1 += b1 * a1\n"
            "vmla.f32	q12, q4, d5[0]              @ out2 += b1 * a2\n"
            "vmla.f32	q14, q4, d5[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q9, q5, d4[0]               @ out4 += b2 * a0\n"
            "vmla.f32	q11, q5, d4[1]              @ out5 += b2 * a1\n"
            "vmla.f32	q13, q5, d5[0]              @ out6 += b2 * a2\n"
            "vmla.f32	q15, q5, d5[1]              @ out7 += b2 * a3\n"
            /* Unroll 3 */
            "vmla.f32	q8, q6, d6[0]               @ out0 += b1 * a0\n"
            "vmla.f32	q10, q6, d6[1]              @ out1 += b1 * a1\n"
            "vmla.f32	q12, q6, d7[0]              @ out2 += b1 * a2\n"
            "vmla.f32	q14, q6, d7[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q9, q7, d6[0]               @ out4 += b2 * a0\n"
            "vmla.f32	q11, q7, d6[1]              @ out5 += b2 * a1\n"
            "vmla.f32	q13, q7, d7[0]              @ out6 += b2 * a2\n"
            "vmla.f32	q15, q7, d7[1]              @ out7 += b2 * a3\n"
            "b		2f\n"
            /* tails==1 final tail */
            "3:                                     @ tail=1\n"
            "vmla.f32	q8, q4, d0[0]               @ out0 += b1 * a0\n"
            "vmla.f32	q10, q4, d0[1]              @ out1 += b1 * a1\n"
            "vmla.f32	q12, q4, d1[0]              @ out2 += b1 * a2\n"
            "vmla.f32	q14, q4, d1[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q9, q5, d0[0]               @ out4 += b2 * a0\n"
            "vmla.f32	q11, q5, d0[1]              @ out5 += b2 * a1\n"
            "vmla.f32	q13, q5, d1[0]              @ out6 += b2 * a2\n"
            "vmla.f32	q15, q5, d1[1]              @ out7 += b2 * a3\n"
            /*aptr - 16 */
            "sub		%[a_ptr], %[a_ptr], #16     @ tail--\n"
            "b		2f                              @ jump to end\n"
            /* tails==2 final tail*/
            "4:                                     @ tail == 2\n"
            "vmla.f32	q8, q6, d2[0]               @ out0 += b1 * a0\n"
            "vmla.f32	q10, q6, d2[1]              @ out1 += b1 * a1\n"
            "vmla.f32	q12, q6, d3[0]              @ out2 += b1 * a2\n"
            "vmla.f32	q14, q6, d3[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q9, q7, d2[0]               @ out4 += b2 * a0\n"
            "vmla.f32	q11, q7, d2[1]              @ out5 += b2 * a1\n"
            "vmla.f32	q13, q7, d3[0]              @ out6 += b2 * a2\n"
            "vmla.f32	q15, q7, d3[1]              @ out7 += b2 * a3\n"
            "b		2f                              @ jump to end\n"
            /* tails==3 final tail*/
            "5:                                     @ tail=3\n"
            "vmla.f32	q8, q4, d4[0]               @ out0 += b1 * a0\n"
            "vmla.f32	q10, q4, d4[1]              @ out1 += b1 * a1\n"
            "vmla.f32	q12, q4, d5[0]              @ out2 += b1 * a2\n"
            "vmla.f32	q14, q4, d5[1]              @ out3 += b1 * a3\n"
            "vmla.f32	q9, q5, d4[0]               @ out4 += b2 * a0\n"
            "vmla.f32	q11, q5, d4[1]              @ out5 += b2 * a1\n"
            "vmla.f32	q13, q5, d5[0]              @ out6 += b2 * a2\n"
            "vmla.f32	q15, q5, d5[1]              @ out7 += b2 * a3\n"
            /*aptr - 16*/
            "sub		%[a_ptr], %[a_ptr], #16       @ tail--\n"
            "2:                                   @ check relu\n"
            //!   relu
            "cmp        %[flag_act], #1           @ check if has relu\n"
            "bne        6f                        @ jump if not relu \n"
            "vmov.u32   q0, #0                    @ for relu\n"
            "vmax.f32   q8, q8, q0                @ for relu\n"
            "vmax.f32   q9, q9, q0                @ for relu\n"
            "vmax.f32   q10, q10, q0              @ for relu\n"
            "vmax.f32   q11, q11, q0              @ for relu\n"
            "vmax.f32   q12, q12, q0              @ for relu\n"
            "vmax.f32   q13, q13, q0              @ for relu\n"
            "vmax.f32   q14, q14, q0              @ for relu\n"
            "vmax.f32   q15, q15, q0              @ for relu\n"
            "b          10f                       @ relu end\n"
            "6:                                   @ no relu \n"
            "cmp        %[flag_act], #0           @ check no act\n"
            "beq        10f                       @ no act end  \n"
            //!   relu6
            "cmp        %[flag_act], #2           @ check if has relu6\n"
            "bne        7f                        @ jump if no relu6 \n"
            "vmov.u32   q0, #0                    @ for relu6\n"
            "vld1.f32   {d2-d3}, [%[alpha]]       @ load relu6 alpha\n"
            "vmax.f32   q8, q8, q0                @ for relu6\n"
            "vmax.f32   q9, q9, q0                @ for relu6\n"
            "vmax.f32   q10, q10, q0              @ for relu6\n"
            "vmax.f32   q11, q11, q0              @ for relu6\n"
            "vmax.f32   q12, q12, q0              @ for relu6\n"
            "vmax.f32   q13, q13, q0              @ for relu6\n"
            "vmax.f32   q14, q14, q0              @ for relu6\n"
            "vmax.f32   q15, q15, q0              @ for relu6\n"

            "vmin.f32   q8, q8, q1                @ for relu6\n"
            "vmin.f32   q9, q9, q1                @ for relu6\n"
            "vmin.f32   q10, q10, q1              @ for relu6\n"
            "vmin.f32   q11, q11, q1              @ for relu6\n"
            "vmin.f32   q12, q12, q1              @ for relu6\n"
            "vmin.f32   q13, q13, q1              @ for relu6\n"
            "vmin.f32   q14, q14, q1              @ for relu6\n"
            "vmin.f32   q15, q15, q1              @ for relu6\n"
            "b          10f                       @ relu6 end \n"
            //! leakey relu
            "7:                                   @ otherwise is leakey relu\n"
            "vmov.u32   q0,   #0                  @ for leakey relu \n"
            "vld1.f32   {d2-d3}, [%[alpha]]       @ load leakey relu alpha\n"
            "vcge.f32   q2, q8, q0                @ vcgeq_u32 \n"
            "vmul.f32   q3, q8, q1                @ vmulq_f32 \n"
            "vbif       q8, q3, q2                @ choose    \n"
            "vcge.f32   q2, q9, q0                @ vcgeq_u32 \n"
            "vmul.f32   q3, q9, q1                @ vmulq_f32 \n"
            "vbif       q9, q3, q2                @ choose    \n"
            "vcge.f32   q2, q10, q0               @ vcgeq_u32 \n"
            "vmul.f32   q3, q10, q1               @ vmulq_f32 \n"
            "vbif       q10, q3, q2               @ choose    \n"
            "vcge.f32   q2, q11, q0               @ vcgeq_u32 \n"
            "vmul.f32   q3, q11, q1               @ vmulq_f32 \n"
            "vbif       q11, q3, q2               @ choose    \n"
            "vcge.f32   q2, q12, q0               @ vcgeq_u32 \n"
            "vmul.f32   q3, q12, q1               @ vmulq_f32 \n"
            "vbif       q12, q3, q2               @ choose    \n"
            "vcge.f32   q2, q13, q0               @ vcgeq_u32 \n"
            "vmul.f32   q3, q13, q1               @ vmulq_f32 \n"
            "vbif       q13, q3, q2               @ choose    \n"
            "vcge.f32   q2, q14, q0               @ vcgeq_u32 \n"
            "vmul.f32   q3, q14, q1               @ vmulq_f32 \n"
            "vbif       q14, q3, q2               @ choose    \n"
            "vcge.f32   q2, q15, q0               @ vcgeq_u32 \n"
            "vmul.f32   q3, q15, q1               @ vmulq_f32 \n"
            "vbif       q15, q3, q2               @ choose    \n"
            "10:                                  @ act end  \n"
            "vst1.32    {d16-d19},  [%[c_ptr0]]!    @ store r0\n"
            "vst1.32    {d20-d23},  [%[c_ptr1]]!    @ store r1\n"
            "vst1.32    {d24-d27},  [%[c_ptr2]]!    @ store r2\n"
            "vst1.32    {d28-d31},  [%[c_ptr3]]!    @ store r3\n"
            : [a_ptr] "+r"(a_ptr),
              [b_ptr] "+r"(b_ptr),
              [c_ptr0] "+r"(c_ptr0),
              [c_ptr1] "+r"(c_ptr1),
              [c_ptr2] "+r"(c_ptr2),
              [c_ptr3] "+r"(c_ptr3),
              [k] "+r"(k),
              [tails] "+r"(tails)
            : [bias_ptr] "r"(bias_local),
              [beta] "r"(beta),
              [alpha] "r"(alpha),
              [flag_act] "r"(flag_act)
            : "q0","q1","q2","q3",
              "q4","q5","q6","q7","q8","q9","q10",
              "q11","q12","q13","q14","q15","cc","memory");
        // clang-format on
        if (flag_p_remain && (xb == bblocks - 1)) {
          for (int i = 0; i < remain; ++i) {
            *pout0++ = cout0[i];
            *pout1++ = cout1[i];
            *pout2++ = cout2[i];
            *pout3++ = cout3[i];
          }
        }
      }
    }
  }
}
#endif  // __aarch64__

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
