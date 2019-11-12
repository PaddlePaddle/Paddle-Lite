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

#include "lite/backends/arm/math/packed_sgemm_c4.h"
#include <arm_neon.h>

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void trans_mat_to_c4(const float* input,
                     float* output,
                     const int ldin,
                     const int M,
                     const int K,
                     bool pack_k) {
  const int m_round = (M + 3) / 4 * 4;
  int k_round = (K + 3) / 4 * 4;
  if (!pack_k) {
    k_round = K;
  }
  const int m_loop = m_round / 4;
  float zero_buf[K];
  memset(zero_buf, 0, K * sizeof(float));
  for (int i = 0; i < m_loop; ++i) {
    const float* in0 = input + i * 4 * ldin;
    const float* in1 = in0 + ldin;
    const float* in2 = in1 + ldin;
    const float* in3 = in2 + ldin;
    if (4 * (i + 1) - M > 0) {
      switch (4 * (i + 1) - M) {
        case 3:
          in1 = zero_buf;
        case 2:
          in2 = zero_buf;
        case 1:
          in3 = zero_buf;
        default:
          break;
      }
    }
    for (int j = 0; j < K; ++j) {
      *output++ = *in0++;
      *output++ = *in1++;
      *output++ = *in2++;
      *output++ = *in3++;
    }
    for (int j = K; j < k_round; ++j) {
      *output++ = static_cast<float>(0);
      *output++ = static_cast<float>(0);
      *output++ = static_cast<float>(0);
      *output++ = static_cast<float>(0);
    }
  }
}

#ifdef __aarch64__

void loadb_c4(float* out,
              const float* in,
              const int xstart,
              const int xend,
              const int k_round,
              const int n) {
  const int xlen = (xend - xstart + NBLOCK_C4 - 1) / NBLOCK_C4 * NBLOCK_C4;
  const int xloop = xlen / NBLOCK_C4;
  const int flag_remain = n < xend;
  const int remain = (n - xstart) - (xloop - 1) * NBLOCK_C4;
  const int ldo = NBLOCK_C4 * k_round;
  const int k_loop = k_round >> 2;
  in += xstart * 4;
#pragma omp parallel for
  for (int i = 0; i < k_loop; ++i) {
    float* out_ptr = out + 4 * NBLOCK_C4 * i;
    const float* in_ptr = in + i * 4 * n;
    for (int j = 0; j < xloop; ++j) {
      float* out_p = out_ptr + j * ldo;
      if ((j == xloop - 1) && flag_remain) {
        for (int r = 0; r < remain; ++r) {
          float32x4_t q0 = vld1q_f32(in_ptr + r * 4);
          vst1q_f32(out_p + r * 4, q0);
        }
        memset(out_p + remain * 4, 0, (NBLOCK_C4 - remain) * 4 * sizeof(float));
      } else {
        asm volatile(
            "ld1 {v0.4s, v1.4s}, [%[in]],  #32  \n"
            "ld1 {v2.4s, v3.4s}, [%[in]],  #32  \n"
            "st1 {v0.4s, v1.4s}, [%[out]], #32  \n"
            "ld1 {v4.4s, v5.4s}, [%[in]],  #32  \n"
            "st1 {v2.4s, v3.4s}, [%[out]], #32  \n"
            "ld1 {v6.4s, v7.4s}, [%[in]],  #32  \n"
            "st1 {v4.4s, v5.4s}, [%[out]], #32  \n"
            "st1 {v6.4s, v7.4s}, [%[out]], #32  \n"
            : [in] "+r"(in_ptr), [out] "+r"(out_p)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
      }
    }
  }
}

void sgemm_prepack_c4(int M,
                      int N,
                      int K,
                      const float* A_packed,
                      const float* B,
                      float* C,
                      const float* bias,
                      bool has_bias,
                      bool has_relu,
                      ARMContext* ctx) {
  const int m_round = (M + 3) / 4 * 4;
  const int k_round = (K + 3) / 4 * 4;
  size_t l2_cache = ctx->llc_size() > 0 ? ctx->llc_size() : 512 * 1024;
  int threads = ctx->threads();
  auto workspace = ctx->workspace_data<float>();
  // l2 = ablock * K * threads + K * bchunk_w + threads * ablock * bchunk_w;
  int bchunk_w = (l2_cache - threads * k_round * sizeof(float)) /
                 ((k_round + threads * MBLOCK_C4) * sizeof(float));
  bchunk_w = bchunk_w > N ? N : bchunk_w;
  bchunk_w = bchunk_w / NBLOCK_C4 * NBLOCK_C4;
  bchunk_w = bchunk_w > NBLOCK_C4 ? bchunk_w : NBLOCK_C4;
  int bchunk_loop = (N + bchunk_w - 1) / bchunk_w;

  const int h_loop = m_round >> 2;  // MBLOCK_C4 == 4;
  const int kcnt = (k_round + KBLOCK_C4 - 1) / KBLOCK_C4;
  const int ldc = N * 4;
  const int lda = k_round * 4;
  float bias_buf[m_round];  // NOLINT
  if (has_bias) {
    memcpy(bias_buf, bias, M * sizeof(float));
    memset(bias_buf + M, 0, (m_round - M) * sizeof(float));
  } else {
    memset(bias_buf, 0, m_round * sizeof(float));
  }
  // bchunk_loop
  float* c = C;
  for (int n = 0; n < bchunk_loop; ++n) {
    int x_start = n * bchunk_w;
    int x_end = x_start + bchunk_w;
    int w_loop = bchunk_w / NBLOCK_C4;
    int flag_remain = 0;
    if (x_end > N) {
      w_loop = (N - x_start + NBLOCK_C4 - 1) / NBLOCK_C4;
      x_end = x_start + w_loop * NBLOCK_C4;
      flag_remain = 1;
    }
    int bblock_remain = N - x_start - (w_loop - 1) * NBLOCK_C4;
    float* bchunk = workspace;
    loadb_c4(bchunk, B, x_start, x_end, k_round, N);
    float* cchunk = c + n * bchunk_w * 4;
#pragma omp parallel for num_threads(threads)
    for (int h = 0; h < h_loop; ++h) {
      float* bias_h = bias_buf + h * 4;
      const float* ablock = A_packed + h * lda;
      const float* bblock = bchunk;
      float* cblock = cchunk + h * ldc;
      for (int w = 0; w < w_loop; ++w) {
        int cnt = kcnt;
        const float* ablock_ptr = ablock;
        float* cblock_ptr = cblock;
        float remain_buf[4 * NBLOCK_C4];
        float* cbuf = remain_buf;
        int has_remain =
            (n == bchunk_loop - 1) && (w == w_loop - 1) && flag_remain;
        asm volatile(
            /* load bias to v0 */
            "ld1  {v0.4s}, [%[bias]] \n"
            "prfm pldl1keep, [%[a]]  \n"
            "prfm pldl1keep, [%[b]]  \n"
            "prfm pldl1keep, [%[b], #64]  \n"
            "mov  v9.16b,   v0.16b   \n" /* mov bias to c0*/
            "mov  v10.16b,  v0.16b   \n" /* mov bias to c1*/
            "mov  v11.16b,  v0.16b   \n" /* mov bias to c2*/
            "mov  v12.16b,  v0.16b   \n" /* mov bias to c3*/
            /* load a0a1 to v1-v2 */
            "ld1   {v1.4s, v2.4s}, [%[a]], #32 \n"
            "mov  v13.16b,  v0.16b   \n" /* mov bias to c4*/
            "mov  v14.16b,  v0.16b   \n" /* mov bias to c5*/
            "mov  v15.16b,  v0.16b   \n" /* mov bias to c6*/
            "mov  v16.16b,  v0.16b   \n" /* mov bias to c7*/
            "movi v30.16b,  #0       \n"
            "1:\n"
            /* load b0b1b2b3 to v5-v8 */
            "ld1   {v5.4s, v6.4s}, [%[b]], #32 \n"
            "ld1   {v7.4s, v8.4s}, [%[b]], #32 \n"
            "prfm  pldl1keep, [%[b]]        \n"
            "fmla  v9.4s,  v1.4s, v5.s[0]   \n"
            "fmla  v10.4s, v1.4s, v6.s[0]   \n"
            "fmla  v11.4s, v1.4s, v7.s[0]   \n"
            "fmla  v12.4s, v1.4s, v8.s[0]   \n"
            /* load b4b5b6b7 to v25-v28 */
            "ld1   {v25.4s, v26.4s}, [%[b]], #32 \n"
            "ld1   {v27.4s, v28.4s}, [%[b]], #32 \n"
            "prfm  pldl1keep, [%[a], #32]   \n"
            "fmla  v9.4s,  v2.4s, v5.s[1]   \n"
            "fmla  v10.4s, v2.4s, v6.s[1]   \n"
            "fmla  v11.4s, v2.4s, v7.s[1]   \n"
            "fmla  v12.4s, v2.4s, v8.s[1]   \n"
            "prfm  pldl1keep, [%[b], #64]   \n"
            "fmla  v13.4s, v1.4s, v25.s[0]  \n"
            "fmla  v14.4s, v1.4s, v26.s[0]  \n"
            "fmla  v15.4s, v1.4s, v27.s[0]  \n"
            "fmla  v16.4s, v1.4s, v28.s[0]  \n"
            /* load a2a3 to v3-v4 */
            "ld1   {v3.4s, v4.4s},  [%[a]], #32 \n"
            "prfm  pldl1keep, [%[b], #128]  \n"
            "fmla  v13.4s, v2.4s, v25.s[1]  \n"
            "fmla  v14.4s, v2.4s, v26.s[1]  \n"
            "fmla  v15.4s, v2.4s, v27.s[1]  \n"
            "fmla  v16.4s, v2.4s, v28.s[1]  \n"
            "subs  %w[cnt], %w[cnt], #1     \n"
            "fmla  v9.4s,  v3.4s, v5.s[2]   \n"
            "fmla  v10.4s, v3.4s, v6.s[2]   \n"
            "fmla  v11.4s, v3.4s, v7.s[2]   \n"
            "fmla  v12.4s, v3.4s, v8.s[2]   \n"
            "fmla  v13.4s, v3.4s, v25.s[2]  \n"
            "fmla  v14.4s, v3.4s, v26.s[2]  \n"
            "fmla  v15.4s, v3.4s, v27.s[2]  \n"
            "fmla  v16.4s, v3.4s, v28.s[2]  \n"
            /* load a0a1 to v1-v2 */
            "ld1   {v1.4s, v2.4s}, [%[a]], #32 \n"
            "fmla  v9.4s,  v4.4s, v5.s[3]   \n"
            "fmla  v10.4s, v4.4s, v6.s[3]   \n"
            "fmla  v11.4s, v4.4s, v7.s[3]   \n"
            "fmla  v12.4s, v4.4s, v8.s[3]   \n"

            "fmla  v13.4s, v4.4s, v25.s[3]  \n"
            "fmla  v14.4s, v4.4s, v26.s[3]  \n"
            "fmla  v15.4s, v4.4s, v27.s[3]  \n"
            "fmla  v16.4s, v4.4s, v28.s[3]  \n"
            "bne   1b\n"
            "cbz   %w[relu], 2f             \n"
            "fmax  v9.4s,  v9.4s,  v30.4s   \n"
            "fmax  v10.4s, v10.4s, v30.4s   \n"
            "fmax  v11.4s, v11.4s, v30.4s   \n"
            "fmax  v12.4s, v12.4s, v30.4s   \n"
            "fmax  v13.4s, v13.4s, v30.4s   \n"
            "fmax  v14.4s, v14.4s, v30.4s   \n"
            "fmax  v15.4s, v15.4s, v30.4s   \n"
            "fmax  v16.4s, v16.4s, v30.4s   \n"
            "2:\n"
            "cbnz  %[frem], 3f\n"
            "st1   {v9.4s,  v10.4s, v11.4s, v12.4s}, [%[c]], #64  \n"
            "st1   {v13.4s, v14.4s, v15.4s, v16.4s}, [%[c]], #64  \n"
            "b     4f\n"
            "3:\n"
            "st1   {v9.4s,  v10.4s, v11.4s, v12.4s}, [%[cbuf]], #64  \n"
            "st1   {v13.4s, v14.4s, v15.4s, v16.4s}, [%[cbuf]], #64  \n"
            "4:\n"
            : [a] "+r"(ablock_ptr),
              [b] "+r"(bblock),
              [c] "+r"(cblock_ptr),
              [cbuf] "+r"(cbuf),
              [cnt] "+r"(cnt)
            : [bias] "r"(bias_h), [relu] "r"(has_relu), [frem] "r"(has_remain)
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
              "v30",
              "cc",
              "memory");
        int clen = NBLOCK_C4;
        if (has_remain) {
          for (int i = 0; i < bblock_remain; ++i) {
            float32x4_t ci = vld1q_f32(remain_buf + i * 4);
            vst1q_f32(cblock + i * 4, ci);
          }
          clen = bblock_remain;
        }
        cblock += clen * 4;
      }
    }
  }
}
#else
void loadb_c4(float* out,
              const float* in,
              const int xstart,
              const int xend,
              const int k_round,
              const int n) {
  const int xlen = (xend - xstart + NBLOCK_C4 - 1) / NBLOCK_C4 * NBLOCK_C4;
  const int xloop = xlen / NBLOCK_C4;
  const int flag_remain = n < xend;
  const int remain = (n - xstart) - (xloop - 1) * NBLOCK_C4;
  const int ldo = NBLOCK_C4 * k_round;
  const int k_loop = k_round >> 2;
  in += xstart * 4;
#pragma omp parallel for
  for (int i = 0; i < k_loop; ++i) {
    float* out_ptr = out + 4 * NBLOCK_C4 * i;
    const float* in_ptr = in + i * 4 * n;
    for (int j = 0; j < xloop; ++j) {
      float* out_p = out_ptr + j * ldo;
      if ((j == xloop - 1) && flag_remain) {
        for (int r = 0; r < remain; ++r) {
          float32x4_t q0 = vld1q_f32(in_ptr + r * 4);
          vst1q_f32(out_p + r * 4, q0);
        }
        memset(out_p + remain * 4, 0, (NBLOCK_C4 - remain) * 4 * sizeof(float));
      } else {
        asm volatile(
            "vld1.32 {d0-d3},   [%[in]]!  \n"
            "vld1.32 {d4-d7},   [%[in]]!  \n"
            "vst1.32 {d0-d3},   [%[out]]! \n"
            "vld1.32 {d8-d11},  [%[in]]!  \n"
            "vst1.32 {d4-d7},   [%[out]]! \n"
            "vld1.32 {d12-d15}, [%[in]]!  \n"
            "vst1.32 {d8-d11},  [%[out]]! \n"
            "vst1.32 {d12-d15}, [%[out]]! \n"
            : [in] "+r"(in_ptr), [out] "+r"(out_p)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
      }
    }
  }
}

void sgemm_prepack_c4(int M,
                      int N,
                      int K,
                      const float* A_packed,
                      const float* B,
                      float* C,
                      const float* bias,
                      bool has_bias,
                      bool has_relu,
                      ARMContext* ctx) {
  const int m_round = (M + 3) / 4 * 4;
  const int k_round = (K + 3) / 4 * 4;
  size_t l2_cache = ctx->llc_size() > 0 ? ctx->llc_size() : 512 * 1024;
  int threads = ctx->threads();
  auto workspace = ctx->workspace_data<float>();
  // l2 = ablock * K * threads + K * bchunk_w + threads * ablock * bchunk_w;
  int bchunk_w = (l2_cache - threads * k_round * sizeof(float)) /
                 ((k_round + threads * MBLOCK_C4) * sizeof(float));
  bchunk_w = bchunk_w > N ? N : bchunk_w;
  bchunk_w = bchunk_w / NBLOCK_C4 * NBLOCK_C4;
  bchunk_w = bchunk_w > NBLOCK_C4 ? bchunk_w : NBLOCK_C4;
  int bchunk_loop = (N + bchunk_w - 1) / bchunk_w;

  const int h_loop = m_round >> 2;  // MBLOCK_C4 == 4;
  const int kcnt = (k_round + KBLOCK_C4 - 1) / KBLOCK_C4;
  const int ldc = N * 4;
  const int lda = k_round * 4;
  float bias_buf[m_round];  // NOLINT
  if (has_bias) {
    memcpy(bias_buf, bias, M * sizeof(float));
    memset(bias_buf + M, 0, (m_round - M) * sizeof(float));
  } else {
    memset(bias_buf, 0, m_round * sizeof(float));
  }
  // bchunk_loop
  float* c = C;
  for (int n = 0; n < bchunk_loop; ++n) {
    int x_start = n * bchunk_w;
    int x_end = x_start + bchunk_w;
    int w_loop = bchunk_w / NBLOCK_C4;
    int flag_remain = 0;
    if (x_end > N) {
      w_loop = (N - x_start + NBLOCK_C4 - 1) / NBLOCK_C4;
      x_end = x_start + w_loop * NBLOCK_C4;
      flag_remain = 1;
    }
    int bblock_remain = N - x_start - (w_loop - 1) * NBLOCK_C4;
    float* bchunk = workspace;
    loadb_c4(bchunk, B, x_start, x_end, k_round, N);
    float* cchunk = c + n * bchunk_w * 4;
#pragma omp parallel for num_threads(threads)
    for (int h = 0; h < h_loop; ++h) {
      float* bias_h = bias_buf + h * 4;
      const float* ablock = A_packed + h * lda;
      const float* bblock = bchunk;
      float* cblock = cchunk + h * ldc;
      for (int w = 0; w < w_loop; ++w) {
        int cnt = kcnt;
        const float* ablock_ptr = ablock;
        float* cblock_ptr = cblock;
        float remain_buf[4 * NBLOCK_C4];
        float* cbuf = remain_buf;
        int has_remain =
            (n == bchunk_loop - 1) && (w == w_loop - 1) && flag_remain;
        asm volatile(
            /* load bias to v0 */
            "vld1.32  {d6-d7}, [%[bias]] \n"
            "pld [%[a]]  \n"
            "pld [%[b]]  \n"
            "pld [%[b], #64]  \n"
            "vmov.32  q8,   q3   \n" /* mov bias to c0*/
            "vmov.32  q9,   q3   \n" /* mov bias to c1*/
            "vmov.32  q10,  q3   \n" /* mov bias to c2*/
            "vmov.32  q11,  q3   \n" /* mov bias to c3*/
            "vld1.32   {d0-d3}, [%[a]]! \n"
            "vmov.32  q12,  q3   \n" /* mov bias to c4*/
            "vmov.32  q13,  q3   \n" /* mov bias to c5*/
            "vmov.32  q14,  q3   \n" /* mov bias to c6*/
            "vmov.32  q15,  q3   \n" /* mov bias to c7*/
            "1:\n"
            /* c0c1c2c3 */
            "vld1.32   {d8-d11},  [%[b]]! \n"
            "vld1.32   {d12-d15}, [%[b]]! \n"
            "pld  [%[b]]                  \n"
            "vmla.f32  q8,  q0, d8[0]     \n"
            "vmla.f32  q9,  q0, d10[0]    \n"
            "vmla.f32  q10, q0, d12[0]    \n"
            "vmla.f32  q11, q0, d14[0]    \n"
            "vld1.32   {d4-d7}, [%[a]]!   \n"
            "vmla.f32  q8,  q1, d8[1]     \n"
            "vmla.f32  q9,  q1, d10[1]    \n"
            "vmla.f32  q10, q1, d12[1]    \n"
            "vmla.f32  q11, q1, d14[1]    \n"
            "pld [%[b], #64]              \n"
            "vmla.f32  q8,  q2, d9[0]     \n"
            "vmla.f32  q9,  q2, d11[0]    \n"
            "vmla.f32  q10, q2, d13[0]    \n"
            "vmla.f32  q11, q2, d15[0]    \n"
            "subs  %[cnt], %[cnt], #1     \n"
            "vmla.f32  q8,  q3, d9[1]     \n"
            "vmla.f32  q9,  q3, d11[1]    \n"
            "vld1.f32  {d8-d11}, [%[b]]!  \n"
            "vmla.f32  q10, q3, d13[1]    \n"
            "vmla.f32  q11, q3, d15[1]    \n"
            "vld1.32   {d12-d15}, [%[b]]! \n"
            /* c4c5c6c7 */
            "vmla.f32  q12, q0, d8[0]     \n"
            "vmla.f32  q13, q0, d10[0]    \n"
            "vmla.f32  q14, q0, d12[0]    \n"
            "vmla.f32  q15, q0, d14[0]    \n"
            "pld  [%[a], #32]             \n"
            "vmla.f32  q12, q1, d8[1]     \n"
            "vmla.f32  q13, q1, d10[1]    \n"
            "vmla.f32  q14, q1, d12[1]    \n"
            "vmla.f32  q15, q1, d14[1]    \n"
            "vld1.32   {d0-d3}, [%[a]]!   \n"
            "vmla.f32  q12, q2, d9[0]     \n"
            "vmla.f32  q13, q2, d11[0]    \n"
            "vmla.f32  q14, q2, d13[0]    \n"
            "vmla.f32  q15, q2, d15[0]    \n"
            "pld [%[b], #64]              \n"
            "vmla.f32  q12, q3, d9[1]     \n"
            "vmla.f32  q13, q3, d11[1]    \n"
            "vmla.f32  q14, q3, d13[1]    \n"
            "vmla.f32  q15, q3, d15[1]    \n"
            "bne   1b\n"
            "cmp   %[relu], #0            \n"
            "beq   2f                     \n"
            "vmov.u32 q0, #0              \n"
            "vmax.f32  q8,   q8,   q0     \n"
            "vmax.f32  q9,   q9,   q0     \n"
            "vmax.f32  q10,  q10,  q0     \n"
            "vmax.f32  q11,  q11,  q0     \n"
            "vmax.f32  q12,  q12,  q0     \n"
            "vmax.f32  q13,  q13,  q0     \n"
            "vmax.f32  q14,  q14,  q0     \n"
            "vmax.f32  q15,  q15,  q0     \n"
            "2:\n"
            "cmp   %[frem], #0\n"
            "bne   3f\n"
            "vst1.32   {d16-d19}, [%[c]]!  \n"
            "vst1.32   {d20-d23}, [%[c]]!  \n"
            "vst1.32   {d24-d27}, [%[c]]!  \n"
            "vst1.32   {d28-d31}, [%[c]]!  \n"
            "b     4f\n"
            "3:\n"
            "vst1.32   {d16-d19}, [%[cbuf]]!  \n"
            "vst1.32   {d20-d23}, [%[cbuf]]!  \n"
            "vst1.32   {d24-d27}, [%[cbuf]]!  \n"
            "vst1.32   {d28-d31}, [%[cbuf]]!  \n"
            "4:\n"
            : [a] "+r"(ablock_ptr),
              [b] "+r"(bblock),
              [c] "+r"(cblock_ptr),
              [cbuf] "+r"(cbuf),
              [cnt] "+r"(cnt)
            : [bias] "r"(bias_h), [relu] "r"(has_relu), [frem] "r"(has_remain)
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
        int clen = NBLOCK_C4;
        if (has_remain) {
          for (int i = 0; i < bblock_remain; ++i) {
            float32x4_t ci = vld1q_f32(remain_buf + i * 4);
            vst1q_f32(cblock + i * 4, ci);
          }
          clen = bblock_remain;
        }
        cblock += clen * 4;
      }
    }
  }
}
#endif

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
