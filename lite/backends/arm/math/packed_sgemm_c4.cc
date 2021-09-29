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
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void loadb_c4(float* out,
              const float* in,
              const int xstart,
              const int xend,
              const int k_round,
              const int n) {
  const int xlen = (xend - xstart + NBLOCK_C4 - 1) / NBLOCK_C4 * NBLOCK_C4;
  int xloop = xlen / NBLOCK_C4;
  const int flag_remain = n < xstart + xlen;
  int remain = 0;
  int remain4 = 0;
  int remain1 = 0;
  if (flag_remain) {
    remain = (n - xstart) - (xloop - 1) * NBLOCK_C4;
    remain4 = remain >> 2;
    remain1 = remain & 3;
    xloop -= 1;
  }
  const int ldo = NBLOCK_C4 * k_round;
  const int kloop = k_round >> 2;
  in += xstart * 4;
  if (xloop > 0) {
    LITE_PARALLEL_BEGIN(i, tid, kloop) {
      float* out_ptr = out + 4 * NBLOCK_C4 * i;
      const float* in_ptr = in + i * 4 * n;
      for (int j = 0; j < xloop; ++j) {
        float* out_p = out_ptr + j * ldo;
#ifdef __aarch64__
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
#else
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
#endif  // __aarch674__
      }
    }
    LITE_PARALLEL_END();
  }
  float* out_remain4 = out + xloop * k_round * NBLOCK_C4;
  const float* in_remain4 = in + xloop * NBLOCK_C4 * 4;
  if (remain4) {
    LITE_PARALLEL_BEGIN(i, tid, kloop) {
      float* out_ptr = out_remain4 + 4 * 4 * i;
      const float* in_ptr = in_remain4 + i * 4 * n;
#ifdef __aarch64__
      asm volatile(
          "ld1 {v0.4s, v1.4s}, [%[in]], #32  \n"
          "ld1 {v2.4s, v3.4s}, [%[in]], #32  \n"
          "st1 {v0.4s, v1.4s}, [%[out]], #32 \n"
          "st1 {v2.4s, v3.4s}, [%[out]], #32 \n"
          : [in] "+r"(in_ptr), [out] "+r"(out_ptr)
          :
          : "v0", "v1", "v2", "v3");
#else
      asm volatile(
          "vld1.32 {d0-d3}, [%[in]]!  \n"
          "vld1.32 {d4-d7}, [%[in]]!  \n"
          "vst1.32 {d0-d3}, [%[out]]! \n"
          "vst1.32 {d4-d7}, [%[out]]! \n"
          : [in] "+r"(in_ptr), [out] "+r"(out_ptr)
          :
          : "q0", "q1", "q2", "q3");
#endif  // __aarch64__
    }
    LITE_PARALLEL_END();
  }
  float* out_remain1 = out_remain4 + remain4 * k_round * 4;
  const float* in_remain1 = in_remain4 + remain4 * 4 * 4;
  if (remain1) {
    LITE_PARALLEL_BEGIN(i, tid, kloop) {
      float* out_ptr = out_remain1 + 4 * remain1 * i;
      const float* in_ptr = in_remain1 + i * 4 * n;
      for (int j = 0; j < remain1; ++j) {
        float32x4_t vin = vld1q_f32(in_ptr);
        in_ptr += 4;
        vst1q_f32(out_ptr, vin);
        out_ptr += 4;
      }
    }
    LITE_PARALLEL_END();
  }
}

void sgemm_prepack_c4_common(int M,
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
    lite::TargetWrapperHost::MemcpySync(bias_buf, bias, M * sizeof(float));
    memset(bias_buf + M, 0, (m_round - M) * sizeof(float));
  } else {
    memset(bias_buf, 0, m_round * sizeof(float));
  }
  // bchunk_loop
  float* c = C;
#ifdef __aarch64__
  for (int n = 0; n < bchunk_loop; ++n) {
    int x_start = n * bchunk_w;
    int x_end = x_start + bchunk_w;
    int w_loop = bchunk_w / NBLOCK_C4;
    int flag_remain = 0;
    int w_loop4 = 0;
    int remain = 0;
    if (x_end > N) {
      w_loop = (N - x_start) / NBLOCK_C4;
      int w_loop_rem = (N - x_start) - w_loop * NBLOCK_C4;
      w_loop4 = w_loop_rem >> 2;
      remain = w_loop_rem & 3;
      x_end = N;
      flag_remain = 1;
    }
    float* bchunk = workspace;
    loadb_c4(bchunk, B, x_start, x_end, k_round, N);
    float* cchunk = c + n * bchunk_w * 4;
    int has_remain = (n == bchunk_loop - 1) && flag_remain;
    LITE_PARALLEL_BEGIN(h, tid, h_loop) {
      float* bias_h = bias_buf + h * 4;
      float32x4_t vzero = vdupq_n_f32(0.f);
      float32x4_t vbias = vld1q_f32(bias_h);
      const float* ablock = A_packed + h * lda;
      const float* bblock = bchunk;
      float* cblock = cchunk + h * ldc;
      for (int w = 0; w < w_loop; ++w) {
        int cnt = kcnt;
        const float* ablock_ptr = ablock;
        // clang-format off
          asm volatile(
            "prfm pldl1keep, [%[a]]         \n"
            "prfm pldl1keep, [%[b]]         \n"
            "prfm pldl1keep, [%[b], #64]    \n"
            "mov  v9.16b,   %[vbias].16b    \n" /* mov bias to c0*/
            "mov  v10.16b,  %[vbias].16b    \n" /* mov bias to c1*/
            "mov  v11.16b,  %[vbias].16b    \n" /* mov bias to c2*/
            "mov  v12.16b,  %[vbias].16b    \n" /* mov bias to c3*/
            /* load a0a1 to v1-v2  */
            "ld1   {v1.4s, v2.4s}, [%[a]], #32 \n"
            "mov  v13.16b,  %[vbias].16b    \n" /* mov bias to c4*/
            "mov  v14.16b,  %[vbias].16b    \n" /* mov bias to c5*/
            "mov  v15.16b,  %[vbias].16b    \n" /* mov bias to c6*/
            "mov  v16.16b,  %[vbias].16b    \n" /* mov bias to c7*/
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
            "fmax  v9.4s,  v9.4s,  %[vzero].4s  \n"
            "fmax  v10.4s, v10.4s, %[vzero].4s  \n"
            "fmax  v11.4s, v11.4s, %[vzero].4s  \n"
            "fmax  v12.4s, v12.4s, %[vzero].4s  \n"
            "fmax  v13.4s, v13.4s, %[vzero].4s  \n"
            "fmax  v14.4s, v14.4s, %[vzero].4s  \n"
            "fmax  v15.4s, v15.4s, %[vzero].4s  \n"
            "fmax  v16.4s, v16.4s, %[vzero].4s  \n"
            "2:\n"
            "st1   {v9.4s,  v10.4s, v11.4s, v12.4s}, [%[c]], #64  \n"
            "st1   {v13.4s, v14.4s, v15.4s, v16.4s}, [%[c]], #64  \n"
            : [a] "+r"(ablock_ptr),
              [b] "+r"(bblock),
              [c] "+r"(cblock),
              [cnt] "+r"(cnt)
            : [bias] "r"(bias_h), [relu] "r"(has_relu), 
              [vbias] "w"(vbias), [vzero] "w" (vzero) 
            : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", 
              "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", 
              "v25", "v26", "v27", "v28", "cc", "memory");
        // clang-format on
      }
      if (has_remain) {
        if (w_loop4 > 0) {
          int cnt = kcnt;
          const float* ablock_ptr = ablock;
          // clang-format off
            asm volatile(
              "prfm pldl1keep, [%[a]]         \n"
              "prfm pldl1keep, [%[b]]         \n"
              "mov  v9.16b,   %[vbias].16b    \n" /* mov bias to c0*/
              "mov  v10.16b,  %[vbias].16b    \n" /* mov bias to c1*/
              "mov  v11.16b,  %[vbias].16b    \n" /* mov bias to c2*/
              "mov  v12.16b,  %[vbias].16b    \n" /* mov bias to c3*/
              /* load a0a1 to v1-v2 */
              "ld1   {v1.4s, v2.4s}, [%[a]], #32 \n"
              "1:\n"
              /* load b0b1b2b3 to v5-v8 */
              "ld1   {v5.4s, v6.4s}, [%[b]], #32 \n"
              "ld1   {v7.4s, v8.4s}, [%[b]], #32 \n"
              "fmla  v9.4s,  v1.4s, v5.s[0]   \n"
              "fmla  v10.4s, v1.4s, v6.s[0]   \n"
              "fmla  v11.4s, v1.4s, v7.s[0]   \n"
              "fmla  v12.4s, v1.4s, v8.s[0]   \n"
              /* load a2a3 to v3-v4 */
              "ld1   {v3.4s, v4.4s},  [%[a]], #32 \n"
              "prfm  pldl1keep, [%[a]]        \n"
              "fmla  v9.4s,  v2.4s, v5.s[1]   \n"
              "fmla  v10.4s, v2.4s, v6.s[1]   \n"
              "fmla  v11.4s, v2.4s, v7.s[1]   \n"
              "fmla  v12.4s, v2.4s, v8.s[1]   \n"
              "prfm  pldl1keep, [%[b]]        \n"
              "subs  %w[cnt], %w[cnt], #1     \n"
              "fmla  v9.4s,  v3.4s, v5.s[2]   \n"
              "fmla  v10.4s, v3.4s, v6.s[2]   \n"
              "fmla  v11.4s, v3.4s, v7.s[2]   \n"
              "fmla  v12.4s, v3.4s, v8.s[2]   \n"
              /* load a0a1 to v1-v2 */
              "ld1   {v1.4s, v2.4s}, [%[a]], #32 \n"
              "fmla  v9.4s,  v4.4s, v5.s[3]   \n"
              "fmla  v10.4s, v4.4s, v6.s[3]   \n"
              "fmla  v11.4s, v4.4s, v7.s[3]   \n"
              "fmla  v12.4s, v4.4s, v8.s[3]   \n"
              "bne   1b\n"
              "cbz   %w[relu], 2f             \n"
              "fmax  v9.4s,  v9.4s,  %[vzero].4s  \n"
              "fmax  v10.4s, v10.4s, %[vzero].4s  \n"
              "fmax  v11.4s, v11.4s, %[vzero].4s  \n"
              "fmax  v12.4s, v12.4s, %[vzero].4s  \n"
              "2:\n"
              "st1   {v9.4s,  v10.4s, v11.4s, v12.4s}, [%[c]], #64  \n"
              : [a] "+r"(ablock_ptr),
                [b] "+r"(bblock),
                [c] "+r"(cblock),
                [cnt] "+r"(cnt)
              : [bias] "r"(bias_h),
                [relu] "r"(has_relu),
                [vbias] "w"(vbias), 
                [vzero] "w" (vzero)   
              : "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                "v8", "v9", "v10", "v11", "v12", "cc", "memory");
          // clang-format on
        }
        if (remain > 0) {
          int cnt = kcnt;
          const float* ablock_ptr = ablock;
          // clang-format off
            asm volatile(
              "prfm pldl1keep, [%[a]]   \n"
              "prfm pldl1keep, [%[b]]   \n"
              "ld1   {v1.4s, v2.4s}, [%[a]], #32 \n"
              "cmp  %w[remain], #3      \n"
              "beq  1f                  \n"
              "cmp  %w[remain], #2      \n"
              "beq  2f                  \n"
              /* remain 1 */
              "mov  v9.16b,   %[vbias].16b  \n" /* mov bias to c0*/
              "mov  v10.16b,  %[vzero].16b  \n" /* mov zero to c1*/
              "3:                                 \n"
              "ld1   {v5.4s}, [%[b]], #16         \n"
              "ld1   {v3.4s,  v4.4s}, [%[a]], #32 \n"
              "fmla  v9.4s,   v1.4s,  v5.s[0]     \n"
              "fmla  v10.4s,  v2.4s,  v5.s[1]     \n"
              "subs  %w[cnt], %w[cnt], #1         \n"
              "ld1   {v1.4s,  v2.4s}, [%[a]], #32 \n"
              "fmla  v9.4s,   v3.4s,  v5.s[2]     \n"
              "fmla  v10.4s,  v4.4s,  v5.s[3]     \n"
              "bne   3b                           \n"
              "fadd  v9.4s,   v9.4s,  v10.4s      \n"
              "cbz   %w[relu], 6f                 \n"
              "fmax  v9.4s,   v9.4s,  %[vzero].4s \n"
              "6:                                 \n"
              "st1   {v9.4s}, [%[c]], #16         \n"
              "b     9f                           \n"
              /* remain 2 */
              "2:                           \n"
              "mov  v9.16b,   %[vbias].16b  \n" /* mov bias to c0*/
              "mov  v10.16b,  %[vbias].16b  \n" /* mov bias to c1*/
              "mov  v11.16b,  %[vzero].16b  \n" /* mov zero to c2*/
              "mov  v12.16b,  %[vzero].16b  \n" /* mov zero to c3*/
              "4:                                 \n"
              "ld1   {v5.4s,  v6.4s}, [%[b]], #32 \n"
              "ld1   {v3.4s,  v4.4s}, [%[a]], #32 \n"
              "fmla  v9.4s,   v1.4s,  v5.s[0]     \n"
              "fmla  v10.4s,  v1.4s,  v6.s[0]     \n"
              "fmla  v11.4s,  v2.4s,  v5.s[1]     \n"
              "fmla  v12.4s,  v2.4s,  v6.s[1]     \n"
              "subs  %w[cnt], %w[cnt], #1         \n"
              "fmla  v9.4s,   v3.4s,  v5.s[2]     \n"
              "fmla  v10.4s,  v3.4s,  v6.s[2]     \n"
              "fmla  v11.4s,  v4.4s,  v5.s[3]     \n"
              "fmla  v12.4s,  v4.4s,  v6.s[3]     \n"
              "ld1   {v1.4s,  v2.4s}, [%[a]], #32 \n"
              "bne   4b                           \n"
              "fadd  v9.4s,   v9.4s,  v11.4s      \n"
              "fadd  v10.4s,  v10.4s, v12.4s      \n"
              "cbz   %w[relu], 7f                 \n"
              "fmax  v9.4s,   v9.4s,  %[vzero].4s \n"
              "fmax  v10.4s,  v10.4s, %[vzero].4s \n"
              "7:                                 \n"
              "st1   {v9.4s, v10.4s}, [%[c]], #32 \n"
              "b     9f                           \n"
              /* remain 3 */
              "1:                       \n"
              "mov  v9.16b,   %[vbias].16b  \n" /* mov bias to c0*/
              "mov  v10.16b,  %[vbias].16b  \n" /* mov bias to c1*/
              "mov  v11.16b,  %[vbias].16b  \n" /* mov bias to c2*/
              "5:                                 \n"
              "ld1   {v5.4s,  v6.4s}, [%[b]], #32 \n"
              "ld1   {v7.4s}, [%[b]], #16         \n"
              "fmla  v9.4s,   v1.4s,  v5.s[0]     \n"
              "fmla  v10.4s,  v1.4s,  v6.s[0]     \n"
              "fmla  v11.4s,  v1.4s,  v7.s[0]     \n"
              "ld1   {v3.4s,  v4.4s}, [%[a]], #32 \n"
              "fmla  v9.4s,   v2.4s,  v5.s[1]     \n"
              "fmla  v10.4s,  v2.4s,  v6.s[1]     \n"
              "fmla  v11.4s,  v2.4s,  v7.s[1]     \n"
              "subs  %w[cnt], %w[cnt], #1         \n"
              "fmla  v9.4s,   v3.4s,  v5.s[2]     \n"
              "fmla  v10.4s,  v3.4s,  v6.s[2]     \n"
              "fmla  v11.4s,  v3.4s,  v7.s[2]     \n"
              "prfm  pldl1keep, [%[a]]            \n"
              "fmla  v9.4s,   v4.4s,  v5.s[3]     \n"
              "fmla  v10.4s,  v4.4s,  v6.s[3]     \n"
              "fmla  v11.4s,  v4.4s,  v7.s[3]     \n"
              "ld1   {v1.4s,  v2.4s}, [%[a]], #32 \n"
              "bne   5b                           \n"
              "cbz   %w[relu], 8f                 \n"
              "fmax  v9.4s,   v9.4s,  %[vzero].4s \n"
              "fmax  v10.4s,  v10.4s, %[vzero].4s \n"
              "fmax  v11.4s,  v11.4s, %[vzero].4s \n"
              "8:                                 \n"
              "st1   {v9.4s, v10.4s}, [%[c]], #32 \n"
              "st1   {v11.4s}, [%[c]], #16        \n"
              "9:\n"
              : [a] "+r"(ablock_ptr),
                [b] "+r"(bblock),
                [c] "+r"(cblock),
                [cnt] "+r"(cnt)
              : [bias] "r"(bias_h), [relu] "r"(has_relu), 
                [remain] "r"(remain), [vbias] "w"(vbias), 
                [vzero] "w" (vzero) 
              : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v9", 
                "v10", "v11", "v12", "cc","memory");
          // clang-format on
        }
      }
    }
    LITE_PARALLEL_END();
  }
#else
  for (int n = 0; n < bchunk_loop; ++n) {
    int x_start = n * bchunk_w;
    int x_end = x_start + bchunk_w;
    int w_loop = bchunk_w / NBLOCK_C4;
    int flag_remain = 0;
    int w_loop4 = 0;
    int remain = 0;
    if (x_end > N) {
      w_loop = (N - x_start) / NBLOCK_C4;
      int w_loop_rem = (N - x_start) - w_loop * NBLOCK_C4;
      w_loop4 = w_loop_rem >> 2;
      remain = w_loop_rem & 3;
      x_end = N;
      flag_remain = 1;
    }
    float* bchunk = workspace;
    loadb_c4(bchunk, B, x_start, x_end, k_round, N);
    float* cchunk = c + n * bchunk_w * 4;
    int has_remain = (n == bchunk_loop - 1) && flag_remain;
    LITE_PARALLEL_BEGIN(h, tid, h_loop) {
      float* bias_h = bias_buf + h * 4;
      const float* ablock = A_packed + h * lda;
      const float* bblock = bchunk;
      float* cblock = cchunk + h * ldc;
      for (int w = 0; w < w_loop; ++w) {
        int cnt = kcnt;
        const float* ablock_ptr = ablock;
        // clang-format off
        asm volatile(
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
            "vst1.32   {d16-d19}, [%[c]]! \n"
            "vst1.32   {d20-d23}, [%[c]]! \n"
            "vst1.32   {d24-d27}, [%[c]]! \n"
            "vst1.32   {d28-d31}, [%[c]]! \n"
            : [a] "+r"(ablock_ptr),
              [b] "+r"(bblock),
              [c] "+r"(cblock),
              [cnt] "+r"(cnt)
            : [bias] "r"(bias_h), 
              [relu] "r"(has_relu)
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
              "q9", "q10", "q11", "q12", "q13", "q14", "q15", "cc", "memory");
        // clang-format on
      }
      if (has_remain) {
        if (w_loop4 > 0) {
          int cnt = kcnt;
          const float* ablock_ptr = ablock;
          // clang-format off
          asm volatile(
            "pld [%[a]]  \n"
            "pld [%[b]]  \n"
            "vld1.32  {d6-d7}, [%[bias]] \n"
            "vld1.32  {d0-d3}, [%[a]]!   \n" /* load a0 a1 */
            "vmov.32  q8,   q3   \n"     /* mov bias to c0 */
            "vmov.32  q9,   q3   \n"     /* mov bias to c1 */
            "vmov.32  q10,  q3   \n"     /* mov bias to c2 */
            "vmov.32  q11,  q3   \n"     /* mov bias to c3 */
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
            "pld [%[a]]                   \n"
            "vmla.f32  q8,  q1, d8[1]     \n"
            "vmla.f32  q9,  q1, d10[1]    \n"
            "vmla.f32  q10, q1, d12[1]    \n"
            "vmla.f32  q11, q1, d14[1]    \n"
            "subs  %[cnt], %[cnt], #1     \n"
            "vmla.f32  q8,  q2, d9[0]     \n"
            "vmla.f32  q9,  q2, d11[0]    \n"
            "vmla.f32  q10, q2, d13[0]    \n"
            "vmla.f32  q11, q2, d15[0]    \n"
            "vld1.32   {d0-d3}, [%[a]]!   \n"
            "vmla.f32  q8,  q3, d9[1]     \n"
            "vmla.f32  q9,  q3, d11[1]    \n"
            "vmla.f32  q10, q3, d13[1]    \n"
            "vmla.f32  q11, q3, d15[1]    \n"
            "bne   1b\n"
            "cmp   %[relu], #0            \n"
            "beq   2f                     \n"
            "vmov.u32  q0, #0             \n"
            "vmax.f32  q8,   q8,   q0     \n"
            "vmax.f32  q9,   q9,   q0     \n"
            "vmax.f32  q10,  q10,  q0     \n"
            "vmax.f32  q11,  q11,  q0     \n"
            "2:\n"
            "vst1.32   {d16-d19}, [%[c]]! \n"
            "vst1.32   {d20-d23}, [%[c]]! \n"
            : [a] "+r"(ablock_ptr),
              [b] "+r"(bblock),
              [c] "+r"(cblock),
              [cnt] "+r"(cnt)
            : [bias] "r"(bias_h), [relu] "r"(has_relu)
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
              "q9", "q10", "q11", "cc", "memory");
          // clang-format on
        }
        if (remain > 0) {
          int cnt = kcnt;
          const float* ablock_ptr = ablock;
          // clang-format off
          asm volatile(
              "pld  [%[a]]  \n"
              "pld  [%[b]]  \n"
              "vld1.32  {d0-d1}, [%[bias]]  \n"
              "vld1.32  {d2-d5}, [%[a]]!    \n"
              "vmov.u32 q15,  #0            \n"
              "cmp  %[remain], #3           \n"
              "beq  1f                      \n"
              "cmp  %[remain], #2           \n"
              "beq  2f                      \n"
              /* remain 1 */
              "vmov.32  q9,   q0  \n" /* mov bias to c0*/
              "vmov.32  q10,  q15 \n" /* mov zero to c1*/
              "3:                             \n"
              "vld1.32   {d10-d11}, [%[b]]!   \n"
              "vld1.32   {d6-d9},   [%[a]]!   \n"
              "vmla.f32  q9,  q1,  d10[0]     \n"
              "vmla.f32  q10, q2,  d10[1]     \n"
              "subs   %[cnt],  %[cnt], #1     \n"
              "vld1.32   {d2-d5},   [%[a]]!   \n"
              "vmla.f32  q9,  q3,  d11[0]     \n"
              "vmla.f32  q10, q4,  d11[1]     \n"
              "bne   3b                       \n"
              "vadd.f32  q9,  q9,  q10        \n"
              "cmp  %[relu],  #0              \n"
              "beq  6f                        \n"
              "vmax.f32  q9,  q9,  q15        \n"
              "6:                             \n"
              "vst1.32   {d18-d19}, [%[c]]!   \n"
              "b     9f                       \n"
              /* remain 2 */
              "2:                             \n"
              "vmov.u32  q9,  q0    \n" /* mov bias to c0*/
              "vmov.u32  q10, q0    \n" /* mov bias to c1*/
              "vmov.u32  q11, q15   \n" /* mov zero to c2*/
              "vmov.u32  q12, q15   \n" /* mov zero to c3*/
              "4:                             \n"
              "vld1.32   {d10-d13}, [%[b]]!   \n"
              "vld1.32   {d6-d9},   [%[a]]!   \n"
              "vmla.f32  q9,   q1,  d10[0]    \n"
              "vmla.f32  q10,  q1,  d12[0]    \n"
              "vmla.f32  q11,  q2,  d10[1]    \n"
              "vmla.f32  q12,  q2,  d12[1]    \n"
              "subs  %[cnt],  %[cnt], #1      \n"
              "vmla.f32  q9,   q3,  d11[0]    \n"
              "vmla.f32  q10,  q3,  d13[0]    \n"
              "vmla.f32  q11,  q4,  d11[1]    \n"
              "vmla.f32  q12,  q4,  d13[1]    \n"
              "vld1.32   {d2-d5},   [%[a]]!   \n"
              "bne   4b                       \n"
              "vadd.f32  q9,   q9,  q11       \n"
              "vadd.f32  q10,  q10, q12       \n"
              "cmp  %[relu],  #0              \n"
              "beq  7f                        \n"
              "vmax.f32  q9,   q9,  q15       \n"
              "vmax.f32  q10,  q10, q15       \n"
              "7:                             \n"
              "vst1.32   {d18-d21}, [%[c]]!   \n"
              "b     9f                       \n"
              /* remain 3 */
              "1:                             \n"
              "vmov.u32  q9,   q0    \n" /* mov bias to c0*/
              "vmov.u32  q10,  q0    \n" /* mov bias to c1*/
              "vmov.u32  q11,  q0    \n" /* mov bias to c2*/
              "5:                             \n"
              "vld1.32   {d10-d13}, [%[b]]!   \n"
              "vld1.32   {d14-d15}, [%[b]]!   \n"
              "vmla.f32  q9,  q1,   d10[0]    \n"
              "vmla.f32  q10, q1,   d12[0]    \n"
              "vmla.f32  q11, q1,   d14[0]    \n"
              "vld1.32   {d6-d9},   [%[a]]!   \n"
              "vmla.f32  q9,  q2,  d10[1]     \n"
              "vmla.f32  q10, q2,  d12[1]     \n"
              "vmla.f32  q11, q2,  d14[1]     \n"
              "subs  %[cnt],  %[cnt], #1      \n"
              "vmla.f32  q9,  q3,  d11[0]     \n"
              "vmla.f32  q10, q3,  d13[0]     \n"
              "vmla.f32  q11, q3,  d15[0]     \n"
              "pld       [%[a]]               \n"
              "vmla.f32  q9,  q4,  d11[1]     \n"
              "vmla.f32  q10, q4,  d13[1]     \n"
              "vmla.f32  q11, q4,  d15[1]     \n"
              "vld1.32   {d2-d5},  [%[a]]!    \n"
              "bne   5b                       \n"
              "cmp  %[relu],  #0              \n"
              "beq  8f                        \n"
              "vmax.f32  q9,  q9,  q15        \n"
              "vmax.f32  q10, q10, q15        \n"
              "vmax.f32  q11, q11, q15        \n"
              "8:                             \n"
              "vst1.32   {d18-d21}, [%[c]]!   \n"
              "vst1.32   {d22-d23}, [%[c]]!   \n"
              "9:\n"
              : [a] "+r"(ablock_ptr),
                [b] "+r"(bblock),
                [c] "+r"(cblock),
                [cnt] "+r"(cnt)
              : [bias] "r"(bias_h), 
                [relu] "r"(has_relu), 
                [remain] "r"(remain)
              : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q9", 
                "q10", "q11", "q12", "q15", "cc","memory");
          // clang-format on
        }
      }
    }
    LITE_PARALLEL_END();
  }
#endif
}

void sgemm_prepack_c4_common_a35(int M,
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
    lite::TargetWrapperHost::MemcpySync(bias_buf, bias, M * sizeof(float));
    memset(bias_buf + M, 0, (m_round - M) * sizeof(float));
  } else {
    memset(bias_buf, 0, m_round * sizeof(float));
  }
  // bchunk_loop
  float* c = C;
#ifdef __aarch64__
  for (int n = 0; n < bchunk_loop; ++n) {
    int x_start = n * bchunk_w;
    int x_end = x_start + bchunk_w;
    int w_loop = bchunk_w / NBLOCK_C4;
    int flag_remain = 0;
    int w_loop4 = 0;
    int remain = 0;
    if (x_end > N) {
      w_loop = (N - x_start) / NBLOCK_C4;
      int w_loop_rem = (N - x_start) - w_loop * NBLOCK_C4;
      w_loop4 = w_loop_rem >> 2;
      remain = w_loop_rem & 3;
      x_end = N;
      flag_remain = 1;
    }
    float* bchunk = workspace;
    loadb_c4(bchunk, B, x_start, x_end, k_round, N);
    float* cchunk = c + n * bchunk_w * 4;
    int has_remain = (n == bchunk_loop - 1) && flag_remain;
    LITE_PARALLEL_BEGIN(h, tid, h_loop) {
      float* bias_h = bias_buf + h * 4;
      float32x4_t vzero = vdupq_n_f32(0.f);
      float32x4_t vbias = vld1q_f32(bias_h);
      const float* ablock = A_packed + h * lda;
      const float* bblock = bchunk;
      float* cblock = cchunk + h * ldc;
      for (int w = 0; w < w_loop; ++w) {
        int cnt = kcnt;
        const float* ablock_ptr = ablock;
        // clang-format off
           asm volatile(
              "prfm pldl1keep, [%[a]]         \n"
              "prfm pldl1keep, [%[b]]         \n"
              "prfm pldl1keep, [%[b], #64]    \n"
              "ld1 {v3.2s}, [%[bias]] \n"
              "mov  v9.16b,   v3.16b    \n"
              "mov  v10.16b,  v3.16b    \n"
              "ld1r   {v5.2s},  [%[b]], #4     \n"
              "mov  v11.16b,  v3.16b    \n"
              "mov  v12.16b,  v3.16b    \n"
              "ld1   {v1.2s}, [%[a]], #8 \n"
              "mov  v13.16b,  v3.16b    \n"
              "mov  v14.16b,  v3.16b    \n"
              "ld1r   {v6.2s},  [%[b]], #4     \n"
              "mov  v15.16b,  v3.16b    \n"
              "mov  v16.16b,  v3.16b    \n"
              "ld1   {v0.2s}, [%[a]], #8 \n"
              "mov  v17.16b,  v3.16b    \n"
              "mov  v18.16b,  v3.16b    \n"
              "ld1   {v2.2s}, [%[a]], #8 \n"
              "mov  v19.16b,  v3.16b    \n"
              "mov  v20.16b,  v3.16b    \n"
              "ld1r   {v7.2s},  [%[b]], #4     \n"
              "mov  v21.16b,  v3.16b    \n"
              "mov  v22.16b,  v3.16b    \n"
              "ld1   {v29.2s},[%[a]], #8 \n"
              "mov  v23.16b,  v3.16b    \n"
              "mov  v24.16b,  v3.16b    \n"
              "ld1r   {v8.2s},  [%[b]], #4     \n"

              "1:\n"
              "ld1r   {v25.2s}, [%[b]], #4     \n"
              "fmla  v9.2s,  v1.2s, v5.2s      \n"
              "ld1   {v3.2s},  [%[a]], #8      \n"
              "fmla  v17.2s, v0.2s, v5.2s      \n"
              "ld1r   {v26.2s}, [%[b]], #4     \n"
              "fmla  v10.2s, v1.2s, v25.2s     \n"
              "ld1   {v30.2s}, [%[a]], #8      \n"
              "fmla  v18.2s, v0.2s, v25.2s     \n"
              "ld1r   {v27.2s}, [%[b]], #4     \n"
              "fmla  v9.2s,  v2.2s, v6.2s      \n"
              "fmla  v17.2s, v29.2s, v6.2s     \n"
              "ld1r   {v28.2s}, [%[b]], #4     \n"
              "fmla  v10.2s, v2.2s, v26.2s     \n"
              "ld1   {v4.2s},  [%[a]], #8      \n"
              "fmla  v18.2s, v29.2s, v26.2s    \n"
              "ld1r   {v5.2s},  [%[b]], #4     \n"
              "fmla  v9.2s, v3.2s, v7.2s       \n"
              "ld1   {v31.2s}, [%[a]], #8      \n"
              "fmla  v17.2s, v30.2s, v7.2s     \n"
              "ld1r   {v6.2s},  [%[b]], #4     \n"
              "fmla  v10.2s, v3.2s, v27.2s     \n"
              "fmla  v18.2s, v30.2s, v27.2s    \n"
              "fmla  v9.2s, v4.2s, v8.2s       \n"
              "ld1r   {v7.2s},  [%[b]], #4     \n"
              "fmla  v17.2s, v31.2s, v8.2s     \n"
              "fmla  v10.2s, v4.2s, v28.2s     \n"
              "fmla  v18.2s, v31.2s, v28.2s    \n"
              "ld1r   {v8.2s},  [%[b]], #4     \n"

              "fmla  v11.2s,  v1.2s, v5.2s   \n"
              "ld1r   {v25.2s}, [%[b]], #4     \n"
              "fmla  v19.2s, v0.2s, v5.2s    \n"
              "ld1r   {v26.2s}, [%[b]], #4     \n"
              "fmla  v12.2s, v1.2s, v25.2s   \n"
              "fmla  v20.2s, v0.2s, v25.2s   \n"
              "ld1r   {v27.2s}, [%[b]], #4     \n"
              "fmla  v11.2s,  v2.2s, v6.2s   \n"
              "fmla  v19.2s, v29.2s, v6.2s   \n"
              "ld1r   {v28.2s}, [%[b]], #4     \n"
              "fmla  v12.2s, v2.2s, v26.2s   \n"
              "fmla  v20.2s, v29.2s, v26.2s  \n"
              "ld1r   {v5.2s},  [%[b]], #4   \n"
              "fmla  v11.2s, v3.2s, v7.2s    \n"
              "fmla  v19.2s, v30.2s, v7.2s   \n"
              "fmla  v12.2s, v3.2s, v27.2s   \n"
              "ld1r   {v6.2s},  [%[b]], #4   \n"
              "fmla  v20.2s, v30.2s, v27.2s  \n"
              "fmla  v11.2s, v4.2s, v8.2s    \n"
              "fmla  v19.2s, v31.2s, v8.2s   \n"
              "ld1r   {v7.2s},  [%[b]], #4   \n"
              "fmla  v12.2s, v4.2s, v28.2s   \n"
              "fmla  v20.2s, v31.2s, v28.2s  \n"
              "ld1r   {v8.2s},  [%[b]], #4   \n"

              "fmla  v13.2s,  v1.2s, v5.2s   \n"
              "ld1r   {v25.2s}, [%[b]], #4   \n"
              "fmla  v21.2s, v0.2s, v5.2s    \n"
              "ld1r   {v26.2s}, [%[b]], #4   \n"
              "fmla  v14.2s, v1.2s, v25.2s   \n"
              "fmla  v22.2s, v0.2s, v25.2s   \n"
              "ld1r   {v27.2s}, [%[b]], #4   \n"
              "fmla  v13.2s,  v2.2s, v6.2s   \n"
              "fmla  v21.2s, v29.2s, v6.2s   \n"
              "ld1r   {v28.2s}, [%[b]], #4   \n"
              "fmla  v14.2s, v2.2s, v26.2s   \n"
              "fmla  v22.2s, v29.2s, v26.2s  \n"
              "ld1r   {v5.2s},  [%[b]], #4   \n"
              "fmla  v13.2s, v3.2s, v7.2s    \n"
              "fmla  v21.2s, v30.2s, v7.2s   \n"
              "fmla  v14.2s, v3.2s, v27.2s   \n"
              "ld1r   {v6.2s},  [%[b]], #4   \n"
              "fmla  v22.2s, v30.2s, v27.2s  \n"
              "fmla  v13.2s, v4.2s, v8.2s    \n"
              "fmla  v21.2s, v31.2s, v8.2s   \n"
              "ld1r   {v7.2s},  [%[b]], #4   \n"
              "fmla  v14.2s, v4.2s, v28.2s   \n"
              "fmla  v22.2s, v31.2s, v28.2s  \n"
              "ld1r   {v8.2s},  [%[b]], #4   \n"

              "fmla  v15.2s,  v1.2s, v5.2s   \n"
              "ld1r   {v25.2s}, [%[b]], #4   \n"
              "fmla  v23.2s, v0.2s, v5.2s    \n"
              "fmla  v16.2s, v1.2s, v25.2s   \n"
              "fmla  v24.2s, v0.2s, v25.2s   \n"
              "ld1r   {v26.2s}, [%[b]], #4   \n"
              "fmla  v15.2s,  v2.2s, v6.2s   \n"
              "fmla  v23.2s, v29.2s, v6.2s   \n"
              "fmla  v16.2s, v2.2s, v26.2s   \n"
              "ld1r   {v27.2s}, [%[b]], #4   \n"
              "fmla  v24.2s, v29.2s, v26.2s  \n"
              "ld1   {v1.2s}, [%[a]], #8      \n"
              "fmla  v15.2s, v3.2s, v7.2s     \n"
              "ld1r   {v28.2s}, [%[b]], #4   \n"
              "ld1   {v0.2s}, [%[a]], #8      \n"
              "fmla  v23.2s, v30.2s, v7.2s    \n"
              "ld1r   {v5.2s},  [%[b]], #4     \n"
              "fmla  v16.2s, v3.2s, v27.2s    \n"
              "ld1   {v2.2s}, [%[a]], #8      \n"
              "ld1r   {v6.2s},  [%[b]], #4     \n"
              "fmla  v24.2s, v30.2s, v27.2s   \n"
              "fmla  v15.2s, v4.2s, v8.2s     \n"
              "fmla  v23.2s, v31.2s, v8.2s    \n"
              "ld1r   {v7.2s},  [%[b]], #4     \n"
              "fmla  v16.2s, v4.2s, v28.2s    \n"
              "ld1   {v29.2s},[%[a]], #8      \n"
              "fmla  v24.2s, v31.2s, v28.2s   \n"
              "ld1r   {v8.2s},  [%[b]], #4     \n"

              "subs  %w[cnt], %w[cnt], #1     \n"
              "bne   1b\n"
              "sub  %[b], %[b], #16       \n"
              "sub  %[a], %[a], #32         \n"
              "cbz   %w[relu], 2f           \n"
              "mov   w0,  #0                \n"
              "dup v0.4s, w0                \n"
              "fmax   v9.2s,  v9.2s, v0.2s  \n"
              "fmax  v10.2s, v10.2s, v0.2s  \n"
              "fmax  v11.2s, v11.2s, v0.2s  \n"
              "fmax  v12.2s, v12.2s, v0.2s  \n"
              "fmax  v13.2s, v13.2s, v0.2s  \n"
              "fmax  v14.2s, v14.2s, v0.2s  \n"
              "fmax  v15.2s, v15.2s, v0.2s  \n"
              "fmax  v16.2s, v16.2s, v0.2s  \n"
              "fmax  v17.2s, v17.2s, v0.2s  \n"
              "fmax  v18.2s, v18.2s, v0.2s  \n"
              "fmax  v19.2s, v19.2s, v0.2s  \n"
              "fmax  v20.2s, v20.2s, v0.2s  \n"
              "fmax  v21.2s, v21.2s, v0.2s  \n"
              "fmax  v22.2s, v22.2s, v0.2s  \n"
              "fmax  v23.2s, v23.2s, v0.2s  \n"
              "fmax  v24.2s, v24.2s, v0.2s  \n"
              "2:\n"
              "st1   {v9.2s}, [%[c]], #8   \n"
              "st1   {v17.2s}, [%[c]], #8  \n"
              "st1   {v10.2s}, [%[c]], #8  \n"
              "st1   {v18.2s}, [%[c]], #8  \n"
              "st1   {v11.2s}, [%[c]], #8  \n"
              "st1   {v19.2s}, [%[c]], #8  \n"
              "st1   {v12.2s}, [%[c]], #8  \n"
              "st1   {v20.2s}, [%[c]], #8  \n"
              "st1   {v13.2s}, [%[c]], #8  \n"
              "st1   {v21.2s}, [%[c]], #8  \n"
              "st1   {v14.2s}, [%[c]], #8  \n"
              "st1   {v22.2s}, [%[c]], #8  \n"
              "st1   {v15.2s}, [%[c]], #8  \n"
              "st1   {v23.2s}, [%[c]], #8  \n"
              "st1   {v16.2s}, [%[c]], #8  \n"
              "st1   {v24.2s}, [%[c]], #8  \n"
              : [a] "+r"(ablock_ptr),
                [b] "+r"(bblock),
                [c] "+r"(cblock),
                [cnt] "+r"(cnt)
              : [bias] "r"(bias_h), [relu] "r"(has_relu) 
             : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
            "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
            "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", 
            "v28", "v29", "v30", "v31", "w0", "cc", "memory"
          );
        // clang-format on
      }
      if (has_remain) {
        if (w_loop4 > 0) {
          int cnt = kcnt;
          const float* ablock_ptr = ablock;
          // clang-format off
           asm volatile(
              "prfm pldl1keep, [%[a]]         \n"
              "prfm pldl1keep, [%[b]]         \n"
              "prfm pldl1keep, [%[b], #64]    \n"
              "ld1 {v3.2s}, [%[bias]] \n"
              "mov  v9.16b,   v3.16b    \n"
              "ld1r   {v5.2s},  [%[b]], #4     \n"
              "ld1   {v1.2s}, [%[a]], #8 \n"
              "mov  v10.16b,  v3.16b    \n"
              "ld1   {v0.2s}, [%[a]], #8 \n"
              "ld1r   {v6.2s},  [%[b]], #4  \n"
              "mov  v11.16b,  v3.16b    \n"
              "ld1r   {v7.2s},  [%[b]], #4  \n"
              "mov  v12.16b,  v3.16b    \n"
              "mov  v17.16b,  v3.16b    \n"
              "ld1   {v2.2s}, [%[a]], #8 \n"
              "mov  v18.16b,  v3.16b    \n"
              "ld1r   {v8.2s},  [%[b]], #4  \n"
              "mov  v19.16b,  v3.16b    \n"
              "mov  v20.16b,  v3.16b    \n"
              "ld1   {v29.2s},[%[a]], #8 \n"

              "1:\n"
              "ld1r   {v25.2s}, [%[b]], #4  \n"
              "fmla  v9.2s,  v1.2s, v5.2s   \n"
              "ld1   {v3.2s},  [%[a]], #8   \n"
              "fmla  v10.2s, v1.2s, v25.2s  \n"
              "ld1r   {v26.2s}, [%[b]], #4  \n"
              "fmla  v17.2s, v0.2s, v5.2s   \n"
              "ld1   {v30.2s}, [%[a]], #8   \n"
              "fmla  v18.2s, v0.2s, v25.2s  \n"
              "ld1r   {v27.2s}, [%[b]], #4  \n"
              "fmla  v9.2s,  v2.2s, v6.2s   \n"
              "ld1   {v4.2s},  [%[a]], #8   \n"
              "fmla  v10.2s, v2.2s, v26.2s  \n"
              "ld1r   {v28.2s}, [%[b]], #4  \n"
              "fmla  v17.2s, v29.2s, v6.2s  \n"
              "ld1   {v31.2s}, [%[a]], #8   \n"
              "fmla  v18.2s, v29.2s, v26.2s \n"
              "fmla  v9.2s, v3.2s, v7.2s    \n"
              "fmla  v10.2s, v3.2s, v27.2s  \n"
              "fmla  v17.2s, v30.2s, v7.2s  \n"
              "fmla  v18.2s, v30.2s, v27.2s \n"
              "fmla  v9.2s, v4.2s, v8.2s    \n"
              "ld1r   {v5.2s},  [%[b]], #4  \n"
              "fmla  v10.2s, v4.2s, v28.2s  \n"
              "ld1r   {v6.2s},  [%[b]], #4  \n"
              "fmla  v17.2s, v31.2s, v8.2s  \n"
              "ld1r   {v7.2s},  [%[b]], #4  \n"
              "fmla  v18.2s, v31.2s, v28.2s \n"
              "ld1r   {v8.2s},  [%[b]], #4  \n"

              "ld1r   {v25.2s}, [%[b]], #4  \n"
              "fmla  v11.2s,  v1.2s, v5.2s   \n"
              "ld1r   {v26.2s}, [%[b]], #4  \n"
              "fmla  v12.2s, v1.2s, v25.2s   \n"
              "ld1r   {v27.2s}, [%[b]], #4  \n"
              "fmla  v19.2s, v0.2s, v5.2s    \n"
              "ld1r   {v28.2s}, [%[b]], #4  \n"
              "fmla  v20.2s, v0.2s, v25.2s   \n"
              "fmla  v11.2s,  v2.2s, v6.2s   \n"
              "fmla  v12.2s, v2.2s, v26.2s   \n"
              "fmla  v19.2s, v29.2s, v6.2s   \n"
              "fmla  v20.2s, v29.2s, v26.2s  \n"
              "fmla  v11.2s, v3.2s, v7.2s    \n"
              "ld1   {v1.2s}, [%[a]], #8 \n"
              "fmla  v12.2s, v3.2s, v27.2s   \n"
              "ld1r   {v5.2s},  [%[b]], #4     \n"
              "fmla  v19.2s, v30.2s, v7.2s   \n"
              "ld1   {v0.2s}, [%[a]], #8 \n"
              "fmla  v20.2s, v30.2s, v27.2s  \n"
              "ld1r   {v6.2s},  [%[b]], #4  \n"
              "fmla  v11.2s, v4.2s, v8.2s    \n"
              "ld1   {v2.2s}, [%[a]], #8 \n"
              "fmla  v12.2s, v4.2s, v28.2s   \n"
              "ld1r   {v7.2s},  [%[b]], #4  \n"
              "fmla  v19.2s, v31.2s, v8.2s   \n"
              "ld1   {v29.2s},[%[a]], #8 \n"
              "fmla  v20.2s, v31.2s, v28.2s  \n"
              "ld1r   {v8.2s},  [%[b]], #4  \n"

              "subs  %w[cnt], %w[cnt], #1     \n"
              "bne   1b\n"

              "sub  %[b], %[b], #16       \n"
              "sub  %[a], %[a], #32         \n"
              "cbz   %w[relu], 2f           \n"
              "mov   w0,  #0                \n"
              "dup v0.4s, w0                \n"
              "fmax   v9.2s,  v9.2s, v0.2s  \n"
              "fmax  v10.2s, v10.2s, v0.2s  \n"
              "fmax  v11.2s, v11.2s, v0.2s  \n"
              "fmax  v12.2s, v12.2s, v0.2s  \n"
              "fmax  v17.2s, v17.2s, v0.2s  \n"
              "fmax  v18.2s, v18.2s, v0.2s  \n"
              "fmax  v19.2s, v19.2s, v0.2s  \n"
              "fmax  v20.2s, v20.2s, v0.2s  \n"
              "2:\n"
              "st1   {v9.2s}, [%[c]], #8   \n"
              "st1   {v17.2s}, [%[c]], #8  \n"
              "st1   {v10.2s}, [%[c]], #8  \n"
              "st1   {v18.2s}, [%[c]], #8  \n"
              "st1   {v11.2s}, [%[c]], #8  \n"
              "st1   {v19.2s}, [%[c]], #8  \n"
              "st1   {v12.2s}, [%[c]], #8  \n"
              "st1   {v20.2s}, [%[c]], #8  \n"
              : [a] "+r"(ablock_ptr),
                [b] "+r"(bblock),
                [c] "+r"(cblock),
                [cnt] "+r"(cnt)
              : [bias] "r"(bias_h), [relu] "r"(has_relu) 
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
              "v10", "v11", "v12", "v17", "v18", "v19", "v20", "v25", "v26", "v27",
              "v28", "v29", "v30", "v31", "w0", "cc", "memory"
            );

          // clang-format on
        }
        if (remain > 0) {
          int cnt = kcnt;
          const float* ablock_ptr = ablock;
          // clang-format off
            asm volatile(
                "ld1   {v1.2s}, [%[a]], #8          \n"
                "ld1   {v0.2s}, [%[a]], #8          \n"
                "ld1   {v2.2s}, [%[a]], #8          \n"
                "ld1   {v29.2s},[%[a]], #8          \n"
                "cmp  %w[remain], #3                \n"
                "beq  1f                            \n"
                "cmp  %w[remain], #2                \n"
                "beq  2f                            \n"
                /* remain 1 */
                "ld1 {v5.2s}, [%[bias]]             \n"
                "mov  v9.16b,   v5.16b              \n"
                "mov  v17.16b,  v5.16b              \n"
                "3:                                 \n"
                "ld1r   {v5.2s},  [%[b]], #4        \n"
                "ld1   {v3.2s},  [%[a]], #8         \n"
                "ld1r   {v6.2s},  [%[b]], #4        \n"
                "fmla  v9.2s,  v1.2s, v5.2s         \n"
                "ld1   {v30.2s}, [%[a]], #8         \n"
                "ld1r   {v7.2s},  [%[b]], #4        \n"
                "fmla  v17.2s, v0.2s, v5.2s         \n"
                "ld1   {v4.2s},  [%[a]], #8         \n"
                "ld1r   {v8.2s},  [%[b]], #4        \n"
                "fmla  v9.2s,  v2.2s, v6.2s         \n"
                "ld1   {v31.2s}, [%[a]], #8         \n"
                "fmla  v17.2s, v29.2s, v6.2s        \n"
                "ld1   {v1.2s}, [%[a]], #8          \n"
                "fmla  v9.2s, v3.2s, v7.2s          \n"
                "ld1   {v0.2s}, [%[a]], #8          \n"
                "fmla  v17.2s, v30.2s, v7.2s        \n"
                "ld1   {v2.2s}, [%[a]], #8          \n"
                "fmla  v9.2s, v4.2s, v8.2s          \n"
                "ld1   {v29.2s},[%[a]], #8          \n"
                "fmla  v17.2s, v31.2s, v8.2s        \n"
                "subs  %w[cnt], %w[cnt], #1         \n"
                "bne   3b                           \n"
                "cbz   %w[relu], 6f                 \n"
                "mov   w0,  #0                      \n"
                "dup v0.4s, w0                      \n"
                "fmax   v9.2s,  v9.2s, v0.2s        \n"
                "fmax  v17.2s, v17.2s, v0.2s        \n"
                "6:                                 \n"
                "st1   {v9.2s}, [%[c]], #8          \n"
                "st1   {v17.2s}, [%[c]], #8         \n"
                "b     9f                           \n"
                /* remain 2 */
                "2:                                 \n"
                "ld1 {v5.2s}, [%[bias]]             \n"
                "mov  v9.16b,   v5.16b    \n"
                "mov  v10.16b,  v5.16b    \n"
                "mov  v17.16b,  v5.16b    \n"
                "mov  v18.16b,  v5.16b    \n"
                "4:                                 \n"
                "ld1r   {v5.2s},  [%[b]], #4        \n"
                "ld1r   {v6.2s},  [%[b]], #4        \n"
                "ld1r   {v7.2s},  [%[b]], #4        \n"
                "ld1r   {v8.2s},  [%[b]], #4        \n"
                "fmla  v9.2s,  v1.2s, v5.2s         \n"
                "ld1r   {v25.2s}, [%[b]], #4        \n"
                "ld1   {v3.2s},  [%[a]], #8         \n"
                "fmla  v10.2s, v1.2s, v25.2s        \n"
                "ld1r   {v26.2s}, [%[b]], #4        \n"
                "ld1   {v30.2s}, [%[a]], #8         \n"
                "fmla  v17.2s, v0.2s, v5.2s         \n"
                "ld1r   {v27.2s}, [%[b]], #4        \n"
                "ld1   {v4.2s},  [%[a]], #8         \n"
                "fmla  v18.2s, v0.2s, v25.2s        \n"
                "ld1r   {v28.2s}, [%[b]], #4        \n"
                "ld1   {v31.2s}, [%[a]], #8         \n"
                "fmla  v9.2s,  v2.2s, v6.2s         \n"
                "fmla  v10.2s, v2.2s, v26.2s        \n"
                "fmla  v17.2s, v29.2s, v6.2s        \n"
                "fmla  v18.2s, v29.2s, v26.2s       \n"
                "fmla  v9.2s, v3.2s, v7.2s          \n"
                "fmla  v10.2s, v3.2s, v27.2s        \n"
                "fmla  v17.2s, v30.2s, v7.2s        \n"
                "fmla  v18.2s, v30.2s, v27.2s       \n"
                "ld1   {v1.2s}, [%[a]], #8          \n"
                "fmla  v9.2s, v4.2s, v8.2s          \n"
                "ld1   {v0.2s}, [%[a]], #8          \n"
                "fmla  v10.2s, v4.2s, v28.2s        \n"
                "ld1   {v2.2s}, [%[a]], #8          \n"
                "fmla  v17.2s, v31.2s, v8.2s        \n"
                "ld1   {v29.2s},[%[a]], #8          \n"
                "fmla  v18.2s, v31.2s, v28.2s       \n"
                "subs  %w[cnt], %w[cnt], #1         \n"
                "bne   4b                           \n"
                "cbz   %w[relu], 7f                 \n"
                "mov   w0,  #0                      \n"
                "dup v0.4s, w0                      \n"
                "fmax   v9.2s,  v9.2s, v0.2s        \n"
                "fmax  v10.2s, v10.2s, v0.2s        \n"
                "fmax  v17.2s, v17.2s, v0.2s        \n"
                "fmax  v18.2s, v18.2s, v0.2s        \n"
                "7:                                 \n"
                "st1   {v9.2s}, [%[c]], #8          \n"
                "st1   {v17.2s}, [%[c]], #8         \n"
                "st1   {v10.2s}, [%[c]], #8         \n"
                "st1   {v18.2s}, [%[c]], #8         \n"
                "b     9f                           \n"
                /* remain 3 */
                "1:                       \n"
                "ld1 {v5.2s}, [%[bias]]   \n"
                "mov  v9.16b,   v5.16b    \n" 
                "mov  v10.16b,  v5.16b    \n" 
                "mov  v11.16b,  v5.16b    \n" 
                "mov  v17.16b,  v5.16b    \n" 
                "mov  v18.16b,  v5.16b    \n" 
                "mov  v19.16b,  v5.16b    \n" 
                "5:                                 \n"
                "ld1r   {v5.2s},  [%[b]], #4        \n"
                "ld1r   {v6.2s},  [%[b]], #4        \n"
                "ld1r   {v7.2s},  [%[b]], #4        \n"
                "ld1r   {v8.2s},  [%[b]], #4        \n"
                "fmla  v9.2s,  v1.2s, v5.2s         \n"
                "ld1r   {v25.2s}, [%[b]], #4        \n"
                "ld1   {v3.2s},  [%[a]], #8         \n"
                "fmla  v10.2s, v1.2s, v25.2s        \n"
                "ld1r   {v26.2s}, [%[b]], #4        \n"
                "ld1   {v30.2s}, [%[a]], #8         \n"
                "fmla  v17.2s, v0.2s, v5.2s         \n"
                "ld1r   {v27.2s}, [%[b]], #4        \n"
                "ld1   {v4.2s},  [%[a]], #8         \n"
                "fmla  v18.2s, v0.2s, v25.2s        \n"
                "ld1r   {v28.2s}, [%[b]], #4        \n"
                "ld1   {v31.2s}, [%[a]], #8         \n"
                "fmla  v9.2s,  v2.2s, v6.2s         \n"
                "fmla  v10.2s, v2.2s, v26.2s        \n"
                "fmla  v17.2s, v29.2s, v6.2s        \n"
                "fmla  v18.2s, v29.2s, v26.2s       \n"
                "fmla  v9.2s, v3.2s, v7.2s          \n"
                "fmla  v10.2s, v3.2s, v27.2s        \n"
                "ld1r   {v5.2s},  [%[b]], #4        \n"
                "fmla  v17.2s, v30.2s, v7.2s        \n"
                "fmla  v18.2s, v30.2s, v27.2s       \n"
                "fmla  v9.2s, v4.2s, v8.2s          \n"
                "ld1r   {v6.2s},  [%[b]], #4        \n"
                "fmla  v10.2s, v4.2s, v28.2s        \n"
                "fmla  v17.2s, v31.2s, v8.2s        \n"
                "fmla  v18.2s, v31.2s, v28.2s       \n"
                "ld1r   {v7.2s},  [%[b]], #4        \n"
                "fmla  v11.2s,  v1.2s, v5.2s        \n"
                "fmla  v19.2s, v0.2s, v5.2s         \n"
                "fmla  v11.2s,  v2.2s, v6.2s        \n"
                "ld1r   {v8.2s},  [%[b]], #4        \n"
                "fmla  v19.2s, v29.2s, v6.2s        \n"
                "ld1   {v1.2s}, [%[a]], #8          \n"
                "fmla  v11.2s, v3.2s, v7.2s         \n"
                "ld1   {v0.2s}, [%[a]], #8          \n"
                "fmla  v19.2s, v30.2s, v7.2s        \n"
                "ld1   {v2.2s}, [%[a]], #8          \n"
                "fmla  v11.2s, v4.2s, v8.2s         \n"
                "ld1   {v29.2s},[%[a]], #8          \n"
                "fmla  v19.2s, v31.2s, v8.2s        \n"
                "subs  %w[cnt], %w[cnt], #1         \n"
                "bne   5b                           \n"
                "cbz   %w[relu], 8f                 \n"
                "mov   w0,  #0                      \n"
                "dup v0.4s, w0                      \n"
                "fmax   v9.2s,  v9.2s, v0.2s  \n"
                "fmax  v10.2s, v10.2s, v0.2s  \n"
                "fmax  v11.2s, v11.2s, v0.2s  \n"
                "fmax  v17.2s, v17.2s, v0.2s  \n"
                "fmax  v18.2s, v18.2s, v0.2s  \n"
                "fmax  v19.2s, v19.2s, v0.2s  \n"
                "8:                           \n"
                "st1   {v9.2s}, [%[c]], #8   \n"
                "st1   {v17.2s}, [%[c]], #8  \n"
                "st1   {v10.2s}, [%[c]], #8  \n"
                "st1   {v18.2s}, [%[c]], #8  \n"
                "st1   {v11.2s}, [%[c]], #8  \n"
                "st1   {v19.2s}, [%[c]], #8  \n"
                "9:\n"
                : [a] "+r"(ablock_ptr),
                  [b] "+r"(bblock),
                  [c] "+r"(cblock),
                  [cnt] "+r"(cnt)
                : [bias] "r"(bias_h), [relu] "r"(has_relu), 
                  [remain] "r"(remain)
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                "v10", "v11", "v12", "v17", "v18", "v19", "v20", "v25", "v26", "v27",
                "v28", "v29", "v30", "v31", "w0", "cc", "memory"
            );
          // clang-format on
        }
      }
    }
    LITE_PARALLEL_END();
  }
#else
  for (int n = 0; n < bchunk_loop; ++n) {
    int x_start = n * bchunk_w;
    int x_end = x_start + bchunk_w;
    int w_loop = bchunk_w / NBLOCK_C4;
    int flag_remain = 0;
    int w_loop4 = 0;
    int remain = 0;
    if (x_end > N) {
      w_loop = (N - x_start) / NBLOCK_C4;
      int w_loop_rem = (N - x_start) - w_loop * NBLOCK_C4;
      w_loop4 = w_loop_rem >> 2;
      remain = w_loop_rem & 3;
      x_end = N;
      flag_remain = 1;
    }
    float* bchunk = workspace;
    loadb_c4(bchunk, B, x_start, x_end, k_round, N);
    float* cchunk = c + n * bchunk_w * 4;
    int has_remain = (n == bchunk_loop - 1) && flag_remain;
    LITE_PARALLEL_BEGIN(h, tid, h_loop) {
      float* bias_h = bias_buf + h * 4;
      const float* ablock = A_packed + h * lda;
      const float* bblock = bchunk;
      float* cblock = cchunk + h * ldc;
      for (int w = 0; w < w_loop; ++w) {
        int cnt = kcnt;
        const float* ablock_ptr = ablock;
        // clang-format off
        asm volatile(
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
            "vst1.32   {d16-d19}, [%[c]]! \n"
            "vst1.32   {d20-d23}, [%[c]]! \n"
            "vst1.32   {d24-d27}, [%[c]]! \n"
            "vst1.32   {d28-d31}, [%[c]]! \n"
            : [a] "+r"(ablock_ptr),
              [b] "+r"(bblock),
              [c] "+r"(cblock),
              [cnt] "+r"(cnt)
            : [bias] "r"(bias_h), 
              [relu] "r"(has_relu)
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
              "q9", "q10", "q11", "q12", "q13", "q14", "q15", "cc", "memory");
        // clang-format on
      }
      if (has_remain) {
        if (w_loop4 > 0) {
          int cnt = kcnt;
          const float* ablock_ptr = ablock;
          // clang-format off
          asm volatile(
            "pld [%[a]]  \n"
            "pld [%[b]]  \n"
            "vld1.32  {d6-d7}, [%[bias]] \n"
            "vld1.32  {d0-d3}, [%[a]]!   \n" /* load a0 a1 */
            "vmov.32  q8,   q3   \n"     /* mov bias to c0 */
            "vmov.32  q9,   q3   \n"     /* mov bias to c1 */
            "vmov.32  q10,  q3   \n"     /* mov bias to c2 */
            "vmov.32  q11,  q3   \n"     /* mov bias to c3 */
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
            "pld [%[a]]                   \n"
            "vmla.f32  q8,  q1, d8[1]     \n"
            "vmla.f32  q9,  q1, d10[1]    \n"
            "vmla.f32  q10, q1, d12[1]    \n"
            "vmla.f32  q11, q1, d14[1]    \n"
            "subs  %[cnt], %[cnt], #1     \n"
            "vmla.f32  q8,  q2, d9[0]     \n"
            "vmla.f32  q9,  q2, d11[0]    \n"
            "vmla.f32  q10, q2, d13[0]    \n"
            "vmla.f32  q11, q2, d15[0]    \n"
            "vld1.32   {d0-d3}, [%[a]]!   \n"
            "vmla.f32  q8,  q3, d9[1]     \n"
            "vmla.f32  q9,  q3, d11[1]    \n"
            "vmla.f32  q10, q3, d13[1]    \n"
            "vmla.f32  q11, q3, d15[1]    \n"
            "bne   1b\n"
            "cmp   %[relu], #0            \n"
            "beq   2f                     \n"
            "vmov.u32  q0, #0             \n"
            "vmax.f32  q8,   q8,   q0     \n"
            "vmax.f32  q9,   q9,   q0     \n"
            "vmax.f32  q10,  q10,  q0     \n"
            "vmax.f32  q11,  q11,  q0     \n"
            "2:\n"
            "vst1.32   {d16-d19}, [%[c]]! \n"
            "vst1.32   {d20-d23}, [%[c]]! \n"
            : [a] "+r"(ablock_ptr),
              [b] "+r"(bblock),
              [c] "+r"(cblock),
              [cnt] "+r"(cnt)
            : [bias] "r"(bias_h), [relu] "r"(has_relu)
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
              "q9", "q10", "q11", "cc", "memory");
          // clang-format on
        }
        if (remain > 0) {
          int cnt = kcnt;
          const float* ablock_ptr = ablock;
          // clang-format off
          asm volatile(
              "pld  [%[a]]  \n"
              "pld  [%[b]]  \n"
              "vld1.32  {d0-d1}, [%[bias]]  \n"
              "vld1.32  {d2-d5}, [%[a]]!    \n"
              "vmov.u32 q15,  #0            \n"
              "cmp  %[remain], #3           \n"
              "beq  1f                      \n"
              "cmp  %[remain], #2           \n"
              "beq  2f                      \n"
              /* remain 1 */
              "vmov.32  q9,   q0  \n" /* mov bias to c0*/
              "vmov.32  q10,  q15 \n" /* mov zero to c1*/
              "3:                             \n"
              "vld1.32   {d10-d11}, [%[b]]!   \n"
              "vld1.32   {d6-d9},   [%[a]]!   \n"
              "vmla.f32  q9,  q1,  d10[0]     \n"
              "vmla.f32  q10, q2,  d10[1]     \n"
              "subs   %[cnt],  %[cnt], #1     \n"
              "vld1.32   {d2-d5},   [%[a]]!   \n"
              "vmla.f32  q9,  q3,  d11[0]     \n"
              "vmla.f32  q10, q4,  d11[1]     \n"
              "bne   3b                       \n"
              "vadd.f32  q9,  q9,  q10        \n"
              "cmp  %[relu],  #0              \n"
              "beq  6f                        \n"
              "vmax.f32  q9,  q9,  q15        \n"
              "6:                             \n"
              "vst1.32   {d18-d19}, [%[c]]!   \n"
              "b     9f                       \n"
              /* remain 2 */
              "2:                             \n"
              "vmov.u32  q9,  q0    \n" /* mov bias to c0*/
              "vmov.u32  q10, q0    \n" /* mov bias to c1*/
              "vmov.u32  q11, q15   \n" /* mov zero to c2*/
              "vmov.u32  q12, q15   \n" /* mov zero to c3*/
              "4:                             \n"
              "vld1.32   {d10-d13}, [%[b]]!   \n"
              "vld1.32   {d6-d9},   [%[a]]!   \n"
              "vmla.f32  q9,   q1,  d10[0]    \n"
              "vmla.f32  q10,  q1,  d12[0]    \n"
              "vmla.f32  q11,  q2,  d10[1]    \n"
              "vmla.f32  q12,  q2,  d12[1]    \n"
              "subs  %[cnt],  %[cnt], #1      \n"
              "vmla.f32  q9,   q3,  d11[0]    \n"
              "vmla.f32  q10,  q3,  d13[0]    \n"
              "vmla.f32  q11,  q4,  d11[1]    \n"
              "vmla.f32  q12,  q4,  d13[1]    \n"
              "vld1.32   {d2-d5},   [%[a]]!   \n"
              "bne   4b                       \n"
              "vadd.f32  q9,   q9,  q11       \n"
              "vadd.f32  q10,  q10, q12       \n"
              "cmp  %[relu],  #0              \n"
              "beq  7f                        \n"
              "vmax.f32  q9,   q9,  q15       \n"
              "vmax.f32  q10,  q10, q15       \n"
              "7:                             \n"
              "vst1.32   {d18-d21}, [%[c]]!   \n"
              "b     9f                       \n"
              /* remain 3 */
              "1:                             \n"
              "vmov.u32  q9,   q0    \n" /* mov bias to c0*/
              "vmov.u32  q10,  q0    \n" /* mov bias to c1*/
              "vmov.u32  q11,  q0    \n" /* mov bias to c2*/
              "5:                             \n"
              "vld1.32   {d10-d13}, [%[b]]!   \n"
              "vld1.32   {d14-d15}, [%[b]]!   \n"
              "vmla.f32  q9,  q1,   d10[0]    \n"
              "vmla.f32  q10, q1,   d12[0]    \n"
              "vmla.f32  q11, q1,   d14[0]    \n"
              "vld1.32   {d6-d9},   [%[a]]!   \n"
              "vmla.f32  q9,  q2,  d10[1]     \n"
              "vmla.f32  q10, q2,  d12[1]     \n"
              "vmla.f32  q11, q2,  d14[1]     \n"
              "subs  %[cnt],  %[cnt], #1      \n"
              "vmla.f32  q9,  q3,  d11[0]     \n"
              "vmla.f32  q10, q3,  d13[0]     \n"
              "vmla.f32  q11, q3,  d15[0]     \n"
              "pld       [%[a]]               \n"
              "vmla.f32  q9,  q4,  d11[1]     \n"
              "vmla.f32  q10, q4,  d13[1]     \n"
              "vmla.f32  q11, q4,  d15[1]     \n"
              "vld1.32   {d2-d5},  [%[a]]!    \n"
              "bne   5b                       \n"
              "cmp  %[relu],  #0              \n"
              "beq  8f                        \n"
              "vmax.f32  q9,  q9,  q15        \n"
              "vmax.f32  q10, q10, q15        \n"
              "vmax.f32  q11, q11, q15        \n"
              "8:                             \n"
              "vst1.32   {d18-d21}, [%[c]]!   \n"
              "vst1.32   {d22-d23}, [%[c]]!   \n"
              "9:\n"
              : [a] "+r"(ablock_ptr),
                [b] "+r"(bblock),
                [c] "+r"(cblock),
                [cnt] "+r"(cnt)
              : [bias] "r"(bias_h), 
                [relu] "r"(has_relu), 
                [remain] "r"(remain)
              : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q9", 
                "q10", "q11", "q12", "q15", "cc","memory");
          // clang-format on
        }
      }
    }
    LITE_PARALLEL_END();
  }
#endif
}

void sgemm_prepack_c4_small_a35(int M,
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
  const int mloop = m_round >> 2;
  const int lda = 4 * k_round;
  const int ldb_byte = 4 * N * sizeof(float);
  const int kcnt = k_round >> 2;
  float bias_buf[m_round];  // NOLINT
  if (has_bias) {
    lite::TargetWrapperHost::MemcpySync(bias_buf, bias, M * sizeof(float));
    memset(bias_buf + M, 0, (m_round - M) * sizeof(float));
  } else {
    memset(bias_buf, 0, m_round * sizeof(float));
  }
#ifdef __aarch64__
  float32x4_t vzero = vdupq_n_f32(0.f);
  const float* bias_ptr = bias_buf;
  for (int m = 0; m < mloop; ++m) {
    float32x4_t vbias = vld1q_f32(bias_ptr);
    const float* b = B;
    int n = N;
    // clang-format off
    for (; n > 7; n -= 8) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
         asm volatile(
          "ld1  {v3.4s},  [%[bias_ptr]] \n"
          "mov  v8.16b,   v3.16b        \n"
          "ld1  {v16.2s}, [%[a]], #8    \n"
          "mov  v9.16b,   v3.16b        \n"
          "mov  v10.16b,  v3.16b        \n"
          "ld1r  {v0.2s},  [%[b]], #4   \n"
          "mov  v11.16b,  v3.16b        \n"
          "mov  v12.16b,  v3.16b        \n"
          "ld1  {v28.2s}, [%[a]], #8    \n"
          "mov  v13.16b,  v3.16b        \n"
          "mov  v14.16b,  v3.16b        \n"
          "ld1r  {v4.2s},  [%[b]], #4   \n"
          "mov  v15.16b,  v3.16b        \n"
          "mov  v20.16b,  v3.16b        \n"
          "ld1  {v17.2s}, [%[a]], #8    \n"
          "mov  v21.16b,  v3.16b        \n"
          "mov  v22.16b,  v3.16b        \n"
          "ld1r  {v1.2s},  [%[b]], #4   \n"
          "mov  v23.16b,  v3.16b        \n"
          "mov  v24.16b,  v3.16b        \n"
          "ld1  {v29.2s}, [%[a]], #8    \n"
          "mov  v25.16b,  v3.16b        \n"
          "mov  v26.16b,  v3.16b        \n"
          "ld1r  {v5.2s},  [%[b]], #4   \n"
          "mov  v27.16b,  v3.16b        \n"

          "1:\n"
          "ld1r  {v2.2s},  [%[b]], #4  \n"
          "fmla v8.2s,  v16.2s, v0.2s  \n"
          "fmla v20.2s, v28.2s, v0.2s  \n"
          "ld1  {v18.2s}, [%[a]], #8   \n"
          "fmla v9.2s,  v16.2s, v2.2s  \n"
          "ld1r  {v6.2s},  [%[b]], #4  \n"
          "fmla v21.2s, v28.2s, v2.2s  \n"
          "ld1  {v30.2s}, [%[a]], #8   \n"
          "fmla v8.2s,   v17.2s, v4.2s \n"
          "ld1r  {v3.2s},  [%[b]], #4  \n"
          "fmla v20.2s,  v29.2s, v4.2s \n"
          "ld1  {v19.2s}, [%[a]], #8   \n"
          "fmla v9.2s,   v17.2s, v6.2s \n"
          "ld1r  {v7.2s},  [%[b]], #4  \n"
          "fmla v21.2s,  v29.2s, v6.2s \n"
          "ld1  {v31.2s}, [%[a]], #8   \n"
          "fmla v8.2s,   v18.2s, v1.2s \n"
          "ld1r  {v0.2s},  [%[b]], #4  \n"
          "fmla v20.2s,  v30.2s, v1.2s \n"
          "fmla v9.2s,   v18.2s, v3.2s \n"
          "fmla v21.2s,  v30.2s, v3.2s \n"
          "ld1r  {v4.2s},  [%[b]], #4 \n"
          "fmla v8.2s,  v19.2s, v5.2s \n"
          "fmla v20.2s,  v31.2s,v5.2s \n"
          "ld1r  {v1.2s},  [%[b]], #4 \n"
          "fmla v9.2s,  v19.2s, v7.2s \n"
          "fmla v21.2s,  v31.2s,v7.2s \n"
          "ld1r  {v5.2s},  [%[b]], #4 \n"

          "ld1r  {v2.2s},  [%[b]], #4 \n"
          "fmla v10.2s, v16.2s, v0.2s \n"
          "fmla v22.2s, v28.2s, v0.2s \n"
          "ld1r  {v6.2s},  [%[b]], #4 \n"
          "fmla v11.2s, v16.2s, v2.2s \n"
          "fmla v23.2s, v28.2s, v2.2s \n"
          "fmla v10.2s,  v17.2s, v4.2s \n"
          "ld1r  {v3.2s},  [%[b]], #4  \n"
          "fmla v22.2s,  v29.2s, v4.2s \n"
          "fmla v11.2s,  v17.2s, v6.2s \n"
          "ld1r  {v7.2s},  [%[b]], #4  \n"
          "fmla v23.2s,  v29.2s, v6.2s \n"
          "fmla v10.2s,  v18.2s, v1.2s \n"
          "ld1r  {v0.2s},  [%[b]], #4  \n"
          "fmla v22.2s,  v30.2s, v1.2s \n"
          "fmla v11.2s,  v18.2s, v3.2s \n"
          "ld1r  {v4.2s},  [%[b]], #4  \n"
          "fmla v23.2s,  v30.2s, v3.2s \n"
          "fmla v10.2s,  v19.2s, v5.2s \n"
          "ld1r  {v1.2s},  [%[b]], #4  \n"
          "fmla v22.2s,  v31.2s, v5.2s \n"
          "fmla v11.2s,  v19.2s, v7.2s \n"
          "fmla v23.2s,  v31.2s, v7.2s \n"
          "ld1r  {v5.2s},  [%[b]], #4  \n"

          "ld1r  {v2.2s},  [%[b]], #4 \n"
          "fmla v12.2s, v16.2s, v0.2s \n"
          "fmla v24.2s, v28.2s, v0.2s \n"
          "ld1r  {v6.2s},  [%[b]], #4 \n"
          "fmla v13.2s, v16.2s, v2.2s \n"
          "fmla v25.2s, v28.2s, v2.2s \n"
          "ld1r  {v3.2s},  [%[b]], #4 \n"
          "fmla v12.2s, v17.2s, v4.2s \n"
          "fmla v24.2s, v29.2s, v4.2s \n"
          "ld1r  {v7.2s},  [%[b]], #4 \n"
          "fmla v13.2s, v17.2s, v6.2s \n"
          "fmla v25.2s, v29.2s, v6.2s \n"
          "ld1r  {v0.2s},  [%[b]], #4 \n"
          "fmla v12.2s, v18.2s, v1.2s \n"
          "fmla v24.2s, v30.2s, v1.2s \n"
          "ld1r  {v4.2s},  [%[b]], #4 \n"
          "fmla v13.2s, v18.2s, v3.2s \n"
          "fmla v25.2s, v30.2s, v3.2s \n"
          "fmla v12.2s, v19.2s, v5.2s \n"
          "ld1r  {v1.2s},  [%[b]], #4 \n"
          "fmla v24.2s, v31.2s, v5.2s \n"
          "fmla v13.2s, v19.2s, v7.2s \n"
          "fmla v25.2s, v31.2s, v7.2s \n"
          "ld1r  {v5.2s},  [%[b]], #4 \n"

          "ld1r  {v2.2s},  [%[b]], #4 \n"
          "fmla v14.2s, v16.2s, v0.2s \n"
          "fmla v26.2s, v28.2s, v0.2s \n"
          "ld1r  {v6.2s},  [%[b]], #4 \n"
          "fmla v15.2s, v16.2s, v2.2s \n"
          "fmla v27.2s, v28.2s, v2.2s \n"
          "ld1r  {v3.2s},  [%[b]], #4 \n"
          "fmla v14.2s, v17.2s, v4.2s \n"
          "fmla v26.2s, v29.2s, v4.2s \n"
          "ld1r  {v7.2s},  [%[b]], #4 \n"
          "fmla v15.2s, v17.2s, v6.2s \n"
          "ld1  {v16.2s}, [%[a]], #8  \n"
          "sub  %[b],   %[b],   #128  \n"
          "fmla v27.2s, v29.2s, v6.2s \n"
          "add  %[b],   %[b],   %[ldb]\n"
          "fmla v14.2s, v18.2s, v1.2s \n"
          "ld1  {v28.2s}, [%[a]], #8  \n"
          "ld1r  {v0.2s},  [%[b]], #4 \n"
          "fmla v26.2s, v30.2s, v1.2s \n"
          "fmla v15.2s, v18.2s, v3.2s \n"
          "ld1  {v17.2s}, [%[a]], #8  \n"
          "ld1r  {v4.2s},  [%[b]], #4 \n"
          "fmla v27.2s, v30.2s, v3.2s \n"
          "fmla v14.2s, v19.2s, v5.2s \n"
          "ld1  {v29.2s}, [%[a]], #8  \n"
          "ld1r  {v1.2s},  [%[b]], #4 \n"
          "fmla v26.2s, v31.2s, v5.2s \n"
          "fmla v15.2s, v19.2s, v7.2s \n"
          "fmla v27.2s, v31.2s, v7.2s \n"
          "subs %w[cnt], %w[cnt], #1  \n"
          "ld1r  {v5.2s},  [%[b]], #4 \n"

          "bne  1b                    \n"
          "sub  %[b], %[b], #16       \n"
          "sub  %[a], %[a], #32       \n"

          "cbz  %w[relu], 2f            \n"
          "mov w0, #0                   \n"
          "dup v0.2s, w0          \n"
          "fmax v8.2s,  v8.2s,  v0.2s \n"
          "fmax v9.2s,  v9.2s,  v0.2s \n"
          "fmax v10.2s, v10.2s, v0.2s \n"
          "fmax v11.2s, v11.2s, v0.2s \n"
          "fmax v12.2s, v12.2s, v0.2s \n"
          "fmax v13.2s, v13.2s, v0.2s \n"
          "fmax v14.2s, v14.2s, v0.2s \n"
          "fmax v15.2s, v15.2s, v0.2s \n"
          "fmax v20.2s, v20.2s, v0.2s \n"
          "fmax v21.2s, v21.2s, v0.2s \n"
          "fmax v22.2s, v22.2s, v0.2s \n"
          "fmax v23.2s, v23.2s, v0.2s \n"
          "fmax v24.2s, v24.2s, v0.2s \n"
          "fmax v25.2s, v25.2s, v0.2s \n"
          "fmax v26.2s, v26.2s, v0.2s \n"
          "fmax v27.2s, v27.2s, v0.2s \n"
          "2:\n"
          "st1  {v8.2s }, [%[c]], #8 \n"
          "st1  {v20.2s}, [%[c]], #8 \n"
          "st1  {v9.2s }, [%[c]], #8 \n"
          "st1  {v21.2s}, [%[c]], #8 \n"
          "st1  {v10.2s}, [%[c]], #8 \n"
          "st1  {v22.2s}, [%[c]], #8 \n"
          "st1  {v11.2s}, [%[c]], #8 \n"
          "st1  {v23.2s}, [%[c]], #8 \n"
          "st1  {v12.2s}, [%[c]], #8 \n"
          "st1  {v24.2s}, [%[c]], #8 \n"
          "st1  {v13.2s}, [%[c]], #8 \n"
          "st1  {v25.2s}, [%[c]], #8 \n"
          "st1  {v14.2s}, [%[c]], #8 \n"
          "st1  {v26.2s}, [%[c]], #8 \n"
          "st1  {v15.2s}, [%[c]], #8 \n"
          "st1  {v27.2s}, [%[c]], #8 \n"
          : [a] "+r" (a_ptr),
            [b] "+r" (b_ptr),
            [c] "+r" (C),
            [cnt] "+r" (cnt)
          : [relu] "r" (has_relu),
            [ldb]  "r" (ldb_byte),
            [bias_ptr] "r" (bias_ptr)
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
            "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
            "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", 
            "v28", "v29", "v30", "v31", "w0", "cc", "memory"
        );
      b += 4 * 8;
    }
    for (; n > 3; n -= 4) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
        asm volatile(
          "ld1  {v3.4s},  [%[bias_ptr]] \n"
          "mov  v8.16b,   v3.16b     \n"
          "ld1r  {v0.2s},  [%[b]], #4\n"
          "mov  v9.16b,   v3.16b     \n"
          "ld1  {v16.2s}, [%[a]], #8 \n"
          "mov  v10.16b,  v3.16b     \n"
          "ld1r  {v4.2s},  [%[b]], #4\n"
          "mov  v11.16b,  v3.16b     \n"
          "ld1  {v28.2s}, [%[a]], #8 \n"
          "mov  v20.16b,  v3.16b     \n"
          "ld1r  {v1.2s},  [%[b]], #4\n"
          "mov  v21.16b,  v3.16b     \n"
          "ld1  {v17.2s}, [%[a]], #8 \n"
          "mov  v22.16b,  v3.16b     \n"
          "ld1r  {v5.2s},  [%[b]], #4\n"
          "mov  v23.16b,  v3.16b     \n"
          "ld1  {v29.2s}, [%[a]], #8 \n"

          "1:\n"
          "fmla v8.2s,  v16.2s, v0.2s \n"
          "ld1r  {v2.2s},  [%[b]], #4 \n"
          "fmla v20.2s, v28.2s, v0.2s \n"
          "ld1  {v18.2s}, [%[a]], #8  \n"
          "fmla v9.2s,  v16.2s, v2.2s \n"
          "ld1r  {v6.2s},  [%[b]], #4 \n"
          "fmla v21.2s, v28.2s, v2.2s \n"
          "ld1  {v30.2s}, [%[a]], #8  \n"
          "fmla v8.2s,   v17.2s, v4.2s\n"
          "ld1r  {v3.2s},  [%[b]], #4 \n"
          "fmla v20.2s,  v29.2s, v4.2s\n"
          "ld1  {v19.2s}, [%[a]], #8  \n"
          "fmla v9.2s,   v17.2s, v6.2s\n"
          "ld1r  {v7.2s},  [%[b]], #4 \n"
          "fmla v21.2s,  v29.2s, v6.2s\n"
          "ld1  {v31.2s}, [%[a]], #8  \n"
          "ld1r  {v0.2s},  [%[b]], #4 \n"
          "fmla v8.2s,   v18.2s, v1.2s\n"
          "fmla v20.2s,  v30.2s, v1.2s\n"
          "ld1r  {v4.2s},  [%[b]], #4 \n"
          "fmla v9.2s,   v18.2s, v3.2s\n"
          "fmla v21.2s,  v30.2s, v3.2s\n"
          "ld1r  {v1.2s},  [%[b]], #4 \n"
          "fmla v8.2s,  v19.2s, v5.2s \n"
          "fmla v20.2s,  v31.2s,v5.2s \n"
          "fmla v9.2s,  v19.2s, v7.2s \n"
          "fmla v21.2s,  v31.2s,v7.2s \n"
          "ld1r  {v5.2s},  [%[b]], #4 \n"

          "ld1r  {v2.2s},  [%[b]], #4 \n"
          "fmla v10.2s, v16.2s, v0.2s \n"
          "fmla v22.2s, v28.2s, v0.2s \n"
          "ld1r  {v6.2s},  [%[b]], #4 \n"
          "fmla v11.2s, v16.2s, v2.2s \n"
          "fmla v23.2s, v28.2s, v2.2s \n"
          "ld1r  {v3.2s},  [%[b]], #4 \n"
          "fmla v10.2s,  v17.2s, v4.2s\n"
          "fmla v22.2s,  v29.2s, v4.2s\n"
          "ld1r  {v7.2s},  [%[b]], #4 \n"
          "fmla v11.2s,  v17.2s, v6.2s\n"
          "fmla v23.2s,  v29.2s, v6.2s\n"
          "sub  %[b],   %[b],   #64   \n"
          "fmla v10.2s,  v18.2s, v1.2s\n"
          "add  %[b],   %[b],   %[ldb]\n"
          "fmla v22.2s,  v30.2s, v1.2s\n"
          "ld1r  {v0.2s},  [%[b]], #4 \n"
          "fmla v11.2s,  v18.2s, v3.2s\n"
          "ld1  {v16.2s}, [%[a]], #8  \n"
          "fmla v23.2s,  v30.2s, v3.2s\n"
          "ld1  {v28.2s}, [%[a]], #8  \n"
          "ld1r  {v4.2s},  [%[b]], #4 \n"
          "fmla v10.2s,  v19.2s, v5.2s\n"
          "ld1  {v17.2s}, [%[a]], #8  \n"
          "fmla v22.2s,  v31.2s, v5.2s\n"
          "ld1r  {v1.2s},  [%[b]], #4 \n"
          "ld1  {v29.2s}, [%[a]], #8  \n"
          "fmla v11.2s,  v19.2s, v7.2s\n"
          "fmla v23.2s,  v31.2s, v7.2s\n"
          "ld1r  {v5.2s},  [%[b]], #4 \n"

          "subs %w[cnt], %w[cnt], #1  \n"
          "bne  1b                    \n"
          "sub  %[b], %[b], #16       \n"
          "sub  %[a], %[a], #32       \n"
          "cbz  %w[relu], 2f          \n"
          "mov w0, #0                 \n"
          "dup v0.2s, w0              \n"
          "fmax v8.2s,  v8.2s,  v0.2s \n"
          "fmax v9.2s,  v9.2s,  v0.2s \n"
          "fmax v10.2s, v10.2s, v0.2s \n"
          "fmax v11.2s, v11.2s, v0.2s \n"
          "fmax v20.2s, v20.2s, v0.2s \n"
          "fmax v21.2s, v21.2s, v0.2s \n"
          "fmax v22.2s, v22.2s, v0.2s \n"
          "fmax v23.2s, v23.2s, v0.2s \n"
          "2:\n"
          "st1  {v8.2s }, [%[c]], #8 \n"
          "st1  {v20.2s}, [%[c]], #8 \n"
          "st1  {v9.2s }, [%[c]], #8 \n"
          "st1  {v21.2s}, [%[c]], #8 \n"
          "st1  {v10.2s}, [%[c]], #8 \n"
          "st1  {v22.2s}, [%[c]], #8 \n"
          "st1  {v11.2s}, [%[c]], #8 \n"
          "st1  {v23.2s}, [%[c]], #8 \n"
          : [a] "+r" (a_ptr),
            [b] "+r" (b_ptr),
            [c] "+r" (C),
            [cnt] "+r" (cnt)
          : [relu] "r" (has_relu),
            [ldb]  "r" (ldb_byte),
            [bias_ptr] "r" (bias_ptr)
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
            "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
            "v28", "v29", "v30", "v31", "w0", "cc", "memory"
        );
      b += 4 * 4;
    }
    for (; n > 0; n--) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
        asm volatile(
          "ld1  {v0.4s},  [%[bias_ptr]] \n"
          "mov  v8.16b,   v0.16b \n"
          "mov  v20.16b,  v0.16b \n"
          "ld1r  {v0.2s},  [%[b]], #4  \n"
          "ld1  {v16.2s}, [%[a]], #8   \n"
          "ld1r  {v4.2s},  [%[b]], #4  \n"
          "ld1  {v28.2s}, [%[a]], #8   \n"
          "ld1r  {v1.2s},  [%[b]], #4  \n"
          "ld1  {v17.2s}, [%[a]], #8   \n"
          "ld1r  {v5.2s},  [%[b]], #4  \n"
          "ld1  {v29.2s}, [%[a]], #8   \n"
          "1:\n"
          "ld1  {v18.2s}, [%[a]], #8   \n"
          "fmla v8.2s,  v16.2s, v0.2s  \n"
          "ld1  {v30.2s}, [%[a]], #8   \n"
          "fmla v20.2s, v28.2s, v0.2s  \n"
          "ld1  {v19.2s}, [%[a]], #8   \n"
          "fmla v8.2s,   v17.2s, v4.2s \n"
          "ld1  {v31.2s}, [%[a]], #8   \n"
          "fmla v20.2s,  v29.2s, v4.2s \n"
          "sub  %[b],   %[b],   #16    \n"
          "fmla v8.2s,   v18.2s, v1.2s \n"
          "add  %[b],   %[b],   %[ldb] \n"
          "fmla v20.2s,  v30.2s, v1.2s \n"
          "ld1r  {v0.2s},  [%[b]], #4  \n"
          "fmla v8.2s,  v19.2s, v5.2s  \n"
          "ld1  {v16.2s}, [%[a]], #8   \n"
          "fmla v20.2s,  v31.2s,v5.2s  \n"
          "ld1r  {v4.2s},  [%[b]], #4  \n"
          "ld1  {v28.2s}, [%[a]], #8   \n"
          "ld1r  {v1.2s},  [%[b]], #4  \n"
          "ld1  {v17.2s}, [%[a]], #8   \n"
          "ld1r  {v5.2s},  [%[b]], #4  \n"
          "ld1  {v29.2s}, [%[a]], #8   \n"
          "subs %w[cnt], %w[cnt], #1   \n"
          "bne  1b                     \n"
          "cbz  %w[relu], 2f           \n"
          "mov w0, #0                  \n"
          "dup v0.2s, w0          \n"
          "fmax v8.2s,  v8.2s,  v0.2s \n"
          "fmax v20.2s, v20.2s, v0.2s \n"
          "2:\n"
          "st1  {v8.2s }, [%[c]], #8 \n"
          "st1  {v20.2s}, [%[c]], #8 \n"
          : [a] "+r" (a_ptr),
            [b] "+r" (b_ptr),
            [c] "+r" (C),
            [cnt] "+r" (cnt)
          : [relu] "r" (has_relu),
            [ldb]  "r" (ldb_byte),
            [bias_ptr] "r" (bias_ptr)
          : "v0", "v8", "v9", "v16", "v17", 
            "v18", "v19", "cc", "memory"
        );
      b += 4;
    }
    bias_ptr += 4;
    A_packed += lda;
  }
// clang-format on
#else
  const float* bias_ptr = bias_buf;
  for (int m = 0; m < mloop; ++m) {
    const float* b = B;
    int n = N;
    for (; n > 7; n -= 8) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
      // clang-format off
      asm volatile(
        "vld1.32  {d6-d7}, [%[bias]] \n"
        /* load a0, a1 */
        "vld1.32  {d8-d11}, [%[a]]!  \n"
        /* mov bias to c0-c7*/
        "vmov.u32   q8,    q3 \n"
        "vmov.u32   q9,    q3 \n"
        "vmov.u32   q10,   q3 \n"
        "vmov.u32   q11,   q3 \n"
        /* load b0, b1 */
        "vld1.32  {d0-d3}, [%[b]]! \n"
        "vmov.u32   q12,   q3 \n"
        "vmov.u32   q13,   q3 \n"
        "vmov.u32   q14,   q3 \n"
        "vmov.u32   q15,   q3 \n"
        "1:\n"
        /* load b2, b3 */
        "vld1.32    {d4-d7},   [%[b]]! \n"
        /* load a2, a3 */
        "vld1.32  {d12-d15},   [%[a]]! \n"
        "vmla.f32   q8,   q4,   d0[0]  \n"
        "vmla.f32   q9,   q4,   d2[0]  \n"
        "vmla.f32   q10,  q4,   d4[0]  \n"
        "vmla.f32   q11,  q4,   d6[0]  \n"
        "pld    [%[b]]                 \n"
        "vmla.f32   q8,   q5,   d0[1]  \n"
        "vmla.f32   q9,   q5,   d2[1]  \n"
        "vmla.f32   q10,  q5,   d4[1]  \n"
        "vmla.f32   q11,  q5,   d6[1]  \n"
        "subs   %[cnt],   %[cnt],  #1  \n"
        "vmla.f32   q8,   q6,   d1[0]  \n"
        "vmla.f32   q9,   q6,   d3[0]  \n"
        "vmla.f32   q10,  q6,   d5[0]  \n"
        "vmla.f32   q11,  q6,   d7[0]  \n"
        "pld    [%[b], #64]            \n"
        "vmla.f32   q8,   q7,   d1[1]  \n"
        "vmla.f32   q9,   q7,   d3[1]  \n"
        /* load b4, b5 */
        "vld1.32    {d0-d3},  [%[b]]!  \n"
        "vmla.f32   q10,  q7,   d5[1]  \n"
        "vmla.f32   q11,  q7,   d7[1]  \n"
        /* load b6, b7 */
        "vld1.32    {d4-d7},  [%[b]]!  \n"
        "vmla.f32   q12,  q4,   d0[0]  \n"
        "vmla.f32   q13,  q4,   d2[0]  \n"
        "vmla.f32   q14,  q4,   d4[0]  \n"
        "vmla.f32   q15,  q4,   d6[0]  \n"
        "sub  %[b],   %[b],   #128     \n"
        "vmla.f32   q12,  q5,   d0[1]  \n"
        "vmla.f32   q13,  q5,   d2[1]  \n"
        "vmla.f32   q14,  q5,   d4[1]  \n"
        "vmla.f32   q15,  q5,   d6[1]  \n"
        "add  %[b],   %[b],   %[ldb]   \n"
        "vmla.f32   q12,  q6,   d1[0]  \n"
        "vmla.f32   q13,  q6,   d3[0]  \n"
        "vmla.f32   q14,  q6,   d5[0]  \n"
        "vmla.f32   q15,  q6,   d7[0]  \n"
        /* load a0, a1 */
        "vld1.32    {d8-d11}, [%[a]]!  \n"
        "vmla.f32   q12,  q7,   d1[1]  \n"
        "vmla.f32   q13,  q7,   d3[1]  \n"
        /* load b0, b1 */
        "vld1.32    {d0-d3},  [%[b]]!  \n"
        "vmla.f32   q14,  q7,   d5[1]  \n"
        "vmla.f32   q15,  q7,   d7[1]  \n"
        "bne  1b                       \n"
        "cmp  %[relu],  #0             \n"
        "beq  2f                       \n"
        "vmov.u32   q0,   #0           \n"
        "vmax.f32   q8,   q8,   q0     \n"
        "vmax.f32   q9,   q9,   q0     \n"
        "vmax.f32   q10,  q10,  q0     \n"
        "vmax.f32   q11,  q11,  q0     \n"
        "vmax.f32   q12,  q12,  q0     \n"
        "vmax.f32   q13,  q13,  q0     \n"
        "vmax.f32   q14,  q14,  q0     \n"
        "vmax.f32   q15,  q15,  q0     \n"
        "2:\n"
        "vst1.32  {d16-d19}, [%[c]]!   \n"
        "vst1.32  {d20-d23}, [%[c]]!   \n"
        "vst1.32  {d24-d27}, [%[c]]!   \n"
        "vst1.32  {d28-d31}, [%[c]]!   \n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [relu] "r" (has_relu),
          [ldb]  "r" (ldb_byte),
          [bias] "r" (bias_ptr)
        : "q0", "q1", "q2", "q3", "q4", "q5",
          "q6", "q7", "q8", "q9", "q10", "q11",
          "q12", "q13", "q14", "q15", "cc", "memory"
      );
      b += 4 * 8;
    }
    for (; n > 3; n -= 4) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
      asm volatile(
        "vld1.32  {d24-d25}, [%[bias]] \n"
        /* load a0, a1 */
        "vld1.32  {d8-d11},  [%[a]]!   \n"
        /* mov bias to c0-c3*/
        "vmov.u32   q8,   q12 \n"
        "vmov.u32   q9,   q12 \n"
        "vmov.u32   q10,  q12 \n"
        "vmov.u32   q11,  q12 \n"
        "vmov.u32   q13,  #0  \n"
        "1:\n"
        /* load b0-b3 */
        "vld1.32  {d0-d3},  [%[b]]! \n"
        "vld1.32  {d4-d7},  [%[b]]! \n"
        /* load a2, a3 */
        "vld1.32  {d12-d15}, [%[a]]!\n"
        "vmla.f32  q8,   q4, d0[0]  \n"
        "vmla.f32  q9,   q4, d2[0]  \n"
        "vmla.f32  q10,  q4, d4[0]  \n"
        "vmla.f32  q11,  q4, d6[0]  \n"
        "sub  %[b], %[b], #64       \n"
        "vmla.f32  q8,   q5, d0[1]  \n"
        "vmla.f32  q9,   q5, d2[1]  \n"
        "vmla.f32  q10,  q5, d4[1]  \n"
        "vmla.f32  q11,  q5, d6[1]  \n"
        "add  %[b], %[b], %[ldb]    \n"
        "vmla.f32  q8,   q6, d1[0]  \n"
        "vmla.f32  q9,   q6, d3[0]  \n"
        "vmla.f32  q10,  q6, d5[0]  \n"
        "vmla.f32  q11,  q6, d7[0]  \n"
        /* load a0, a1 */
        "vld1.32  {d8-d11}, [%[a]]! \n"
        "vmla.f32  q8,   q7, d1[1]  \n"
        "vmla.f32  q9,   q7, d3[1]  \n"
        "vmla.f32  q10,  q7, d5[1]  \n"
        "vmla.f32  q11,  q7, d7[1]  \n"
        "subs %[cnt], %[cnt], #1    \n"
        "bne  1b                    \n"
        "cmp  %[relu],  #0          \n"
        "beq  2f                    \n"
        "vmax.f32 q8,   q8,   q13   \n"
        "vmax.f32 q9,   q9,   q13   \n"
        "vmax.f32 q10,  q10,  q13   \n"
        "vmax.f32 q11,  q11,  q13   \n"
        "2:\n"
        "vst1.32  {d16-d19}, [%[c]]!\n"
        "vst1.32  {d20-d23}, [%[c]]!\n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [relu] "r" (has_relu),
          [ldb]  "r" (ldb_byte),
          [bias] "r" (bias_ptr)
        : "q0", "q1", "q2", "q3", "q4", "q5",
          "q6", "q7", "q8", "q9", "q10", "q11",
          "q12", "q13", "cc", "memory"
      );
      b += 4 * 4;
    }
    for (; n > 0; n--) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
      asm volatile(
        "vld1.32  {d14-d15}, [%[bias]] \n"
        "vmov.u32   q8,   #0  \n"
        /* load a0, a1 */
        "vld1.32  {d2-d5}, [%[a]]! \n"
        /* mov bias to c0 */
        "vmov.u32   q5,   q7  \n"
        "vmov.u32   q6,   q8  \n"
        "1:\n"
        /* load b0 */
        "vld1.32  {d0-d1},  [%[b]]! \n"
        /* load a2, a3 */
        "vld1.32  {d6-d9},  [%[a]]! \n"
        "vmla.f32   q5, q1, d0[0]   \n"
        "vmla.f32   q6, q2, d0[1]   \n"
        "sub  %[b], %[b],   #16     \n"
        "subs %[cnt], %[cnt], #1    \n"
        "add  %[b], %[b], %[ldb]    \n"
        "vmla.f32   q5, q3, d1[0]   \n"
        "vmla.f32   q6, q4, d1[1]   \n"
         /* load a0, a1 */
        "vld1.32  {d2-d5}, [%[a]]!  \n"
        "bne  1b                    \n"
        "vadd.f32   q5, q5,   q6    \n"
        "cmp  %[relu],  #0          \n"
        "beq  2f                    \n"
        "vmax.f32   q5, q5,   q8    \n"
        "2:\n"
        "vst1.32  {d10-d11}, [%[c]]!\n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [relu] "r" (has_relu),
          [ldb]  "r" (ldb_byte),
          [bias] "r" (bias_ptr)
        : "q0", "q1", "q2", "q3", "q4", 
          "q5", "q6", "q7", "q8", "cc", "memory"
      );
      // clang-format on
      b += 4;
    }
    bias_ptr += 4;
    A_packed += lda;
  }
#endif
}

void sgemm_prepack_c4_small(int M,
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
  const int mloop = m_round >> 2;
  const int lda = 4 * k_round;
  const int ldb_byte = 4 * N * sizeof(float);
  const int kcnt = k_round >> 2;
  float bias_buf[m_round];  // NOLINT
  if (has_bias) {
    lite::TargetWrapperHost::MemcpySync(bias_buf, bias, M * sizeof(float));
    memset(bias_buf + M, 0, (m_round - M) * sizeof(float));
  } else {
    memset(bias_buf, 0, m_round * sizeof(float));
  }
#ifdef __aarch64__
  float32x4_t vzero = vdupq_n_f32(0.f);
  const float* bias_ptr = bias_buf;
  for (int m = 0; m < mloop; ++m) {
    float32x4_t vbias = vld1q_f32(bias_ptr);
    const float* b = B;
    int n = N;
    // clang-format off
    for (; n > 7; n -= 8) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
        asm volatile(
          /* load a0, a1 */
          "ld1  {v16.4s, v17.4s}, [%[a]], #32 \n"
          /* mov bias to c0-c7*/
          "mov  v8.16b,   %[vbias].16b \n"
          "mov  v9.16b,   %[vbias].16b \n"
          "mov  v10.16b,  %[vbias].16b \n"
          "mov  v11.16b,  %[vbias].16b \n"
          /* load b0, b1 */
          "ld1  {v0.4s,  v1.4s}, [%[b]], #32 \n"
          "mov  v12.16b,  %[vbias].16b \n"
          "mov  v13.16b,  %[vbias].16b \n"
          "mov  v14.16b,  %[vbias].16b \n"
          "mov  v15.16b,  %[vbias].16b \n"
          "1:\n"
          /* load b2, b3 */
          "ld1  {v2.4s,  v3.4s},  [%[b]], #32 \n"
          /* load a2, a3 */
          "ld1  {v18.4s, v19.4s}, [%[a]], #32 \n"
          "fmla v8.4s,  v16.4s, v0.s[0] \n"
          "fmla v9.4s,  v16.4s, v1.s[0] \n"
          "fmla v10.4s, v16.4s, v2.s[0] \n"
          "fmla v11.4s, v16.4s, v3.s[0] \n"
          "prfm pldl1keep, [%[b]]       \n"
          "fmla v8.4s,  v17.4s, v0.s[1] \n"
          "fmla v9.4s,  v17.4s, v1.s[1] \n"
          "fmla v10.4s, v17.4s, v2.s[1] \n"
          "fmla v11.4s, v17.4s, v3.s[1] \n"
          /* load b4, b5 */
          "ld1  {v4.4s, v5.4s}, [%[b]], #32 \n"
          "fmla v8.4s,  v18.4s, v0.s[2] \n"
          "fmla v9.4s,  v18.4s, v1.s[2] \n"
          "fmla v10.4s, v18.4s, v2.s[2] \n"
          "fmla v11.4s, v18.4s, v3.s[2] \n"
          /* load b6, b7 */
          "ld1  {v6.4s, v7.4s}, [%[b]], #32 \n"
          "fmla v8.4s,  v19.4s, v0.s[3] \n"
          "fmla v9.4s,  v19.4s, v1.s[3] \n"
          "fmla v10.4s, v19.4s, v2.s[3] \n"
          "fmla v11.4s, v19.4s, v3.s[3] \n"
          "sub  %[b],   %[b],   #128    \n"
          "fmla v12.4s, v16.4s, v4.s[0] \n"
          "fmla v13.4s, v16.4s, v5.s[0] \n"
          "fmla v14.4s, v16.4s, v6.s[0] \n"
          "fmla v15.4s, v16.4s, v7.s[0] \n"
          "add  %[b],   %[b],   %[ldb]  \n"
          "fmla v12.4s, v17.4s, v4.s[1] \n"
          "fmla v13.4s, v17.4s, v5.s[1] \n"
          "fmla v14.4s, v17.4s, v6.s[1] \n"
          "fmla v15.4s, v17.4s, v7.s[1] \n"
          /* load a0, a1 */
          "ld1  {v16.4s, v17.4s}, [%[a]], #32 \n"
          "fmla v12.4s, v18.4s, v4.s[2] \n"
          "fmla v13.4s, v18.4s, v5.s[2] \n"
          "fmla v14.4s, v18.4s, v6.s[2] \n"
          "fmla v15.4s, v18.4s, v7.s[2] \n"
          /* load b0, b1 */
          "ld1  {v0.4s,  v1.4s}, [%[b]], #32 \n"
          "fmla v12.4s, v19.4s, v4.s[3] \n"
          "fmla v13.4s, v19.4s, v5.s[3] \n"
          "fmla v14.4s, v19.4s, v6.s[3] \n"
          "fmla v15.4s, v19.4s, v7.s[3] \n"
          "subs %w[cnt], %w[cnt], #1    \n"
          "bne  1b                      \n"
          "cbz  %w[relu], 2f            \n"
          "fmax v8.4s,  v8.4s,  %[vzero].4s \n"
          "fmax v9.4s,  v9.4s,  %[vzero].4s \n"
          "fmax v10.4s, v10.4s, %[vzero].4s \n"
          "fmax v11.4s, v11.4s, %[vzero].4s \n"
          "fmax v12.4s, v12.4s, %[vzero].4s \n"
          "fmax v13.4s, v13.4s, %[vzero].4s \n"
          "fmax v14.4s, v14.4s, %[vzero].4s \n"
          "fmax v15.4s, v15.4s, %[vzero].4s \n"
          "2:\n"
          "st1  {v8.4s,  v9.4s,  v10.4s, v11.4s}, [%[c]], #64 \n"
          "st1  {v12.4s, v13.4s, v14.4s, v15.4s}, [%[c]], #64 \n"
          : [a] "+r" (a_ptr),
            [b] "+r" (b_ptr),
            [c] "+r" (C),
            [cnt] "+r" (cnt)
          : [relu] "r" (has_relu),
            [ldb]  "r" (ldb_byte),
            [vbias] "w" (vbias),
            [vzero] "w" (vzero)
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
            "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
            "v19", "cc", "memory"
          );

      b += 4 * 8;
    }
    for (; n > 3; n -= 4) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
        asm volatile(
          /* load a0, a1 */
          "ld1  {v16.4s, v17.4s}, [%[a]], #32 \n"
          /* mov bias to c0-c3*/
          "mov  v8.16b,   %[vbias].16b \n"
          "mov  v9.16b,   %[vbias].16b \n"
          "mov  v10.16b,  %[vbias].16b \n"
          "mov  v11.16b,  %[vbias].16b \n"
          "1:\n"
          /* load b0-b3 */
          "ld1  {v0.4s,  v1.4s},  [%[b]], #32 \n"
          "ld1  {v2.4s,  v3.4s},  [%[b]], #32 \n"
          /* load a2, a3 */
          "ld1  {v18.4s, v19.4s}, [%[a]], #32 \n"
          "fmla v8.4s,  v16.4s, v0.s[0] \n"
          "fmla v9.4s,  v16.4s, v1.s[0] \n"
          "fmla v10.4s, v16.4s, v2.s[0] \n"
          "fmla v11.4s, v16.4s, v3.s[0] \n"
          "sub  %[b],   %[b],   #64     \n"
          "fmla v8.4s,  v17.4s, v0.s[1] \n"
          "fmla v9.4s,  v17.4s, v1.s[1] \n"
          "fmla v10.4s, v17.4s, v2.s[1] \n"
          "fmla v11.4s, v17.4s, v3.s[1] \n"
          "add  %[b],   %[b],   %[ldb]  \n"
          "fmla v8.4s,  v18.4s, v0.s[2] \n"
          "fmla v9.4s,  v18.4s, v1.s[2] \n"
          "fmla v10.4s, v18.4s, v2.s[2] \n"
          "fmla v11.4s, v18.4s, v3.s[2] \n"
          /* load a0, a1 */
          "ld1  {v16.4s, v17.4s}, [%[a]], #32 \n"
          "fmla v8.4s,  v19.4s, v0.s[3] \n"
          "fmla v9.4s,  v19.4s, v1.s[3] \n"
          "fmla v10.4s, v19.4s, v2.s[3] \n"
          "fmla v11.4s, v19.4s, v3.s[3] \n"
          "subs %w[cnt], %w[cnt], #1    \n"
          "bne  1b                      \n"
          "cbz  %w[relu], 2f            \n"
          "fmax v8.4s,  v8.4s,  %[vzero].4s \n"
          "fmax v9.4s,  v9.4s,  %[vzero].4s \n"
          "fmax v10.4s, v10.4s, %[vzero].4s \n"
          "fmax v11.4s, v11.4s, %[vzero].4s \n"
          "2:\n"
          "st1  {v8.4s,  v9.4s,  v10.4s, v11.4s}, [%[c]], #64 \n"
          : [a] "+r" (a_ptr),
            [b] "+r" (b_ptr),
            [c] "+r" (C),
            [cnt] "+r" (cnt)
          : [relu] "r" (has_relu),
            [ldb]  "r" (ldb_byte),
            [vbias] "w" (vbias),
            [vzero] "w" (vzero)
          : "v0", "v1", "v2", "v3", "v8", "v9",
            "v10", "v11", "v16", "v17", "v18",
            "v19", "cc", "memory"
          ); 
      b += 4 * 4;
    }
    for (; n > 0; n--) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
        asm volatile(
          /* load a0, a1 */
          "ld1  {v16.4s, v17.4s}, [%[a]], #32 \n"
          /* mov bias to c0 */
          "mov  v8.16b,   %[vbias].16b \n"
          "mov  v9.16b,   %[vzero].16b \n"
          "1:\n"
          /* load b0 */
          "ld1  {v0.4s},  [%[b]], #16  \n"
          /* load a2, a3 */
          "ld1  {v18.4s, v19.4s}, [%[a]], #32 \n"
          "fmla v8.4s,  v16.4s, v0.s[0] \n"
          "fmla v9.4s,  v17.4s, v0.s[1] \n"
          "sub  %[b],   %[b],   #16     \n"
          "subs %w[cnt], %w[cnt], #1    \n"
          "add  %[b],   %[b],   %[ldb]  \n"
          "fmla v8.4s,  v18.4s, v0.s[2] \n"
          "fmla v9.4s,  v19.4s, v0.s[3] \n"
           /* load a0, a1 */
          "ld1  {v16.4s, v17.4s}, [%[a]], #32 \n"
          "bne  1b                      \n"
          "fadd v8.4s,  v8.4s,  v9.4s   \n"
          "cbz  %w[relu], 2f            \n"
          "fmax v8.4s,  v8.4s,  %[vzero].4s \n"
          "2:\n"
          "st1  {v8.4s}, [%[c]], #16    \n"
          : [a] "+r" (a_ptr),
            [b] "+r" (b_ptr),
            [c] "+r" (C),
            [cnt] "+r" (cnt)
          : [relu] "r" (has_relu),
            [ldb]  "r" (ldb_byte),
            [vbias] "w" (vbias),
            [vzero] "w" (vzero)
          : "v0", "v8", "v9", "v16", "v17", 
            "v18", "v19", "cc", "memory"
          );
      b += 4;
    }
    bias_ptr += 4;
    A_packed += lda;
  }
// clang-format on
#else
  const float* bias_ptr = bias_buf;
  for (int m = 0; m < mloop; ++m) {
    const float* b = B;
    int n = N;
    for (; n > 7; n -= 8) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
      // clang-format off
      asm volatile(
        "vld1.32  {d6-d7}, [%[bias]] \n"
        /* load a0, a1 */
        "vld1.32  {d8-d11}, [%[a]]!  \n"
        /* mov bias to c0-c7*/
        "vmov.u32   q8,    q3 \n"
        "vmov.u32   q9,    q3 \n"
        "vmov.u32   q10,   q3 \n"
        "vmov.u32   q11,   q3 \n"
        /* load b0, b1 */
        "vld1.32  {d0-d3}, [%[b]]! \n"
        "vmov.u32   q12,   q3 \n"
        "vmov.u32   q13,   q3 \n"
        "vmov.u32   q14,   q3 \n"
        "vmov.u32   q15,   q3 \n"
        "1:\n"
        /* load b2, b3 */
        "vld1.32    {d4-d7},   [%[b]]! \n"
        /* load a2, a3 */
        "vld1.32  {d12-d15},   [%[a]]! \n"
        "vmla.f32   q8,   q4,   d0[0]  \n"
        "vmla.f32   q9,   q4,   d2[0]  \n"
        "vmla.f32   q10,  q4,   d4[0]  \n"
        "vmla.f32   q11,  q4,   d6[0]  \n"
        "pld    [%[b]]                 \n"
        "vmla.f32   q8,   q5,   d0[1]  \n"
        "vmla.f32   q9,   q5,   d2[1]  \n"
        "vmla.f32   q10,  q5,   d4[1]  \n"
        "vmla.f32   q11,  q5,   d6[1]  \n"
        "subs   %[cnt],   %[cnt],  #1  \n"
        "vmla.f32   q8,   q6,   d1[0]  \n"
        "vmla.f32   q9,   q6,   d3[0]  \n"
        "vmla.f32   q10,  q6,   d5[0]  \n"
        "vmla.f32   q11,  q6,   d7[0]  \n"
        "pld    [%[b], #64]            \n"
        "vmla.f32   q8,   q7,   d1[1]  \n"
        "vmla.f32   q9,   q7,   d3[1]  \n"
        /* load b4, b5 */
        "vld1.32    {d0-d3},  [%[b]]!  \n"
        "vmla.f32   q10,  q7,   d5[1]  \n"
        "vmla.f32   q11,  q7,   d7[1]  \n"
        /* load b6, b7 */
        "vld1.32    {d4-d7},  [%[b]]!  \n"
        "vmla.f32   q12,  q4,   d0[0]  \n"
        "vmla.f32   q13,  q4,   d2[0]  \n"
        "vmla.f32   q14,  q4,   d4[0]  \n"
        "vmla.f32   q15,  q4,   d6[0]  \n"
        "sub  %[b],   %[b],   #128     \n"
        "vmla.f32   q12,  q5,   d0[1]  \n"
        "vmla.f32   q13,  q5,   d2[1]  \n"
        "vmla.f32   q14,  q5,   d4[1]  \n"
        "vmla.f32   q15,  q5,   d6[1]  \n"
        "add  %[b],   %[b],   %[ldb]   \n"
        "vmla.f32   q12,  q6,   d1[0]  \n"
        "vmla.f32   q13,  q6,   d3[0]  \n"
        "vmla.f32   q14,  q6,   d5[0]  \n"
        "vmla.f32   q15,  q6,   d7[0]  \n"
        /* load a0, a1 */
        "vld1.32    {d8-d11}, [%[a]]!  \n"
        "vmla.f32   q12,  q7,   d1[1]  \n"
        "vmla.f32   q13,  q7,   d3[1]  \n"
        /* load b0, b1 */
        "vld1.32    {d0-d3},  [%[b]]!  \n"
        "vmla.f32   q14,  q7,   d5[1]  \n"
        "vmla.f32   q15,  q7,   d7[1]  \n"
        "bne  1b                       \n"
        "cmp  %[relu],  #0             \n"
        "beq  2f                       \n"
        "vmov.u32   q0,   #0           \n"
        "vmax.f32   q8,   q8,   q0     \n"
        "vmax.f32   q9,   q9,   q0     \n"
        "vmax.f32   q10,  q10,  q0     \n"
        "vmax.f32   q11,  q11,  q0     \n"
        "vmax.f32   q12,  q12,  q0     \n"
        "vmax.f32   q13,  q13,  q0     \n"
        "vmax.f32   q14,  q14,  q0     \n"
        "vmax.f32   q15,  q15,  q0     \n"
        "2:\n"
        "vst1.32  {d16-d19}, [%[c]]!   \n"
        "vst1.32  {d20-d23}, [%[c]]!   \n"
        "vst1.32  {d24-d27}, [%[c]]!   \n"
        "vst1.32  {d28-d31}, [%[c]]!   \n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [relu] "r" (has_relu),
          [ldb]  "r" (ldb_byte),
          [bias] "r" (bias_ptr)
        : "q0", "q1", "q2", "q3", "q4", "q5",
          "q6", "q7", "q8", "q9", "q10", "q11",
          "q12", "q13", "q14", "q15", "cc", "memory"
      );
      b += 4 * 8;
    }
    for (; n > 3; n -= 4) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
      asm volatile(
        "vld1.32  {d24-d25}, [%[bias]] \n"
        /* load a0, a1 */
        "vld1.32  {d8-d11},  [%[a]]!   \n"
        /* mov bias to c0-c3*/
        "vmov.u32   q8,   q12 \n"
        "vmov.u32   q9,   q12 \n"
        "vmov.u32   q10,  q12 \n"
        "vmov.u32   q11,  q12 \n"
        "vmov.u32   q13,  #0  \n"
        "1:\n"
        /* load b0-b3 */
        "vld1.32  {d0-d3},  [%[b]]! \n"
        "vld1.32  {d4-d7},  [%[b]]! \n"
        /* load a2, a3 */
        "vld1.32  {d12-d15}, [%[a]]!\n"
        "vmla.f32  q8,   q4, d0[0]  \n"
        "vmla.f32  q9,   q4, d2[0]  \n"
        "vmla.f32  q10,  q4, d4[0]  \n"
        "vmla.f32  q11,  q4, d6[0]  \n"
        "sub  %[b], %[b], #64       \n"
        "vmla.f32  q8,   q5, d0[1]  \n"
        "vmla.f32  q9,   q5, d2[1]  \n"
        "vmla.f32  q10,  q5, d4[1]  \n"
        "vmla.f32  q11,  q5, d6[1]  \n"
        "add  %[b], %[b], %[ldb]    \n"
        "vmla.f32  q8,   q6, d1[0]  \n"
        "vmla.f32  q9,   q6, d3[0]  \n"
        "vmla.f32  q10,  q6, d5[0]  \n"
        "vmla.f32  q11,  q6, d7[0]  \n"
        /* load a0, a1 */
        "vld1.32  {d8-d11}, [%[a]]! \n"
        "vmla.f32  q8,   q7, d1[1]  \n"
        "vmla.f32  q9,   q7, d3[1]  \n"
        "vmla.f32  q10,  q7, d5[1]  \n"
        "vmla.f32  q11,  q7, d7[1]  \n"
        "subs %[cnt], %[cnt], #1    \n"
        "bne  1b                    \n"
        "cmp  %[relu],  #0          \n"
        "beq  2f                    \n"
        "vmax.f32 q8,   q8,   q13   \n"
        "vmax.f32 q9,   q9,   q13   \n"
        "vmax.f32 q10,  q10,  q13   \n"
        "vmax.f32 q11,  q11,  q13   \n"
        "2:\n"
        "vst1.32  {d16-d19}, [%[c]]!\n"
        "vst1.32  {d20-d23}, [%[c]]!\n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [relu] "r" (has_relu),
          [ldb]  "r" (ldb_byte),
          [bias] "r" (bias_ptr)
        : "q0", "q1", "q2", "q3", "q4", "q5",
          "q6", "q7", "q8", "q9", "q10", "q11",
          "q12", "q13", "cc", "memory"
      );
      b += 4 * 4;
    }
    for (; n > 0; n--) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
      asm volatile(
        "vld1.32  {d14-d15}, [%[bias]] \n"
        "vmov.u32   q8,   #0  \n"
        /* load a0, a1 */
        "vld1.32  {d2-d5}, [%[a]]! \n"
        /* mov bias to c0 */
        "vmov.u32   q5,   q7  \n"
        "vmov.u32   q6,   q8  \n"
        "1:\n"
        /* load b0 */
        "vld1.32  {d0-d1},  [%[b]]! \n"
        /* load a2, a3 */
        "vld1.32  {d6-d9},  [%[a]]! \n"
        "vmla.f32   q5, q1, d0[0]   \n"
        "vmla.f32   q6, q2, d0[1]   \n"
        "sub  %[b], %[b],   #16     \n"
        "subs %[cnt], %[cnt], #1    \n"
        "add  %[b], %[b], %[ldb]    \n"
        "vmla.f32   q5, q3, d1[0]   \n"
        "vmla.f32   q6, q4, d1[1]   \n"
         /* load a0, a1 */
        "vld1.32  {d2-d5}, [%[a]]!  \n"
        "bne  1b                    \n"
        "vadd.f32   q5, q5,   q6    \n"
        "cmp  %[relu],  #0          \n"
        "beq  2f                    \n"
        "vmax.f32   q5, q5,   q8    \n"
        "2:\n"
        "vst1.32  {d10-d11}, [%[c]]!\n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [relu] "r" (has_relu),
          [ldb]  "r" (ldb_byte),
          [bias] "r" (bias_ptr)
        : "q0", "q1", "q2", "q3", "q4", 
          "q5", "q6", "q7", "q8", "cc", "memory"
      );
      // clang-format on
      b += 4;
    }
    bias_ptr += 4;
    A_packed += lda;
  }
#endif
}

void sgemm_prepack_c4_small(int M,
                            int N,
                            int K,
                            const float* A_packed,
                            const float* B,
                            float* C,
                            ARMContext* ctx) {
  const int m_round = (M + 3) / 4 * 4;
  const int k_round = (K + 3) / 4 * 4;
  const int mloop = m_round >> 2;
  const int lda = 4 * k_round;
  const int ldb_byte = 4 * N * sizeof(float);
  const int kcnt = k_round >> 2;
#ifdef __aarch64__
  float32x4_t vzero = vdupq_n_f32(0.f);
  for (int m = 0; m < mloop; ++m) {
    const float* b = B;
    int n = N;
    // clang-format off
      for (; n > 7; n -= 8) {
        int cnt = kcnt;
        const float* a_ptr = A_packed;
        const float* b_ptr = b;
          asm volatile(
            "0:\n"
            /* load a0, a1 */
            "ld1  {v16.4s, v17.4s}, [%[a]], #32 \n"
            /* load b0, b1 */
            "ld1  {v0.4s,  v1.4s}, [%[b]], #32 \n"
            /* load b2, b3 */
            "ld1  {v2.4s,  v3.4s},  [%[b]], #32 \n"
            /* load a2, a3 */
            "fmul v8.4s,  v16.4s, v0.s[0] \n"
            "fmul v9.4s,  v16.4s, v1.s[0] \n"
            "fmul v10.4s, v16.4s, v2.s[0] \n"
            "fmul v11.4s, v16.4s, v3.s[0] \n"
            "ld1  {v18.4s, v19.4s}, [%[a]], #32 \n"
            "prfm pldl1keep, [%[b]]       \n"
            "fmla v8.4s,  v17.4s, v0.s[1] \n"
            "fmla v9.4s,  v17.4s, v1.s[1] \n"
            "fmla v10.4s, v17.4s, v2.s[1] \n"
            "fmla v11.4s, v17.4s, v3.s[1] \n"
            /* load b4, b5 */
            "ld1  {v4.4s, v5.4s}, [%[b]], #32 \n"
            "fmla v8.4s,  v18.4s, v0.s[2] \n"
            "fmla v9.4s,  v18.4s, v1.s[2] \n"
            "fmla v10.4s, v18.4s, v2.s[2] \n"
            "fmla v11.4s, v18.4s, v3.s[2] \n"
            /* load b6, b7 */
            "ld1  {v6.4s, v7.4s}, [%[b]], #32 \n"
            "fmla v8.4s,  v19.4s, v0.s[3] \n"
            "fmla v9.4s,  v19.4s, v1.s[3] \n"
            "fmla v10.4s, v19.4s, v2.s[3] \n"
            "fmla v11.4s, v19.4s, v3.s[3] \n"
            "sub  %[b],   %[b],   #128    \n"
            "fmul v12.4s, v16.4s, v4.s[0] \n"
            "fmul v13.4s, v16.4s, v5.s[0] \n"
            "fmul v14.4s, v16.4s, v6.s[0] \n"
            "fmul v15.4s, v16.4s, v7.s[0] \n"
            "add  %[b],   %[b],   %[ldb]  \n"
            "fmla v12.4s, v17.4s, v4.s[1] \n"
            "fmla v13.4s, v17.4s, v5.s[1] \n"
            "fmla v14.4s, v17.4s, v6.s[1] \n"
            "fmla v15.4s, v17.4s, v7.s[1] \n"
            /* load a0, a1 */
            "ld1  {v16.4s, v17.4s}, [%[a]], #32 \n"
            "fmla v12.4s, v18.4s, v4.s[2] \n"
            "fmla v13.4s, v18.4s, v5.s[2] \n"
            "fmla v14.4s, v18.4s, v6.s[2] \n"
            "fmla v15.4s, v18.4s, v7.s[2] \n"
            /* load b0, b1 */
            "ld1  {v0.4s,  v1.4s}, [%[b]], #32 \n"
            "fmla v12.4s, v19.4s, v4.s[3] \n"
            "fmla v13.4s, v19.4s, v5.s[3] \n"
            "fmla v14.4s, v19.4s, v6.s[3] \n"
            "fmla v15.4s, v19.4s, v7.s[3] \n"
            "subs %w[cnt], %w[cnt], #1    \n"
            "beq  2f                      \n"
            "1:\n"
            /* load b2, b3 */
            "ld1  {v2.4s,  v3.4s},  [%[b]], #32 \n"
            "fmla v8.4s,  v16.4s, v0.s[0] \n"
            "fmla v9.4s,  v16.4s, v1.s[0] \n"
            "fmla v10.4s, v16.4s, v2.s[0] \n"
            "fmla v11.4s, v16.4s, v3.s[0] \n"
            /* load a2, a3 */
            "ld1  {v18.4s, v19.4s}, [%[a]], #32 \n"
            "prfm pldl1keep, [%[b]]       \n"
            "fmla v8.4s,  v17.4s, v0.s[1] \n"
            "fmla v9.4s,  v17.4s, v1.s[1] \n"
            "fmla v10.4s, v17.4s, v2.s[1] \n"
            "fmla v11.4s, v17.4s, v3.s[1] \n"
            /* load b4, b5 */
            "ld1  {v4.4s, v5.4s}, [%[b]], #32 \n"
            "fmla v8.4s,  v18.4s, v0.s[2] \n"
            "fmla v9.4s,  v18.4s, v1.s[2] \n"
            "fmla v10.4s, v18.4s, v2.s[2] \n"
            "fmla v11.4s, v18.4s, v3.s[2] \n"
            /* load b6, b7 */
            "ld1  {v6.4s, v7.4s}, [%[b]], #32 \n"
            "fmla v8.4s,  v19.4s, v0.s[3] \n"
            "fmla v9.4s,  v19.4s, v1.s[3] \n"
            "fmla v10.4s, v19.4s, v2.s[3] \n"
            "fmla v11.4s, v19.4s, v3.s[3] \n"
            "sub  %[b],   %[b],   #128    \n"
            "fmla v12.4s, v16.4s, v4.s[0] \n"
            "fmla v13.4s, v16.4s, v5.s[0] \n"
            "fmla v14.4s, v16.4s, v6.s[0] \n"
            "fmla v15.4s, v16.4s, v7.s[0] \n"
            "add  %[b],   %[b],   %[ldb]  \n"
            "fmla v12.4s, v17.4s, v4.s[1] \n"
            "fmla v13.4s, v17.4s, v5.s[1] \n"
            "fmla v14.4s, v17.4s, v6.s[1] \n"
            "fmla v15.4s, v17.4s, v7.s[1] \n"
            /* load a0, a1 */
            "ld1  {v16.4s, v17.4s}, [%[a]], #32 \n"
            "fmla v12.4s, v18.4s, v4.s[2] \n"
            "fmla v13.4s, v18.4s, v5.s[2] \n"
            "fmla v14.4s, v18.4s, v6.s[2] \n"
            "fmla v15.4s, v18.4s, v7.s[2] \n"
            /* load b0, b1 */
            "ld1  {v0.4s,  v1.4s}, [%[b]], #32 \n"
            "fmla v12.4s, v19.4s, v4.s[3] \n"
            "fmla v13.4s, v19.4s, v5.s[3] \n"
            "fmla v14.4s, v19.4s, v6.s[3] \n"
            "fmla v15.4s, v19.4s, v7.s[3] \n"
            "subs %w[cnt], %w[cnt], #1    \n"
            "bne  1b                      \n"
            "2:\n"
            "st1  {v8.4s,  v9.4s,  v10.4s, v11.4s}, [%[c]], #64 \n"
            "st1  {v12.4s, v13.4s, v14.4s, v15.4s}, [%[c]], #64 \n"
            : [a] "+r" (a_ptr),
              [b] "+r" (b_ptr),
              [c] "+r" (C),
              [cnt] "+r" (cnt)
            : [ldb]  "r" (ldb_byte),
              [vzero] "w" (vzero)
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
              "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
              "v19", "cc", "memory"
            );
        b += 4 * 8;
      }
      for (; n > 3; n -= 4) {
        int cnt = kcnt;
        const float* a_ptr = A_packed;
        const float* b_ptr = b;
          asm volatile(
            "0:\n"
            /* load a0, a1 */
            "ld1  {v16.4s, v17.4s}, [%[a]], #32 \n"
            /* load b0-b3 */
            "ld1  {v0.4s,  v1.4s},  [%[b]], #32 \n"
            "ld1  {v2.4s,  v3.4s},  [%[b]], #32 \n"
            "fmul v8.4s,  v16.4s, v0.s[0] \n"
            "fmul v9.4s,  v16.4s, v1.s[0] \n"
            "fmul v10.4s, v16.4s, v2.s[0] \n"
            "fmul v11.4s, v16.4s, v3.s[0] \n"
            /* load a2, a3 */
            "ld1  {v18.4s, v19.4s}, [%[a]], #32 \n"
            "sub  %[b],   %[b],   #64     \n"
            "fmla v8.4s,  v17.4s, v0.s[1] \n"
            "fmla v9.4s,  v17.4s, v1.s[1] \n"
            "fmla v10.4s, v17.4s, v2.s[1] \n"
            "fmla v11.4s, v17.4s, v3.s[1] \n"
            "add  %[b],   %[b],   %[ldb]  \n"
            "fmla v8.4s,  v18.4s, v0.s[2] \n"
            "fmla v9.4s,  v18.4s, v1.s[2] \n"
            "fmla v10.4s, v18.4s, v2.s[2] \n"
            "fmla v11.4s, v18.4s, v3.s[2] \n"
            /* load a0, a1 */
            "ld1  {v16.4s, v17.4s}, [%[a]], #32 \n"
            "fmla v8.4s,  v19.4s, v0.s[3] \n"
            "fmla v9.4s,  v19.4s, v1.s[3] \n"
            "fmla v10.4s, v19.4s, v2.s[3] \n"
            "fmla v11.4s, v19.4s, v3.s[3] \n"
            "subs %w[cnt], %w[cnt], #1    \n"
            "beq  2f                      \n"
            "1:\n"
            /* load b0-b3 */
            "ld1  {v0.4s,  v1.4s},  [%[b]], #32 \n"
            "ld1  {v2.4s,  v3.4s},  [%[b]], #32 \n"
            "fmla v8.4s,  v16.4s, v0.s[0] \n"
            "fmla v9.4s,  v16.4s, v1.s[0] \n"
            "fmla v10.4s, v16.4s, v2.s[0] \n"
            "fmla v11.4s, v16.4s, v3.s[0] \n"
            /* load a2, a3 */
            "ld1  {v18.4s, v19.4s}, [%[a]], #32 \n"
            "sub  %[b],   %[b],   #64     \n"
            "fmla v8.4s,  v17.4s, v0.s[1] \n"
            "fmla v9.4s,  v17.4s, v1.s[1] \n"
            "fmla v10.4s, v17.4s, v2.s[1] \n"
            "fmla v11.4s, v17.4s, v3.s[1] \n"
            "add  %[b],   %[b],   %[ldb]  \n"
            "fmla v8.4s,  v18.4s, v0.s[2] \n"
            "fmla v9.4s,  v18.4s, v1.s[2] \n"
            "fmla v10.4s, v18.4s, v2.s[2] \n"
            "fmla v11.4s, v18.4s, v3.s[2] \n"
            /* load a0, a1 */
            "ld1  {v16.4s, v17.4s}, [%[a]], #32 \n"
            "fmla v8.4s,  v19.4s, v0.s[3] \n"
            "fmla v9.4s,  v19.4s, v1.s[3] \n"
            "fmla v10.4s, v19.4s, v2.s[3] \n"
            "fmla v11.4s, v19.4s, v3.s[3] \n"
            "subs %w[cnt], %w[cnt], #1    \n"
            "bne  1b                      \n"
            "2:\n"
            "st1  {v8.4s,  v9.4s,  v10.4s, v11.4s}, [%[c]], #64 \n"
          : [a] "+r" (a_ptr),
            [b] "+r" (b_ptr),
            [c] "+r" (C),
            [cnt] "+r" (cnt)
          : [ldb]  "r" (ldb_byte),
            [vzero] "w" (vzero)
          : "v0", "v1", "v2", "v3", "v8", "v9",
            "v10", "v11", "v16", "v17", "v18",
            "v19", "cc", "memory"
          );
        b += 4 * 4;
      }
      for (; n > 0; n--) {
        int cnt = kcnt;
        const float* a_ptr = A_packed;
        const float* b_ptr = b;
           asm volatile(
             "0:\n"
             /* load a0, a1 */
             "ld1  {v16.4s, v17.4s}, [%[a]], #32 \n"
             /* load b0 */
             "ld1  {v0.4s},  [%[b]], #16  \n"
             "fmul v8.4s,  v16.4s, v0.s[0] \n"
             "fmul v9.4s,  v17.4s, v0.s[1] \n"
             /* load a2, a3 */
             "ld1  {v18.4s, v19.4s}, [%[a]], #32 \n"
             "sub  %[b],   %[b],   #16     \n"
             "subs %w[cnt], %w[cnt], #1    \n"
             "add  %[b],   %[b],   %[ldb]  \n"
             "fmla v8.4s,  v18.4s, v0.s[2] \n"
             "fmla v9.4s,  v19.4s, v0.s[3] \n"
              /* load a0, a1 */
             "ld1  {v16.4s, v17.4s}, [%[a]], #32 \n"
             "beq  2f                      \n"
             "1:\n"
             /* load b0 */
             "ld1  {v0.4s},  [%[b]], #16  \n"
             "fmla v8.4s,  v16.4s, v0.s[0] \n"
             "fmla v9.4s,  v17.4s, v0.s[1] \n"
             /* load a2, a3 */
             "ld1  {v18.4s, v19.4s}, [%[a]], #32 \n"
             "sub  %[b],   %[b],   #16     \n"
             "subs %w[cnt], %w[cnt], #1    \n"
             "add  %[b],   %[b],   %[ldb]  \n"
             "fmla v8.4s,  v18.4s, v0.s[2] \n"
             "fmla v9.4s,  v19.4s, v0.s[3] \n"
              /* load a0, a1 */
             "ld1  {v16.4s, v17.4s}, [%[a]], #32 \n"
             "bne  1b                      \n"
             "2:\n"
             "fadd v8.4s,  v8.4s,  v9.4s   \n"
             "st1  {v8.4s}, [%[c]], #16    \n"
             : [a] "+r" (a_ptr),
               [b] "+r" (b_ptr),
               [c] "+r" (C),
               [cnt] "+r" (cnt)
             : [ldb]  "r" (ldb_byte),
               [vzero] "w" (vzero)
             : "v0", "v8", "v9", "v16", "v17", 
               "v18", "v19", "cc", "memory"
             );
        b += 4;
      }
    A_packed += lda;
  }
}
// clang-format on
#else
  for (int m = 0; m < mloop; ++m) {
    const float* b = B;
    int n = N;
    for (; n > 7; n -= 8) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
      // clang-format off
      asm volatile(
        "0:\n"
        /* load a0, a1 */
        "vld1.32  {d8-d11}, [%[a]]!  \n"
        "vld1.32  {d0-d3}, [%[b]]! \n"
        /* load b2, b3 */
        "vld1.32    {d4-d7},   [%[b]]! \n"
        "vmul.f32   q8,   q4,   d0[0]  \n"
        "vmul.f32   q9,   q4,   d2[0]  \n"
        "vmul.f32   q10,  q4,   d4[0]  \n"
        "vmul.f32   q11,  q4,   d6[0]  \n"
        /* load a2, a3 */
        "vld1.32  {d12-d15},   [%[a]]! \n"
        "pld    [%[b]]                 \n"
        "vmla.f32   q8,   q5,   d0[1]  \n"
        "vmla.f32   q9,   q5,   d2[1]  \n"
        "vmla.f32   q10,  q5,   d4[1]  \n"
        "vmla.f32   q11,  q5,   d6[1]  \n"
        "subs   %[cnt],   %[cnt],  #1  \n"
        "vmla.f32   q8,   q6,   d1[0]  \n"
        "vmla.f32   q9,   q6,   d3[0]  \n"
        "vmla.f32   q10,  q6,   d5[0]  \n"
        "vmla.f32   q11,  q6,   d7[0]  \n"
        "pld    [%[b], #64]            \n"
        "vmla.f32   q8,   q7,   d1[1]  \n"
        "vmla.f32   q9,   q7,   d3[1]  \n"
        /* load b4, b5 */
        "vld1.32    {d0-d3},  [%[b]]!  \n"
        "vmla.f32   q10,  q7,   d5[1]  \n"
        "vmla.f32   q11,  q7,   d7[1]  \n"
        /* load b6, b7 */
        "vld1.32    {d4-d7},  [%[b]]!  \n"
        "vmul.f32   q12,  q4,   d0[0]  \n"
        "vmul.f32   q13,  q4,   d2[0]  \n"
        "vmul.f32   q14,  q4,   d4[0]  \n"
        "vmul.f32   q15,  q4,   d6[0]  \n"
        "sub  %[b],   %[b],   #128     \n"
        "vmla.f32   q12,  q5,   d0[1]  \n"
        "vmla.f32   q13,  q5,   d2[1]  \n"
        "vmla.f32   q14,  q5,   d4[1]  \n"
        "vmla.f32   q15,  q5,   d6[1]  \n"
        "add  %[b],   %[b],   %[ldb]   \n"
        "vmla.f32   q12,  q6,   d1[0]  \n"
        "vmla.f32   q13,  q6,   d3[0]  \n"
        "vmla.f32   q14,  q6,   d5[0]  \n"
        "vmla.f32   q15,  q6,   d7[0]  \n"
        /* load a0, a1 */
        "vld1.32    {d8-d11}, [%[a]]!  \n"
        "vmla.f32   q12,  q7,   d1[1]  \n"
        "vmla.f32   q13,  q7,   d3[1]  \n"
        /* load b0, b1 */
        "vld1.32    {d0-d3},  [%[b]]!  \n"
        "vmla.f32   q14,  q7,   d5[1]  \n"
        "vmla.f32   q15,  q7,   d7[1]  \n"
        "beq  2f                       \n"
        "1:\n"
        /* load b2, b3 */
        "vld1.32    {d4-d7},   [%[b]]! \n"
        "vmla.f32   q8,   q4,   d0[0]  \n"
        "vmla.f32   q9,   q4,   d2[0]  \n"
        "vmla.f32   q10,  q4,   d4[0]  \n"
        "vmla.f32   q11,  q4,   d6[0]  \n"
        /* load a2, a3 */
        "vld1.32  {d12-d15},   [%[a]]! \n"
        "pld    [%[b]]                 \n"
        "vmla.f32   q8,   q5,   d0[1]  \n"
        "vmla.f32   q9,   q5,   d2[1]  \n"
        "vmla.f32   q10,  q5,   d4[1]  \n"
        "vmla.f32   q11,  q5,   d6[1]  \n"
        "subs   %[cnt],   %[cnt],  #1  \n"
        "vmla.f32   q8,   q6,   d1[0]  \n"
        "vmla.f32   q9,   q6,   d3[0]  \n"
        "vmla.f32   q10,  q6,   d5[0]  \n"
        "vmla.f32   q11,  q6,   d7[0]  \n"
        "pld    [%[b], #64]            \n"
        "vmla.f32   q8,   q7,   d1[1]  \n"
        "vmla.f32   q9,   q7,   d3[1]  \n"
        /* load b4, b5 */
        "vld1.32    {d0-d3},  [%[b]]!  \n"
        "vmla.f32   q10,  q7,   d5[1]  \n"
        "vmla.f32   q11,  q7,   d7[1]  \n"
        /* load b6, b7 */
        "vld1.32    {d4-d7},  [%[b]]!  \n"
        "vmla.f32   q12,  q4,   d0[0]  \n"
        "vmla.f32   q13,  q4,   d2[0]  \n"
        "vmla.f32   q14,  q4,   d4[0]  \n"
        "vmla.f32   q15,  q4,   d6[0]  \n"
        "sub  %[b],   %[b],   #128     \n"
        "vmla.f32   q12,  q5,   d0[1]  \n"
        "vmla.f32   q13,  q5,   d2[1]  \n"
        "vmla.f32   q14,  q5,   d4[1]  \n"
        "vmla.f32   q15,  q5,   d6[1]  \n"
        "add  %[b],   %[b],   %[ldb]   \n"
        "vmla.f32   q12,  q6,   d1[0]  \n"
        "vmla.f32   q13,  q6,   d3[0]  \n"
        "vmla.f32   q14,  q6,   d5[0]  \n"
        "vmla.f32   q15,  q6,   d7[0]  \n"
        /* load a0, a1 */
        "vld1.32    {d8-d11}, [%[a]]!  \n"
        "vmla.f32   q12,  q7,   d1[1]  \n"
        "vmla.f32   q13,  q7,   d3[1]  \n"
        /* load b0, b1 */
        "vld1.32    {d0-d3},  [%[b]]!  \n"
        "vmla.f32   q14,  q7,   d5[1]  \n"
        "vmla.f32   q15,  q7,   d7[1]  \n"
        "bne  1b                       \n"
        "2:\n"
        "vst1.32  {d16-d19}, [%[c]]!   \n"
        "vst1.32  {d20-d23}, [%[c]]!   \n"
        "vst1.32  {d24-d27}, [%[c]]!   \n"
        "vst1.32  {d28-d31}, [%[c]]!   \n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [ldb]  "r" (ldb_byte)
        : "q0", "q1", "q2", "q3", "q4", "q5",
          "q6", "q7", "q8", "q9", "q10", "q11",
          "q12", "q13", "q14", "q15", "cc", "memory"
      );
      b += 4 * 8;
    }
    for (; n > 3; n -= 4) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
      asm volatile(
        "0:\n"
        /* load a0, a1 */
        "vld1.32  {d8-d11},  [%[a]]!   \n"
        /* load b0-b3 */
        "vld1.32  {d0-d3},  [%[b]]! \n"
        "vld1.32  {d4-d7},  [%[b]]! \n"
        "vmul.f32  q8,   q4, d0[0]  \n"
        "vmul.f32  q9,   q4, d2[0]  \n"
        "vmul.f32  q10,  q4, d4[0]  \n"
        "vmul.f32  q11,  q4, d6[0]  \n"
        /* load a2, a3 */
        "vld1.32  {d12-d15}, [%[a]]!\n"
        "sub  %[b], %[b], #64       \n"
        "vmla.f32  q8,   q5, d0[1]  \n"
        "vmla.f32  q9,   q5, d2[1]  \n"
        "vmla.f32  q10,  q5, d4[1]  \n"
        "vmla.f32  q11,  q5, d6[1]  \n"
        "add  %[b], %[b], %[ldb]    \n"
        "vmla.f32  q8,   q6, d1[0]  \n"
        "vmla.f32  q9,   q6, d3[0]  \n"
        "vmla.f32  q10,  q6, d5[0]  \n"
        "vmla.f32  q11,  q6, d7[0]  \n"
        /* load a0, a1 */
        "vld1.32  {d8-d11}, [%[a]]! \n"
        "vmla.f32  q8,   q7, d1[1]  \n"
        "vmla.f32  q9,   q7, d3[1]  \n"
        "vmla.f32  q10,  q7, d5[1]  \n"
        "vmla.f32  q11,  q7, d7[1]  \n"
        "subs %[cnt], %[cnt], #1    \n"
        "beq  2f                    \n"
        "1:\n"
        /* load b0-b3 */
        "vld1.32  {d0-d3},  [%[b]]! \n"
        "vld1.32  {d4-d7},  [%[b]]! \n"
        "vmla.f32  q8,   q4, d0[0]  \n"
        "vmla.f32  q9,   q4, d2[0]  \n"
        "vmla.f32  q10,  q4, d4[0]  \n"
        "vmla.f32  q11,  q4, d6[0]  \n"
        /* load a2, a3 */
        "vld1.32  {d12-d15}, [%[a]]!\n"
        "sub  %[b], %[b], #64       \n"
        "vmla.f32  q8,   q5, d0[1]  \n"
        "vmla.f32  q9,   q5, d2[1]  \n"
        "vmla.f32  q10,  q5, d4[1]  \n"
        "vmla.f32  q11,  q5, d6[1]  \n"
        "add  %[b], %[b], %[ldb]    \n"
        "vmla.f32  q8,   q6, d1[0]  \n"
        "vmla.f32  q9,   q6, d3[0]  \n"
        "vmla.f32  q10,  q6, d5[0]  \n"
        "vmla.f32  q11,  q6, d7[0]  \n"
        /* load a0, a1 */
        "vld1.32  {d8-d11}, [%[a]]! \n"
        "vmla.f32  q8,   q7, d1[1]  \n"
        "vmla.f32  q9,   q7, d3[1]  \n"
        "vmla.f32  q10,  q7, d5[1]  \n"
        "vmla.f32  q11,  q7, d7[1]  \n"
        "subs %[cnt], %[cnt], #1    \n"
        "bne  1b                    \n"
        "2:\n"
        "vst1.32  {d16-d19}, [%[c]]!\n"
        "vst1.32  {d20-d23}, [%[c]]!\n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [ldb]  "r" (ldb_byte)
        : "q0", "q1", "q2", "q3", "q4", "q5",
          "q6", "q7", "q8", "q9", "q10", "q11",
          "q12", "q13", "cc", "memory"
      );
      b += 4 * 4;
    }
    for (; n > 0; n--) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
      asm volatile(
        "0:\n"
        /* load a0, a1 */
        "vld1.32  {d2-d5}, [%[a]]! \n"
        /* load b0 */
        "vld1.32  {d0-d1},  [%[b]]! \n"
        "vmul.f32   q5, q1, d0[0]   \n"
        "vmul.f32   q6, q2, d0[1]   \n"
        /* load a2, a3 */
        "vld1.32  {d6-d9},  [%[a]]! \n"
        "sub  %[b], %[b],   #16     \n"
        "subs %[cnt], %[cnt], #1    \n"
        "add  %[b], %[b], %[ldb]    \n"
        "vmla.f32   q5, q3, d1[0]   \n"
        "vmla.f32   q6, q4, d1[1]   \n"
         /* load a0, a1 */
        "vld1.32  {d2-d5}, [%[a]]!  \n"
        "beq  2f                    \n"
        "1:\n"
        /* load b0 */
        "vld1.32  {d0-d1},  [%[b]]! \n"
        "vmla.f32   q5, q1, d0[0]   \n"
        "vmla.f32   q6, q2, d0[1]   \n"
        /* load a2, a3 */
        "vld1.32  {d6-d9},  [%[a]]! \n"
        "sub  %[b], %[b],   #16     \n"
        "subs %[cnt], %[cnt], #1    \n"
        "add  %[b], %[b], %[ldb]    \n"
        "vmla.f32   q5, q3, d1[0]   \n"
        "vmla.f32   q6, q4, d1[1]   \n"
         /* load a0, a1 */
        "vld1.32  {d2-d5}, [%[a]]!  \n"
        "bne  1b                    \n"
        "2:\n"
        "vadd.f32   q5, q5,   q6    \n"
        "vst1.32  {d10-d11}, [%[c]]!\n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [ldb]  "r" (ldb_byte)
        : "q0", "q1", "q2", "q3", "q4", 
          "q5", "q6", "q7", "q8", "cc", "memory"
      );
      // clang-format on
      b += 4;
    }
    A_packed += lda;
  }
}
#endif

void sgemm_prepack_c4_small_a35(int M,
                                int N,
                                int K,
                                const float* A_packed,
                                const float* B,
                                float* C,
                                ARMContext* ctx) {
  const int m_round = (M + 3) / 4 * 4;
  const int k_round = (K + 3) / 4 * 4;
  const int mloop = m_round >> 2;
  const int lda = 4 * k_round;
  const int ldb_byte = 4 * N * sizeof(float);
  const int kcnt = k_round >> 2;
#ifdef __aarch64__
  float32x4_t vzero = vdupq_n_f32(0.f);
  for (int m = 0; m < mloop; ++m) {
    const float* b = B;
    int n = N;
    // clang-format off
    for (; n > 7; n -= 8) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
        asm volatile(
        "prfm pldl1keep, [%[b]]       \n"
        "prfm pldl1keep, [%[b], #64]  \n"
        "prfm pldl1keep, [%[a]]       \n"
        "mov w0, #0                \n"
        "dup v3.4s, w0             \n"
        "mov  v8.16b,   v3.16b     \n"
        "mov  v9.16b,   v3.16b     \n"
        "ldr  d16, [%[a], #0]      \n"
        "mov  v10.16b,  v3.16b     \n"
        "mov  v11.16b,  v3.16b     \n"
        "ld1r  {v0.2s},  [%[b]], #4\n"
        "mov  v12.16b,  v3.16b     \n"
        "mov  v13.16b,  v3.16b     \n"
        "ldr  d28, [%[a], #8]      \n"
        "mov  v14.16b,  v3.16b     \n"
        "mov  v15.16b,  v3.16b     \n"
        "ld1r  {v4.2s},  [%[b]], #4\n"
        "mov  v20.16b,  v3.16b     \n"
        "mov  v21.16b,  v3.16b     \n"
        "ldr  d17, [%[a], #16]     \n"
        "mov  v22.16b,  v3.16b     \n"
        "mov  v23.16b,  v3.16b     \n"
        "ld1r  {v1.2s},  [%[b]], #4\n"
        "mov  v24.16b,  v3.16b     \n"
        "mov  v25.16b,  v3.16b     \n"
        "ldr  d29, [%[a], #24]     \n"
        "mov  v26.16b,  v3.16b     \n"
        "add  %[a], %[a], #32      \n"
        "mov  v27.16b,  v3.16b     \n"
        "ld1r  {v5.2s},  [%[b]], #4\n"
        "1:\n"

        "ld1r  {v2.2s},  [%[b]], #4 \n"
        "fmla v8.2s,  v16.2s, v0.2s \n"
        "fmla v20.2s, v28.2s, v0.2s \n"
        "ldr  d18, [%[a], #0]       \n"
        "ld1r  {v6.2s},  [%[b]], #4 \n"
        "fmla v9.2s,  v16.2s, v2.2s \n"
        "fmla v21.2s, v28.2s, v2.2s \n"
        "ldr  d30, [%[a], #8]       \n"
        "ld1r  {v3.2s},  [%[b]], #4 \n"
        "fmla v8.2s,   v17.2s, v4.2s \n"
        "fmla v20.2s,  v29.2s, v4.2s \n"
        "ldr  d19, [%[a], #16]       \n"
        "ld1r  {v7.2s},  [%[b]], #4  \n"
        "fmla v9.2s,   v17.2s, v6.2s \n"
        "fmla v21.2s,  v29.2s, v6.2s \n"
        "ldr  d31, [%[a], #24]       \n"
        "ld1r  {v0.2s},  [%[b]], #4  \n"
        "fmla v8.2s,   v18.2s, v1.2s \n"
        "add  %[a], %[a], #32      \n"
        "fmla v20.2s,  v30.2s, v1.2s \n"
        "fmla v9.2s,   v18.2s, v3.2s \n"
        "ld1r  {v4.2s},  [%[b]], #4  \n"
        "fmla v21.2s,  v30.2s, v3.2s \n"
        "fmla v8.2s,  v19.2s, v5.2s  \n"
        "fmla v20.2s,  v31.2s,v5.2s  \n"
        "ld1r  {v1.2s},  [%[b]], #4  \n"
        "fmla v9.2s,  v19.2s, v7.2s  \n"
        "ld1r  {v5.2s},  [%[b]], #4  \n"
        "fmla v21.2s,  v31.2s,v7.2s  \n"

        "ld1r  {v2.2s},  [%[b]], #4 \n"
        "fmla v10.2s, v16.2s, v0.2s \n"
        "fmla v22.2s, v28.2s, v0.2s \n"
        "ld1r  {v6.2s},  [%[b]], #4 \n"
        "fmla v11.2s, v16.2s, v2.2s \n"
        "fmla v23.2s, v28.2s, v2.2s \n"
        "fmla v10.2s,  v17.2s, v4.2s\n"
        "ld1r  {v3.2s},  [%[b]], #4 \n"
        "fmla v22.2s,  v29.2s, v4.2s\n"
        "fmla v11.2s,  v17.2s, v6.2s\n"
        "ld1r  {v7.2s},  [%[b]], #4 \n"
        "fmla v23.2s,  v29.2s, v6.2s\n"
        "fmla v10.2s,  v18.2s, v1.2s\n"
        "ld1r  {v0.2s},  [%[b]], #4 \n"
        "fmla v22.2s,  v30.2s, v1.2s\n"
        "fmla v11.2s,  v18.2s, v3.2s\n"
        "ld1r  {v4.2s},  [%[b]], #4 \n"
        "fmla v23.2s,  v30.2s, v3.2s\n"
        "fmla v10.2s,  v19.2s, v5.2s\n"
        "ld1r  {v1.2s},  [%[b]], #4 \n"
        "fmla v22.2s,  v31.2s, v5.2s\n"
        "fmla v11.2s,  v19.2s, v7.2s\n"
        "ld1r  {v5.2s},  [%[b]], #4 \n"
        "fmla v23.2s,  v31.2s, v7.2s\n"

        "ld1r  {v2.2s},  [%[b]], #4 \n"
        "fmla v12.2s, v16.2s, v0.2s \n"
        "fmla v24.2s, v28.2s, v0.2s \n"
        "ld1r  {v6.2s},  [%[b]], #4 \n"
        "fmla v13.2s, v16.2s, v2.2s \n"
        "fmla v25.2s, v28.2s, v2.2s \n"
        "ld1r  {v3.2s},  [%[b]], #4 \n"
        "fmla v12.2s, v17.2s, v4.2s \n"
        "fmla v24.2s, v29.2s, v4.2s \n"
        "ld1r  {v7.2s},  [%[b]], #4 \n"
        "fmla v13.2s, v17.2s, v6.2s \n"
        "fmla v25.2s, v29.2s, v6.2s \n"
        "ld1r  {v0.2s},  [%[b]], #4 \n"
        "fmla v12.2s, v18.2s, v1.2s \n"
        "fmla v24.2s, v30.2s, v1.2s \n"
        "ld1r  {v4.2s},  [%[b]], #4 \n"
        "fmla v13.2s, v18.2s, v3.2s \n"
        "fmla v25.2s, v30.2s, v3.2s \n"
        "fmla v12.2s, v19.2s, v5.2s \n"
        "ld1r  {v1.2s},  [%[b]], #4 \n"
        "fmla v24.2s, v31.2s, v5.2s \n"
        "fmla v13.2s, v19.2s, v7.2s \n"
        "ld1r  {v5.2s},  [%[b]], #4 \n"
        "fmla v25.2s, v31.2s, v7.2s \n"

        "ld1r  {v2.2s},  [%[b]], #4 \n"
        "fmla v14.2s, v16.2s, v0.2s \n"
        "fmla v26.2s, v28.2s, v0.2s \n"
        "ld1r  {v6.2s},  [%[b]], #4 \n"
        "fmla v15.2s, v16.2s, v2.2s \n"
        "fmla v27.2s, v28.2s, v2.2s \n"
        "ld1r  {v3.2s},  [%[b]], #4 \n"
        "fmla v14.2s, v17.2s, v4.2s \n"
        "fmla v26.2s, v29.2s, v4.2s \n"
        "ld1r  {v7.2s},  [%[b]], #4 \n"
        "fmla v15.2s, v17.2s, v6.2s \n"
        "ldr  d16, [%[a], #0]       \n"
        "sub  %[b],   %[b],   #128  \n"
        "fmla v27.2s, v29.2s, v6.2s \n"
        "add  %[b],   %[b],   %[ldb]\n"
        "fmla v14.2s, v18.2s, v1.2s \n"
        "ldr  d28, [%[a], #8]       \n"
        "ld1r  {v0.2s},  [%[b]], #4 \n"
        "fmla v26.2s, v30.2s, v1.2s \n"
        "fmla v15.2s, v18.2s, v3.2s \n"
        "ldr  d17, [%[a], #16]       \n"
        "ld1r  {v4.2s},  [%[b]], #4 \n"
        "fmla v27.2s, v30.2s, v3.2s \n"
        "fmla v14.2s, v19.2s, v5.2s \n"
        "ldr  d29, [%[a], #24]       \n"
        "ld1r  {v1.2s},  [%[b]], #4 \n"
        "fmla v26.2s, v31.2s, v5.2s \n"
        "add  %[a], %[a], #32      \n"
        "fmla v15.2s, v19.2s, v7.2s \n"
        "ld1r  {v5.2s},  [%[b]], #4 \n"
        "fmla v27.2s, v31.2s, v7.2s \n"
        "subs %w[cnt], %w[cnt], #1  \n"

        "bne  1b                    \n"
        "sub  %[b], %[b], #16       \n"
        "sub  %[a], %[a], #32       \n"

        "2:\n"
        "st1  {v8.2s }, [%[c]], #8 \n"
        "st1  {v20.2s}, [%[c]], #8 \n"
        "st1  {v9.2s }, [%[c]], #8 \n"
        "st1  {v21.2s}, [%[c]], #8 \n"
        "st1  {v10.2s}, [%[c]], #8 \n"
        "st1  {v22.2s}, [%[c]], #8 \n"
        "st1  {v11.2s}, [%[c]], #8 \n"
        "st1  {v23.2s}, [%[c]], #8 \n"
        "st1  {v12.2s}, [%[c]], #8 \n"
        "st1  {v24.2s}, [%[c]], #8 \n"
        "st1  {v13.2s}, [%[c]], #8 \n"
        "st1  {v25.2s}, [%[c]], #8 \n"
        "st1  {v14.2s}, [%[c]], #8 \n"
        "st1  {v26.2s}, [%[c]], #8 \n"
        "st1  {v15.2s}, [%[c]], #8 \n"
        "st1  {v27.2s}, [%[c]], #8 \n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [ldb]  "r" (ldb_byte)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
          "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
          "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", 
          "v28", "v29", "v30", "v31", "w0", "cc", "memory"
      );
      b += 4 * 8;
    }
    for (; n > 3; n -= 4) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
        asm volatile(
          "prfm pldl1keep, [%[a]]\n"
          "prfm pldl1keep, [%[b]]\n"
          "mov w0, #0             \n"
          "dup v3.4s, w0          \n"
          "mov  v8.16b,   v3.16b \n"
          "ld1r  {v0.2s},  [%[b]], #4\n"
          "mov  v9.16b,   v3.16b \n"
          "ld1  {v16.2s}, [%[a]], #8\n"
          "mov  v10.16b,  v3.16b \n"
          "ld1r  {v4.2s},  [%[b]], #4\n"
          "mov  v11.16b,  v3.16b \n"
          "ld1  {v28.2s}, [%[a]], #8\n"
          "mov  v20.16b,  v3.16b \n"
          "ld1r  {v1.2s},  [%[b]], #4\n"
          "mov  v21.16b,  v3.16b \n"
          "ld1  {v17.2s}, [%[a]], #8\n"
          "mov  v22.16b,  v3.16b \n"
          "ld1r  {v5.2s},  [%[b]], #4\n"
          "mov  v23.16b,  v3.16b \n"
          "ld1  {v29.2s}, [%[a]], #8\n"
          "1:\n"

          "fmla v8.2s,  v16.2s, v0.2s \n"
          "ld1r  {v2.2s},  [%[b]], #4 \n"
          "fmla v20.2s, v28.2s, v0.2s \n"
          "ld1  {v18.2s}, [%[a]], #8  \n"
          "fmla v9.2s,  v16.2s, v2.2s \n"
          "ld1r  {v6.2s},  [%[b]], #4 \n"
          "fmla v21.2s, v28.2s, v2.2s \n"
          "ld1  {v30.2s}, [%[a]], #8  \n"
          "fmla v8.2s,   v17.2s, v4.2s\n"
          "ld1r  {v3.2s},  [%[b]], #4 \n"
          "fmla v20.2s,  v29.2s, v4.2s\n"
          "ld1  {v19.2s}, [%[a]], #8  \n"
          "fmla v9.2s,   v17.2s, v6.2s\n"
          "ld1r  {v7.2s},  [%[b]], #4 \n"
          "fmla v21.2s,  v29.2s, v6.2s\n"
          "ld1  {v31.2s}, [%[a]], #8  \n"
          "fmla v8.2s,   v18.2s, v1.2s\n"
          "ld1r  {v0.2s},  [%[b]], #4 \n"
          "fmla v20.2s,  v30.2s, v1.2s\n"
          "fmla v9.2s,   v18.2s, v3.2s\n"
          "ld1r  {v4.2s},  [%[b]], #4 \n"
          "fmla v21.2s,  v30.2s, v3.2s\n"
          "fmla v8.2s,  v19.2s, v5.2s \n"
          "ld1r  {v1.2s},  [%[b]], #4 \n"
          "fmla v20.2s,  v31.2s,v5.2s \n"
          "fmla v9.2s,  v19.2s, v7.2s \n"
          "ld1r  {v5.2s},  [%[b]], #4 \n"
          "fmla v21.2s,  v31.2s,v7.2s \n"

          "ld1r  {v2.2s},  [%[b]], #4 \n"
          "fmla v10.2s, v16.2s, v0.2s \n"
          "fmla v22.2s, v28.2s, v0.2s \n"
          "fmla v11.2s, v16.2s, v2.2s \n"
          "ld1r  {v6.2s},  [%[b]], #4 \n"
          "fmla v23.2s, v28.2s, v2.2s \n"
          "fmla v10.2s,  v17.2s, v4.2s\n"
          "ld1r  {v3.2s},  [%[b]], #4 \n"
          "fmla v22.2s,  v29.2s, v4.2s\n"
          "fmla v11.2s,  v17.2s, v6.2s\n"
          "ld1r  {v7.2s},  [%[b]], #4 \n"
          "fmla v23.2s,  v29.2s, v6.2s \n"
          "fmla v10.2s,  v18.2s, v1.2s \n"
          "sub  %[b],   %[b],   #64    \n"
          "fmla v22.2s,  v30.2s, v1.2s \n"
          "add  %[b],   %[b],   %[ldb] \n"
          "fmla v11.2s,  v18.2s, v3.2s \n"
          "ld1r  {v0.2s},  [%[b]], #4  \n"
          "ld1  {v16.2s}, [%[a]], #8   \n"
          "fmla v23.2s,  v30.2s, v3.2s \n"
          "ld1  {v28.2s}, [%[a]], #8   \n"
          "ld1r  {v4.2s},  [%[b]], #4  \n"
          "fmla v10.2s,  v19.2s, v5.2s \n"
          "ld1  {v17.2s}, [%[a]], #8   \n"
          "fmla v22.2s,  v31.2s, v5.2s \n"
          "ld1r  {v1.2s},  [%[b]], #4  \n"
          "ld1  {v29.2s}, [%[a]], #8   \n"
          "fmla v11.2s,  v19.2s, v7.2s \n"
          "fmla v23.2s,  v31.2s, v7.2s \n"
          "ld1r  {v5.2s},  [%[b]], #4  \n"

          "subs %w[cnt], %w[cnt], #1   \n"
          "bne  1b                     \n"
          "sub  %[b], %[b], #16       \n"
          "sub  %[a], %[a], #32       \n"
          "2:\n"
          "st1  {v8.2s }, [%[c]], #8 \n"
          "st1  {v20.2s}, [%[c]], #8 \n"
          "st1  {v9.2s }, [%[c]], #8 \n"
          "st1  {v21.2s}, [%[c]], #8 \n"
          "st1  {v10.2s}, [%[c]], #8 \n"
          "st1  {v22.2s}, [%[c]], #8 \n"
          "st1  {v11.2s}, [%[c]], #8 \n"
          "st1  {v23.2s}, [%[c]], #8 \n"
          : [a] "+r" (a_ptr),
            [b] "+r" (b_ptr),
            [c] "+r" (C),
            [cnt] "+r" (cnt)
          : [ldb]  "r" (ldb_byte)
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
            "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
            "v28", "v29", "v30", "v31", "w0", "cc", "memory");
      b += 4 * 4;
    }
    for (; n > 0; n--) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
        asm volatile(
          "mov w0, #0                \n"
          "dup v0.4s, w0             \n"
          "mov  v8.16b,   v0.16b     \n"
          "mov  v20.16b,  v0.16b     \n"
          "ld1  {v16.2s}, [%[a]], #8 \n"
          "ld1r  {v0.2s},  [%[b]], #4\n"
          "ld1  {v28.2s}, [%[a]], #8 \n"
          "ld1r  {v4.2s},  [%[b]], #4\n"
          "ld1  {v17.2s}, [%[a]], #8 \n"
          "ld1  {v29.2s}, [%[a]], #8 \n"
          "1:\n"
          "fmla v8.2s,  v16.2s, v0.2s \n"
          "ld1  {v18.2s}, [%[a]], #8  \n"
          "ld1r  {v1.2s},  [%[b]], #4 \n"
          "fmla v20.2s, v28.2s, v0.2s \n"
          "ld1  {v30.2s}, [%[a]], #8  \n"
          "ld1r  {v5.2s},  [%[b]], #4 \n"
          "ld1  {v19.2s}, [%[a]], #8  \n"
          "sub  %[b],   %[b],   #16   \n"
          "ld1  {v31.2s}, [%[a]], #8  \n"
          "add  %[b],   %[b],   %[ldb]\n"
          "fmla v8.2s,   v17.2s, v4.2s\n"
          "ld1  {v16.2s}, [%[a]], #8  \n"
          "fmla v20.2s,  v29.2s, v4.2s\n"

          "ld1  {v28.2s}, [%[a]], #8  \n"
          "fmla v8.2s,   v18.2s, v1.2s\n"
          "ld1  {v17.2s}, [%[a]], #8  \n"
          "fmla v20.2s,  v30.2s, v1.2s\n"
          "ld1  {v29.2s}, [%[a]], #8  \n"
          "fmla v8.2s,  v19.2s, v5.2s \n"
          "ld1r  {v0.2s},  [%[b]], #4 \n"
          "fmla v20.2s,  v31.2s,v5.2s \n"
          "ld1r  {v4.2s},  [%[b]], #4 \n"

          "subs %w[cnt], %w[cnt], #1    \n"
          "bne  1b                      \n"
          "2:\n"
          "st1  {v8.2s }, [%[c]], #8 \n"
          "st1  {v20.2s}, [%[c]], #8 \n"
          : [a] "+r" (a_ptr),
            [b] "+r" (b_ptr),
            [c] "+r" (C),
            [cnt] "+r" (cnt)
          : [ldb]  "r" (ldb_byte)
          : "v0", "v8", "v9", "v16", "v17", 
            "v18", "v19", "cc", "w0", "memory"
        );
      b += 4;
    }
    A_packed += lda;
  }
}
// clang-format on
#else
  for (int m = 0; m < mloop; ++m) {
    const float* b = B;
    int n = N;
    for (; n > 7; n -= 8) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
      // clang-format off
      asm volatile(
        "0:\n"
        /* load a0, a1 */
        "vld1.32  {d8-d11}, [%[a]]!  \n"
        "vld1.32  {d0-d3}, [%[b]]! \n"
        /* load b2, b3 */
        "vld1.32    {d4-d7},   [%[b]]! \n"
        "vmul.f32   q8,   q4,   d0[0]  \n"
        "vmul.f32   q9,   q4,   d2[0]  \n"
        "vmul.f32   q10,  q4,   d4[0]  \n"
        "vmul.f32   q11,  q4,   d6[0]  \n"
        /* load a2, a3 */
        "vld1.32  {d12-d15},   [%[a]]! \n"
        "pld    [%[b]]                 \n"
        "vmla.f32   q8,   q5,   d0[1]  \n"
        "vmla.f32   q9,   q5,   d2[1]  \n"
        "vmla.f32   q10,  q5,   d4[1]  \n"
        "vmla.f32   q11,  q5,   d6[1]  \n"
        "subs   %[cnt],   %[cnt],  #1  \n"
        "vmla.f32   q8,   q6,   d1[0]  \n"
        "vmla.f32   q9,   q6,   d3[0]  \n"
        "vmla.f32   q10,  q6,   d5[0]  \n"
        "vmla.f32   q11,  q6,   d7[0]  \n"
        "pld    [%[b], #64]            \n"
        "vmla.f32   q8,   q7,   d1[1]  \n"
        "vmla.f32   q9,   q7,   d3[1]  \n"
        /* load b4, b5 */
        "vld1.32    {d0-d3},  [%[b]]!  \n"
        "vmla.f32   q10,  q7,   d5[1]  \n"
        "vmla.f32   q11,  q7,   d7[1]  \n"
        /* load b6, b7 */
        "vld1.32    {d4-d7},  [%[b]]!  \n"
        "vmul.f32   q12,  q4,   d0[0]  \n"
        "vmul.f32   q13,  q4,   d2[0]  \n"
        "vmul.f32   q14,  q4,   d4[0]  \n"
        "vmul.f32   q15,  q4,   d6[0]  \n"
        "sub  %[b],   %[b],   #128     \n"
        "vmla.f32   q12,  q5,   d0[1]  \n"
        "vmla.f32   q13,  q5,   d2[1]  \n"
        "vmla.f32   q14,  q5,   d4[1]  \n"
        "vmla.f32   q15,  q5,   d6[1]  \n"
        "add  %[b],   %[b],   %[ldb]   \n"
        "vmla.f32   q12,  q6,   d1[0]  \n"
        "vmla.f32   q13,  q6,   d3[0]  \n"
        "vmla.f32   q14,  q6,   d5[0]  \n"
        "vmla.f32   q15,  q6,   d7[0]  \n"
        /* load a0, a1 */
        "vld1.32    {d8-d11}, [%[a]]!  \n"
        "vmla.f32   q12,  q7,   d1[1]  \n"
        "vmla.f32   q13,  q7,   d3[1]  \n"
        /* load b0, b1 */
        "vld1.32    {d0-d3},  [%[b]]!  \n"
        "vmla.f32   q14,  q7,   d5[1]  \n"
        "vmla.f32   q15,  q7,   d7[1]  \n"
        "beq  2f                       \n"
        "1:\n"
        /* load b2, b3 */
        "vld1.32    {d4-d7},   [%[b]]! \n"
        "vmla.f32   q8,   q4,   d0[0]  \n"
        "vmla.f32   q9,   q4,   d2[0]  \n"
        "vmla.f32   q10,  q4,   d4[0]  \n"
        "vmla.f32   q11,  q4,   d6[0]  \n"
        /* load a2, a3 */
        "vld1.32  {d12-d15},   [%[a]]! \n"
        "pld    [%[b]]                 \n"
        "vmla.f32   q8,   q5,   d0[1]  \n"
        "vmla.f32   q9,   q5,   d2[1]  \n"
        "vmla.f32   q10,  q5,   d4[1]  \n"
        "vmla.f32   q11,  q5,   d6[1]  \n"
        "subs   %[cnt],   %[cnt],  #1  \n"
        "vmla.f32   q8,   q6,   d1[0]  \n"
        "vmla.f32   q9,   q6,   d3[0]  \n"
        "vmla.f32   q10,  q6,   d5[0]  \n"
        "vmla.f32   q11,  q6,   d7[0]  \n"
        "pld    [%[b], #64]            \n"
        "vmla.f32   q8,   q7,   d1[1]  \n"
        "vmla.f32   q9,   q7,   d3[1]  \n"
        /* load b4, b5 */
        "vld1.32    {d0-d3},  [%[b]]!  \n"
        "vmla.f32   q10,  q7,   d5[1]  \n"
        "vmla.f32   q11,  q7,   d7[1]  \n"
        /* load b6, b7 */
        "vld1.32    {d4-d7},  [%[b]]!  \n"
        "vmla.f32   q12,  q4,   d0[0]  \n"
        "vmla.f32   q13,  q4,   d2[0]  \n"
        "vmla.f32   q14,  q4,   d4[0]  \n"
        "vmla.f32   q15,  q4,   d6[0]  \n"
        "sub  %[b],   %[b],   #128     \n"
        "vmla.f32   q12,  q5,   d0[1]  \n"
        "vmla.f32   q13,  q5,   d2[1]  \n"
        "vmla.f32   q14,  q5,   d4[1]  \n"
        "vmla.f32   q15,  q5,   d6[1]  \n"
        "add  %[b],   %[b],   %[ldb]   \n"
        "vmla.f32   q12,  q6,   d1[0]  \n"
        "vmla.f32   q13,  q6,   d3[0]  \n"
        "vmla.f32   q14,  q6,   d5[0]  \n"
        "vmla.f32   q15,  q6,   d7[0]  \n"
        /* load a0, a1 */
        "vld1.32    {d8-d11}, [%[a]]!  \n"
        "vmla.f32   q12,  q7,   d1[1]  \n"
        "vmla.f32   q13,  q7,   d3[1]  \n"
        /* load b0, b1 */
        "vld1.32    {d0-d3},  [%[b]]!  \n"
        "vmla.f32   q14,  q7,   d5[1]  \n"
        "vmla.f32   q15,  q7,   d7[1]  \n"
        "bne  1b                       \n"
        "2:\n"
        "vst1.32  {d16-d19}, [%[c]]!   \n"
        "vst1.32  {d20-d23}, [%[c]]!   \n"
        "vst1.32  {d24-d27}, [%[c]]!   \n"
        "vst1.32  {d28-d31}, [%[c]]!   \n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [ldb]  "r" (ldb_byte)
        : "q0", "q1", "q2", "q3", "q4", "q5",
          "q6", "q7", "q8", "q9", "q10", "q11",
          "q12", "q13", "q14", "q15", "cc", "memory"
      );
      b += 4 * 8;
    }
    for (; n > 3; n -= 4) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
      asm volatile(
        "0:\n"
        /* load a0, a1 */
        "vld1.32  {d8-d11},  [%[a]]!   \n"
        /* load b0-b3 */
        "vld1.32  {d0-d3},  [%[b]]! \n"
        "vld1.32  {d4-d7},  [%[b]]! \n"
        "vmul.f32  q8,   q4, d0[0]  \n"
        "vmul.f32  q9,   q4, d2[0]  \n"
        "vmul.f32  q10,  q4, d4[0]  \n"
        "vmul.f32  q11,  q4, d6[0]  \n"
        /* load a2, a3 */
        "vld1.32  {d12-d15}, [%[a]]!\n"
        "sub  %[b], %[b], #64       \n"
        "vmla.f32  q8,   q5, d0[1]  \n"
        "vmla.f32  q9,   q5, d2[1]  \n"
        "vmla.f32  q10,  q5, d4[1]  \n"
        "vmla.f32  q11,  q5, d6[1]  \n"
        "add  %[b], %[b], %[ldb]    \n"
        "vmla.f32  q8,   q6, d1[0]  \n"
        "vmla.f32  q9,   q6, d3[0]  \n"
        "vmla.f32  q10,  q6, d5[0]  \n"
        "vmla.f32  q11,  q6, d7[0]  \n"
        /* load a0, a1 */
        "vld1.32  {d8-d11}, [%[a]]! \n"
        "vmla.f32  q8,   q7, d1[1]  \n"
        "vmla.f32  q9,   q7, d3[1]  \n"
        "vmla.f32  q10,  q7, d5[1]  \n"
        "vmla.f32  q11,  q7, d7[1]  \n"
        "subs %[cnt], %[cnt], #1    \n"
        "beq  2f                    \n"
        "1:\n"
        /* load b0-b3 */
        "vld1.32  {d0-d3},  [%[b]]! \n"
        "vld1.32  {d4-d7},  [%[b]]! \n"
        "vmla.f32  q8,   q4, d0[0]  \n"
        "vmla.f32  q9,   q4, d2[0]  \n"
        "vmla.f32  q10,  q4, d4[0]  \n"
        "vmla.f32  q11,  q4, d6[0]  \n"
        /* load a2, a3 */
        "vld1.32  {d12-d15}, [%[a]]!\n"
        "sub  %[b], %[b], #64       \n"
        "vmla.f32  q8,   q5, d0[1]  \n"
        "vmla.f32  q9,   q5, d2[1]  \n"
        "vmla.f32  q10,  q5, d4[1]  \n"
        "vmla.f32  q11,  q5, d6[1]  \n"
        "add  %[b], %[b], %[ldb]    \n"
        "vmla.f32  q8,   q6, d1[0]  \n"
        "vmla.f32  q9,   q6, d3[0]  \n"
        "vmla.f32  q10,  q6, d5[0]  \n"
        "vmla.f32  q11,  q6, d7[0]  \n"
        /* load a0, a1 */
        "vld1.32  {d8-d11}, [%[a]]! \n"
        "vmla.f32  q8,   q7, d1[1]  \n"
        "vmla.f32  q9,   q7, d3[1]  \n"
        "vmla.f32  q10,  q7, d5[1]  \n"
        "vmla.f32  q11,  q7, d7[1]  \n"
        "subs %[cnt], %[cnt], #1    \n"
        "bne  1b                    \n"
        "2:\n"
        "vst1.32  {d16-d19}, [%[c]]!\n"
        "vst1.32  {d20-d23}, [%[c]]!\n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [ldb]  "r" (ldb_byte)
        : "q0", "q1", "q2", "q3", "q4", "q5",
          "q6", "q7", "q8", "q9", "q10", "q11",
          "q12", "q13", "cc", "memory"
      );
      b += 4 * 4;
    }
    for (; n > 0; n--) {
      int cnt = kcnt;
      const float* a_ptr = A_packed;
      const float* b_ptr = b;
      asm volatile(
        "0:\n"
        /* load a0, a1 */
        "vld1.32  {d2-d5}, [%[a]]! \n"
        /* load b0 */
        "vld1.32  {d0-d1},  [%[b]]! \n"
        "vmul.f32   q5, q1, d0[0]   \n"
        "vmul.f32   q6, q2, d0[1]   \n"
        /* load a2, a3 */
        "vld1.32  {d6-d9},  [%[a]]! \n"
        "sub  %[b], %[b],   #16     \n"
        "subs %[cnt], %[cnt], #1    \n"
        "add  %[b], %[b], %[ldb]    \n"
        "vmla.f32   q5, q3, d1[0]   \n"
        "vmla.f32   q6, q4, d1[1]   \n"
         /* load a0, a1 */
        "vld1.32  {d2-d5}, [%[a]]!  \n"
        "beq  2f                    \n"
        "1:\n"
        /* load b0 */
        "vld1.32  {d0-d1},  [%[b]]! \n"
        "vmla.f32   q5, q1, d0[0]   \n"
        "vmla.f32   q6, q2, d0[1]   \n"
        /* load a2, a3 */
        "vld1.32  {d6-d9},  [%[a]]! \n"
        "sub  %[b], %[b],   #16     \n"
        "subs %[cnt], %[cnt], #1    \n"
        "add  %[b], %[b], %[ldb]    \n"
        "vmla.f32   q5, q3, d1[0]   \n"
        "vmla.f32   q6, q4, d1[1]   \n"
         /* load a0, a1 */
        "vld1.32  {d2-d5}, [%[a]]!  \n"
        "bne  1b                    \n"
        "2:\n"
        "vadd.f32   q5, q5,   q6    \n"
        "vst1.32  {d10-d11}, [%[c]]!\n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [ldb]  "r" (ldb_byte)
        : "q0", "q1", "q2", "q3", "q4", 
          "q5", "q6", "q7", "q8", "cc", "memory"
      );
      // clang-format on
      b += 4;
    }
    A_packed += lda;
  }
}
#endif

void sgemm_prepack_c8_int16_small(int M,
                                  int N,
                                  int K,
                                  const int16_t* A_packed,
                                  const int16_t* B,
                                  int32_t* C,
                                  ARMContext* ctx,
                                  int beta) {
  const int m_round = (M + 7) / 8 * 8;
  const int k_round = (K + 7) / 8 * 8;
  const int mloop = m_round >> 3;
  const int lda = 8 * k_round;
  const int ldb_byte = 8 * N * sizeof(int16_t);
  const int kcnt = k_round >> 3;
#ifdef __aarch64__
  float32x4_t vzero = vdupq_n_f32(0.f);
#endif
  for (int m = 0; m < mloop; ++m) {
    const int16_t* b = B;
    int n = N;
#ifdef __aarch64__
    for (; n > 7; n -= 8) {
      int cnt = kcnt;
      const int16_t* a_ptr = A_packed;
      const int16_t* b_ptr = b;
      // clang-format off
      asm volatile(
        "ld1 {v0.8h, v1.8h}, [%[a]], #32 \n" //load a0, a1
        "ld1 {v4.8h, v5.8h}, [%[b]], #32 \n" //load b0, b1
        "ld1 {v6.8h, v7.8h}, [%[b]], #32 \n" //load b2, b3

        "smull v20.4s, v0.4h, v4.h[0] \n"
        "smull v21.4s, v0.4h, v5.h[0] \n"
        "smull v22.4s, v0.4h, v6.h[0] \n"
        "smull v23.4s, v0.4h, v7.h[0] \n"
        "ld1 {v8.8h, v9.8h}, [%[b]], #32 \n" //load b0, b1
        "ld1 {v10.8h, v11.8h}, [%[b]], #32 \n" //load b2, b3

        "smull2 v24.4s, v0.8h, v4.h[0] \n"
        "smull2 v25.4s, v0.8h, v5.h[0] \n"
        "smull2 v26.4s, v0.8h, v6.h[0] \n"
        "smull2 v27.4s, v0.8h, v7.h[0] \n"
        "ld1 {v2.8h, v3.8h}, [%[a]], #32 \n" //load a2, a3

        "smlal v20.4s, v1.4h, v4.h[1] \n"
        "smlal v21.4s, v1.4h, v5.h[1] \n"
        "smlal v22.4s, v1.4h, v6.h[1] \n"
        "smlal v23.4s, v1.4h, v7.h[1] \n"

        "smlal2 v24.4s, v1.8h, v4.h[1] \n"
        "smlal2 v25.4s, v1.8h, v5.h[1] \n"
        "smlal2 v26.4s, v1.8h, v6.h[1] \n"
        "smlal2 v27.4s, v1.8h, v7.h[1] \n"

        "smull v12.4s, v0.4h, v8.h[0] \n"
        "smull v13.4s, v0.4h, v9.h[0] \n"
        "smull v14.4s, v0.4h, v10.h[0] \n"
        "smull v15.4s, v0.4h, v11.h[0] \n"

        "smull2 v16.4s, v0.8h, v8.h[0] \n"
        "smull2 v17.4s, v0.8h, v9.h[0] \n"
        "smull2 v18.4s, v0.8h, v10.h[0] \n"
        "smull2 v19.4s, v0.8h, v11.h[0] \n"

        "smlal v12.4s, v1.4h, v8.h[1] \n"
        "smlal v13.4s, v1.4h, v9.h[1] \n"
        "smlal v14.4s, v1.4h, v10.h[1] \n"
        "smlal v15.4s, v1.4h, v11.h[1] \n"

        "smlal2 v16.4s, v1.8h, v8.h[1] \n"
        "smlal2 v17.4s, v1.8h, v9.h[1] \n"
        "smlal2 v18.4s, v1.8h, v10.h[1] \n"
        "smlal2 v19.4s, v1.8h, v11.h[1] \n"

        "smlal v20.4s, v2.4h, v4.h[2] \n"
        "smlal v21.4s, v2.4h, v5.h[2] \n"
        "smlal v22.4s, v2.4h, v6.h[2] \n"
        "smlal v23.4s, v2.4h, v7.h[2] \n"
        "ld1 {v0.8h, v1.8h}, [%[a]], #32 \n" //load a0, a1
        "smlal2 v24.4s, v2.8h, v4.h[2] \n"
        "smlal2 v25.4s, v2.8h, v5.h[2] \n"
        "smlal2 v26.4s, v2.8h, v6.h[2] \n"
        "smlal2 v27.4s, v2.8h, v7.h[2] \n"
        "smlal v12.4s, v2.4h, v8.h[2] \n"
        "smlal v13.4s, v2.4h, v9.h[2] \n"
        "smlal v14.4s, v2.4h, v10.h[2] \n"
        "smlal v15.4s, v2.4h, v11.h[2] \n"
        "smlal2 v16.4s, v2.8h, v8.h[2] \n"
        "smlal2 v17.4s, v2.8h, v9.h[2] \n"
        "smlal2 v18.4s, v2.8h, v10.h[2] \n"
        "smlal2 v19.4s, v2.8h, v11.h[2] \n"

        "smlal v20.4s, v3.4h, v4.h[3] \n"
        "smlal v21.4s, v3.4h, v5.h[3] \n"
        "smlal v22.4s, v3.4h, v6.h[3] \n"
        "smlal v23.4s, v3.4h, v7.h[3] \n"
        "smlal2 v24.4s, v3.8h, v4.h[3] \n"
        "smlal2 v25.4s, v3.8h, v5.h[3] \n"
        "smlal2 v26.4s, v3.8h, v6.h[3] \n"
        "smlal2 v27.4s, v3.8h, v7.h[3] \n"
        "smlal v12.4s, v3.4h, v8.h[3] \n"
        "smlal v13.4s, v3.4h, v9.h[3] \n"
        "smlal v14.4s, v3.4h, v10.h[3] \n"
        "smlal v15.4s, v3.4h, v11.h[3] \n"
        "smlal2 v16.4s, v3.8h, v8.h[3] \n"
        "smlal2 v17.4s, v3.8h, v9.h[3] \n"
        "smlal2 v18.4s, v3.8h, v10.h[3] \n"
        "smlal2 v19.4s, v3.8h, v11.h[3] \n"

        "smlal v20.4s, v0.4h, v4.h[4] \n"
        "smlal v21.4s, v0.4h, v5.h[4] \n"
        "smlal v22.4s, v0.4h, v6.h[4] \n"
        "smlal v23.4s, v0.4h, v7.h[4] \n"

        "smlal2 v24.4s, v0.8h, v4.h[4] \n"
        "smlal2 v25.4s, v0.8h, v5.h[4] \n"
        "smlal2 v26.4s, v0.8h, v6.h[4] \n"
        "smlal2 v27.4s, v0.8h, v7.h[4] \n"
        "ld1 {v2.8h, v3.8h}, [%[a]], #32 \n" //load a2, a3

        "smlal v20.4s, v1.4h, v4.h[5] \n"
        "smlal v21.4s, v1.4h, v5.h[5] \n"
        "smlal v22.4s, v1.4h, v6.h[5] \n"
        "smlal v23.4s, v1.4h, v7.h[5] \n"

        "smlal2 v24.4s, v1.8h, v4.h[5] \n"
        "smlal2 v25.4s, v1.8h, v5.h[5] \n"
        "smlal2 v26.4s, v1.8h, v6.h[5] \n"
        "smlal2 v27.4s, v1.8h, v7.h[5] \n"

        "smlal v12.4s, v0.4h, v8.h[4] \n"
        "smlal v13.4s, v0.4h, v9.h[4] \n"
        "smlal v14.4s, v0.4h, v10.h[4] \n"
        "smlal v15.4s, v0.4h, v11.h[4] \n"

        "smlal2 v16.4s, v0.8h, v8.h[4] \n"
        "smlal2 v17.4s, v0.8h, v9.h[4] \n"
        "smlal2 v18.4s, v0.8h, v10.h[4] \n"
        "smlal2 v19.4s, v0.8h, v11.h[4] \n"

        "smlal v12.4s, v1.4h, v8.h[5] \n"
        "smlal v13.4s, v1.4h, v9.h[5] \n"
        "smlal v14.4s, v1.4h, v10.h[5] \n"
        "smlal v15.4s, v1.4h, v11.h[5] \n"

        "smlal2 v16.4s, v1.8h, v8.h[5] \n"
        "smlal2 v17.4s, v1.8h, v9.h[5] \n"
        "smlal2 v18.4s, v1.8h, v10.h[5] \n"
        "smlal2 v19.4s, v1.8h, v11.h[5] \n"

        "smlal v20.4s, v2.4h, v4.h[6] \n"
        "smlal v21.4s, v2.4h, v5.h[6] \n"
        "smlal v22.4s, v2.4h, v6.h[6] \n"
        "smlal v23.4s, v2.4h, v7.h[6] \n"
        "ld1 {v0.8h, v1.8h}, [%[a]], #32 \n" //load a0, a1
        "smlal2 v24.4s, v2.8h, v4.h[6] \n"
        "smlal2 v25.4s, v2.8h, v5.h[6] \n"
        "smlal2 v26.4s, v2.8h, v6.h[6] \n"
        "smlal2 v27.4s, v2.8h, v7.h[6] \n"
        "sub %[b], %[b], #128         \n"
        "add %[b], %[b], %[ldb]        \n"
        "smlal v20.4s, v3.4h, v4.h[7] \n"
        "smlal v21.4s, v3.4h, v5.h[7] \n"
        "smlal v22.4s, v3.4h, v6.h[7] \n"
        "smlal v23.4s, v3.4h, v7.h[7] \n"
        "smlal2 v24.4s, v3.8h, v4.h[7] \n"
        "smlal2 v25.4s, v3.8h, v5.h[7] \n"
        "smlal2 v26.4s, v3.8h, v6.h[7] \n"
        "smlal2 v27.4s, v3.8h, v7.h[7] \n"
        "ld1 {v4.8h, v5.8h}, [%[b]], #32 \n" //load b0, b1
        "ld1 {v6.8h, v7.8h}, [%[b]], #32 \n" //load b2, b3

        "smlal v12.4s, v2.4h, v8.h[6] \n"
        "smlal v13.4s, v2.4h, v9.h[6] \n"
        "smlal v14.4s, v2.4h, v10.h[6] \n"
        "smlal v15.4s, v2.4h, v11.h[6] \n"
        "smlal2 v16.4s, v2.8h, v8.h[6] \n"
        "smlal2 v17.4s, v2.8h, v9.h[6] \n"
        "smlal2 v18.4s, v2.8h, v10.h[6] \n"
        "smlal2 v19.4s, v2.8h, v11.h[6] \n"
        "subs   %w[cnt], %w[cnt], #1      \n"

        "smlal v12.4s, v3.4h, v8.h[7] \n"
        "smlal v13.4s, v3.4h, v9.h[7] \n"
        "smlal v14.4s, v3.4h, v10.h[7] \n"
        "smlal v15.4s, v3.4h, v11.h[7] \n"
        "smlal2 v16.4s, v3.8h, v8.h[7] \n"
        "smlal2 v17.4s, v3.8h, v9.h[7] \n"
        "smlal2 v18.4s, v3.8h, v10.h[7] \n"
        "smlal2 v19.4s, v3.8h, v11.h[7] \n"

        "beq 2f                         \n"
        "1:\n"
        "smlal v20.4s, v0.4h, v4.h[0] \n"
        "smlal v21.4s, v0.4h, v5.h[0] \n"
        "smlal v22.4s, v0.4h, v6.h[0] \n"
        "smlal v23.4s, v0.4h, v7.h[0] \n"
        "ld1 {v8.8h, v9.8h}, [%[b]], #32 \n" //load b0, b1
        "ld1 {v10.8h, v11.8h}, [%[b]], #32 \n" //load b2, b3

        "smlal2 v24.4s, v0.8h, v4.h[0] \n"
        "smlal2 v25.4s, v0.8h, v5.h[0] \n"
        "smlal2 v26.4s, v0.8h, v6.h[0] \n"
        "smlal2 v27.4s, v0.8h, v7.h[0] \n"
        "ld1 {v2.8h, v3.8h}, [%[a]], #32 \n" //load a2, a3

        "smlal v20.4s, v1.4h, v4.h[1] \n"
        "smlal v21.4s, v1.4h, v5.h[1] \n"
        "smlal v22.4s, v1.4h, v6.h[1] \n"
        "smlal v23.4s, v1.4h, v7.h[1] \n"

        "smlal2 v24.4s, v1.8h, v4.h[1] \n"
        "smlal2 v25.4s, v1.8h, v5.h[1] \n"
        "smlal2 v26.4s, v1.8h, v6.h[1] \n"
        "smlal2 v27.4s, v1.8h, v7.h[1] \n"

        "smlal v12.4s, v0.4h, v8.h[0] \n"
        "smlal v13.4s, v0.4h, v9.h[0] \n"
        "smlal v14.4s, v0.4h, v10.h[0] \n"
        "smlal v15.4s, v0.4h, v11.h[0] \n"

        "smlal2 v16.4s, v0.8h, v8.h[0] \n"
        "smlal2 v17.4s, v0.8h, v9.h[0] \n"
        "smlal2 v18.4s, v0.8h, v10.h[0] \n"
        "smlal2 v19.4s, v0.8h, v11.h[0] \n"

        "smlal v12.4s, v1.4h, v8.h[1] \n"
        "smlal v13.4s, v1.4h, v9.h[1] \n"
        "smlal v14.4s, v1.4h, v10.h[1] \n"
        "smlal v15.4s, v1.4h, v11.h[1] \n"

        "smlal2 v16.4s, v1.8h, v8.h[1] \n"
        "smlal2 v17.4s, v1.8h, v9.h[1] \n"
        "smlal2 v18.4s, v1.8h, v10.h[1] \n"
        "smlal2 v19.4s, v1.8h, v11.h[1] \n"

        "smlal v20.4s, v2.4h, v4.h[2] \n"
        "smlal v21.4s, v2.4h, v5.h[2] \n"
        "smlal v22.4s, v2.4h, v6.h[2] \n"
        "smlal v23.4s, v2.4h, v7.h[2] \n"
        "ld1 {v0.8h, v1.8h}, [%[a]], #32 \n" //load a0, a1
        "smlal2 v24.4s, v2.8h, v4.h[2] \n"
        "smlal2 v25.4s, v2.8h, v5.h[2] \n"
        "smlal2 v26.4s, v2.8h, v6.h[2] \n"
        "smlal2 v27.4s, v2.8h, v7.h[2] \n"
        "smlal v12.4s, v2.4h, v8.h[2] \n"
        "smlal v13.4s, v2.4h, v9.h[2] \n"
        "smlal v14.4s, v2.4h, v10.h[2] \n"
        "smlal v15.4s, v2.4h, v11.h[2] \n"
        "smlal2 v16.4s, v2.8h, v8.h[2] \n"
        "smlal2 v17.4s, v2.8h, v9.h[2] \n"
        "smlal2 v18.4s, v2.8h, v10.h[2] \n"
        "smlal2 v19.4s, v2.8h, v11.h[2] \n"

        "smlal v20.4s, v3.4h, v4.h[3] \n"
        "smlal v21.4s, v3.4h, v5.h[3] \n"
        "smlal v22.4s, v3.4h, v6.h[3] \n"
        "smlal v23.4s, v3.4h, v7.h[3] \n"
        "smlal2 v24.4s, v3.8h, v4.h[3] \n"
        "smlal2 v25.4s, v3.8h, v5.h[3] \n"
        "smlal2 v26.4s, v3.8h, v6.h[3] \n"
        "smlal2 v27.4s, v3.8h, v7.h[3] \n"
        "smlal v12.4s, v3.4h, v8.h[3] \n"
        "smlal v13.4s, v3.4h, v9.h[3] \n"
        "smlal v14.4s, v3.4h, v10.h[3] \n"
        "smlal v15.4s, v3.4h, v11.h[3] \n"
        "smlal2 v16.4s, v3.8h, v8.h[3] \n"
        "smlal2 v17.4s, v3.8h, v9.h[3] \n"
        "smlal2 v18.4s, v3.8h, v10.h[3] \n"
        "smlal2 v19.4s, v3.8h, v11.h[3] \n"

        "smlal v20.4s, v0.4h, v4.h[4] \n"
        "smlal v21.4s, v0.4h, v5.h[4] \n"
        "smlal v22.4s, v0.4h, v6.h[4] \n"
        "smlal v23.4s, v0.4h, v7.h[4] \n"

        "smlal2 v24.4s, v0.8h, v4.h[4] \n"
        "smlal2 v25.4s, v0.8h, v5.h[4] \n"
        "smlal2 v26.4s, v0.8h, v6.h[4] \n"
        "smlal2 v27.4s, v0.8h, v7.h[4] \n"
        "ld1 {v2.8h, v3.8h}, [%[a]], #32 \n" //load a2, a3

        "smlal v20.4s, v1.4h, v4.h[5] \n"
        "smlal v21.4s, v1.4h, v5.h[5] \n"
        "smlal v22.4s, v1.4h, v6.h[5] \n"
        "smlal v23.4s, v1.4h, v7.h[5] \n"

        "smlal2 v24.4s, v1.8h, v4.h[5] \n"
        "smlal2 v25.4s, v1.8h, v5.h[5] \n"
        "smlal2 v26.4s, v1.8h, v6.h[5] \n"
        "smlal2 v27.4s, v1.8h, v7.h[5] \n"

        "smlal v12.4s, v0.4h, v8.h[4] \n"
        "smlal v13.4s, v0.4h, v9.h[4] \n"
        "smlal v14.4s, v0.4h, v10.h[4] \n"
        "smlal v15.4s, v0.4h, v11.h[4] \n"

        "smlal2 v16.4s, v0.8h, v8.h[4] \n"
        "smlal2 v17.4s, v0.8h, v9.h[4] \n"
        "smlal2 v18.4s, v0.8h, v10.h[4] \n"
        "smlal2 v19.4s, v0.8h, v11.h[4] \n"

        "smlal v12.4s, v1.4h, v8.h[5] \n"
        "smlal v13.4s, v1.4h, v9.h[5] \n"
        "smlal v14.4s, v1.4h, v10.h[5] \n"
        "smlal v15.4s, v1.4h, v11.h[5] \n"

        "smlal2 v16.4s, v1.8h, v8.h[5] \n"
        "smlal2 v17.4s, v1.8h, v9.h[5] \n"
        "smlal2 v18.4s, v1.8h, v10.h[5] \n"
        "smlal2 v19.4s, v1.8h, v11.h[5] \n"

        "smlal v20.4s, v2.4h, v4.h[6] \n"
        "smlal v21.4s, v2.4h, v5.h[6] \n"
        "smlal v22.4s, v2.4h, v6.h[6] \n"
        "smlal v23.4s, v2.4h, v7.h[6] \n"
        "ld1 {v0.8h, v1.8h}, [%[a]], #32 \n" //load a0, a1
        "smlal2 v24.4s, v2.8h, v4.h[6] \n"
        "smlal2 v25.4s, v2.8h, v5.h[6] \n"
        "smlal2 v26.4s, v2.8h, v6.h[6] \n"
        "smlal2 v27.4s, v2.8h, v7.h[6] \n"
        "sub %[b], %[b], #128         \n"
        "add %[b], %[b], %[ldb]        \n"
        "smlal v20.4s, v3.4h, v4.h[7] \n"
        "smlal v21.4s, v3.4h, v5.h[7] \n"
        "smlal v22.4s, v3.4h, v6.h[7] \n"
        "smlal v23.4s, v3.4h, v7.h[7] \n"
        "smlal2 v24.4s, v3.8h, v4.h[7] \n"
        "smlal2 v25.4s, v3.8h, v5.h[7] \n"
        "smlal2 v26.4s, v3.8h, v6.h[7] \n"
        "smlal2 v27.4s, v3.8h, v7.h[7] \n"
        "ld1 {v4.8h, v5.8h}, [%[b]], #32 \n" //load b0, b1
        "ld1 {v6.8h, v7.8h}, [%[b]], #32 \n" //load b2, b3

        "smlal v12.4s, v2.4h, v8.h[6] \n"
        "smlal v13.4s, v2.4h, v9.h[6] \n"
        "smlal v14.4s, v2.4h, v10.h[6] \n"
        "smlal v15.4s, v2.4h, v11.h[6] \n"
        "smlal2 v16.4s, v2.8h, v8.h[6] \n"
        "smlal2 v17.4s, v2.8h, v9.h[6] \n"
        "smlal2 v18.4s, v2.8h, v10.h[6] \n"
        "smlal2 v19.4s, v2.8h, v11.h[6] \n"
        "subs   %w[cnt], %w[cnt], #1      \n"

        "smlal v12.4s, v3.4h, v8.h[7] \n"
        "smlal v13.4s, v3.4h, v9.h[7] \n"
        "smlal v14.4s, v3.4h, v10.h[7] \n"
        "smlal v15.4s, v3.4h, v11.h[7] \n"
        "smlal2 v16.4s, v3.8h, v8.h[7] \n"
        "smlal2 v17.4s, v3.8h, v9.h[7] \n"
        "smlal2 v18.4s, v3.8h, v10.h[7] \n"
        "smlal2 v19.4s, v3.8h, v11.h[7] \n"

        "bne 1b                         \n"
        "2:                             \n"
        "dup v10.4s, %w[beta]        \n"
        "mul v20.4s, v20.4s, v10.4s \n"
        "mul v21.4s, v21.4s, v10.4s \n"
        "mul v22.4s, v22.4s, v10.4s \n"
        "mul v23.4s, v23.4s, v10.4s \n"
        "mul v24.4s, v24.4s, v10.4s \n"
        "mul v25.4s, v25.4s, v10.4s \n"
        "mul v26.4s, v26.4s, v10.4s \n"
        "mul v27.4s, v27.4s, v10.4s \n"
        "mul v12.4s, v12.4s, v10.4s \n"
        "mul v13.4s, v13.4s, v10.4s \n"
        "mul v14.4s, v14.4s, v10.4s \n"
        "mul v15.4s, v15.4s, v10.4s \n"
        "mul v16.4s, v16.4s, v10.4s \n"
        "mul v17.4s, v17.4s, v10.4s \n"
        "mul v18.4s, v18.4s, v10.4s \n"
        "mul v19.4s, v19.4s, v10.4s \n"
        "stp q20, q24, [%[c]], #32 \n"
        "stp q21, q25, [%[c]], #32 \n"
        "stp q22, q26, [%[c]], #32 \n"
        "stp q23, q27, [%[c]], #32 \n"
        "stp q12, q16, [%[c]], #32 \n"
        "stp q13, q17, [%[c]], #32 \n"
        "stp q14, q18, [%[c]], #32 \n"
        "stp q15, q19, [%[c]], #32 \n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [ldb] "r" (ldb_byte),
          [beta] "r" (beta)
        : "v0", "v1", "v2", "v3", "v4","v5", "v6", "v7", "v8", "v9",
          "v10", "v11", "13", "14", "15", "16", "17", "18", "19","v20",
           "v21", "v22", "v23", "v24", "v25", "v26", "v27", "cc", "memory"
      );
      // clang format on
      b += 64;
    }
    for (; n > 3; n -= 4) {
      int cnt = kcnt;
      const int16_t* a_ptr = A_packed;
      const int16_t* b_ptr = b;
      // clang-format off
      asm volatile(
        "ld1 {v0.8h, v1.8h}, [%[a]], #32 \n"
        "ld1 {v4.8h, v5.8h}, [%[b]], #32 \n"

        "smull v8.4s, v0.4h, v4.h[0] \n"
        "smull v9.4s, v0.4h, v5.h[0] \n"
        "ld1 {v6.8h, v7.8h}, [%[b]], #32 \n"
        "smull2 v10.4s, v0.8h, v4.h[0] \n"
        "smull2 v11.4s, v0.8h, v5.h[0] \n"

        "smlal v8.4s, v1.4h, v4.h[1] \n"
        "smlal v9.4s, v1.4h, v5.h[1] \n"
        "smlal2 v10.4s, v1.8h, v4.h[1] \n"
        "smlal2 v11.4s, v1.8h, v5.h[1] \n"
        "ld1 {v2.8h, v3.8h}, [%[a]], #32 \n"

        "smull v12.4s, v0.4h, v6.h[0] \n"
        "smull v13.4s, v0.4h, v7.h[0] \n"
        "smull2 v14.4s, v0.8h, v6.h[0] \n"
        "smull2 v15.4s, v0.8h, v7.h[0] \n"
        "smlal v12.4s, v1.4h, v6.h[1] \n"
        "smlal v13.4s, v1.4h, v7.h[1] \n"
        "smlal2 v14.4s, v1.8h, v6.h[1] \n"
        "smlal2 v15.4s, v1.8h, v7.h[1] \n"

        "smlal v8.4s, v2.4h, v4.h[2] \n"
        "smlal v9.4s, v2.4h, v5.h[2] \n"
        "ld1 {v0.8h, v1.8h}, [%[a]], #32 \n"
        "smlal2 v10.4s, v2.8h, v4.h[2] \n"
        "smlal2 v11.4s, v2.8h, v5.h[2] \n"
        "smlal v8.4s, v3.4h, v4.h[3] \n"
        "smlal v9.4s, v3.4h, v5.h[3] \n"
        "smlal2 v10.4s, v3.8h, v4.h[3] \n"
        "smlal2 v11.4s, v3.8h, v5.h[3] \n"

        "smlal v12.4s, v2.4h, v6.h[2] \n"
        "smlal v13.4s, v2.4h, v7.h[2] \n"
        "smlal2 v14.4s, v2.8h, v6.h[2] \n"
        "smlal2 v15.4s, v2.8h, v7.h[2] \n"
        "smlal v12.4s, v3.4h, v6.h[3] \n"
        "smlal v13.4s, v3.4h, v7.h[3] \n"
        "smlal2 v14.4s, v3.8h, v6.h[3] \n"
        "smlal2 v15.4s, v3.8h, v7.h[3] \n"

        "smlal v8.4s, v0.4h, v4.h[4] \n"
        "smlal v9.4s, v0.4h, v5.h[4] \n"
        "smlal2 v10.4s, v0.8h, v4.h[4] \n"
        "smlal2 v11.4s, v0.8h, v5.h[4] \n"

        "smlal v8.4s, v1.4h, v4.h[5] \n"
        "smlal v9.4s, v1.4h, v5.h[5] \n"
        "smlal2 v10.4s, v1.8h, v4.h[5] \n"
        "smlal2 v11.4s, v1.8h, v5.h[5] \n"
        "ld1 {v2.8h, v3.8h}, [%[a]], #32 \n"

        "smlal v12.4s, v0.4h, v6.h[4] \n"
        "smlal v13.4s, v0.4h, v7.h[4] \n"
        "smlal2 v14.4s, v0.8h, v6.h[4] \n"
        "smlal2 v15.4s, v0.8h, v7.h[4] \n"
        "smlal v12.4s, v1.4h, v6.h[5] \n"
        "smlal v13.4s, v1.4h, v7.h[5] \n"
        "smlal2 v14.4s, v1.8h, v6.h[5] \n"
        "smlal2 v15.4s, v1.8h, v7.h[5] \n"

        "smlal v8.4s, v2.4h, v4.h[6] \n"
        "smlal v9.4s, v2.4h, v5.h[6] \n"
        "ld1 {v0.8h, v1.8h}, [%[a]], #32 \n"
        "smlal2 v10.4s, v2.8h, v4.h[6] \n"
        "smlal2 v11.4s, v2.8h, v5.h[6] \n"
        "smlal v8.4s, v3.4h, v4.h[7] \n"
        "smlal v9.4s, v3.4h, v5.h[7] \n"
        "smlal2 v10.4s, v3.8h, v4.h[7] \n"
        "smlal2 v11.4s, v3.8h, v5.h[7] \n"
        "sub %[b], %[b], #64           \n"
        "add %[b], %[b], %[ldb]        \n"

        "smlal v12.4s, v2.4h, v6.h[6] \n"
        "smlal v13.4s, v2.4h, v7.h[6] \n"
        "subs %w[cnt], %w[cnt], #1        \n"
        "ld1 {v4.8h, v5.8h}, [%[b]], #32 \n"
        "smlal2 v14.4s, v2.8h, v6.h[6] \n"
        "smlal2 v15.4s, v2.8h, v7.h[6] \n"
        "smlal v12.4s, v3.4h, v6.h[7] \n"
        "smlal v13.4s, v3.4h, v7.h[7] \n"
        "smlal2 v14.4s, v3.8h, v6.h[7] \n"
        "smlal2 v15.4s, v3.8h, v7.h[7] \n"

        "beq 2f \n"
        "1: \n"
        "smlal v8.4s, v0.4h, v4.h[0] \n"
        "smlal v9.4s, v0.4h, v5.h[0] \n"
        "ld1 {v6.8h, v7.8h}, [%[b]], #32 \n"
        "smlal2 v10.4s, v0.8h, v4.h[0] \n"
        "smlal2 v11.4s, v0.8h, v5.h[0] \n"

        "smlal v8.4s, v1.4h, v4.h[1] \n"
        "smlal v9.4s, v1.4h, v5.h[1] \n"
        "smlal2 v10.4s, v1.8h, v4.h[1] \n"
        "smlal2 v11.4s, v1.8h, v5.h[1] \n"
        "ld1 {v2.8h, v3.8h}, [%[a]], #32 \n"

        "smlal v12.4s, v0.4h, v6.h[0] \n"
        "smlal v13.4s, v0.4h, v7.h[0] \n"
        "smlal2 v14.4s, v0.8h, v6.h[0] \n"
        "smlal2 v15.4s, v0.8h, v7.h[0] \n"
        "smlal v12.4s, v1.4h, v6.h[1] \n"
        "smlal v13.4s, v1.4h, v7.h[1] \n"
        "smlal2 v14.4s, v1.8h, v6.h[1] \n"
        "smlal2 v15.4s, v1.8h, v7.h[1] \n"

        "smlal v8.4s, v2.4h, v4.h[2] \n"
        "smlal v9.4s, v2.4h, v5.h[2] \n"
        "ld1 {v0.8h, v1.8h}, [%[a]], #32 \n"
        "smlal2 v10.4s, v2.8h, v4.h[2] \n"
        "smlal2 v11.4s, v2.8h, v5.h[2] \n"
        "smlal v8.4s, v3.4h, v4.h[3] \n"
        "smlal v9.4s, v3.4h, v5.h[3] \n"
        "smlal2 v10.4s, v3.8h, v4.h[3] \n"
        "smlal2 v11.4s, v3.8h, v5.h[3] \n"

        "smlal v12.4s, v2.4h, v6.h[2] \n"
        "smlal v13.4s, v2.4h, v7.h[2] \n"
        "smlal2 v14.4s, v2.8h, v6.h[2] \n"
        "smlal2 v15.4s, v2.8h, v7.h[2] \n"
        "smlal v12.4s, v3.4h, v6.h[3] \n"
        "smlal v13.4s, v3.4h, v7.h[3] \n"
        "smlal2 v14.4s, v3.8h, v6.h[3] \n"
        "smlal2 v15.4s, v3.8h, v7.h[3] \n"

        "smlal v8.4s, v0.4h, v4.h[4] \n"
        "smlal v9.4s, v0.4h, v5.h[4] \n"
        "ld1 {v2.8h, v3.8h}, [%[a]], #32 \n"
        "smlal2 v10.4s, v0.8h, v4.h[4] \n"
        "smlal2 v11.4s, v0.8h, v5.h[4] \n"

        "smlal v8.4s, v1.4h, v4.h[5] \n"
        "smlal v9.4s, v1.4h, v5.h[5] \n"
        "smlal2 v10.4s, v1.8h, v4.h[5] \n"
        "smlal2 v11.4s, v1.8h, v5.h[5] \n"

        "smlal v12.4s, v0.4h, v6.h[4] \n"
        "smlal v13.4s, v0.4h, v7.h[4] \n"
        "smlal2 v14.4s, v0.8h, v6.h[4] \n"
        "smlal2 v15.4s, v0.8h, v7.h[4] \n"
        "smlal v12.4s, v1.4h, v6.h[5] \n"
        "smlal v13.4s, v1.4h, v7.h[5] \n"
        "smlal2 v14.4s, v1.8h, v6.h[5] \n"
        "smlal2 v15.4s, v1.8h, v7.h[5] \n"

        "smlal v8.4s, v2.4h, v4.h[6] \n"
        "smlal v9.4s, v2.4h, v5.h[6] \n"
        "ld1 {v0.8h, v1.8h}, [%[a]], #32 \n"
        "smlal2 v10.4s, v2.8h, v4.h[6] \n"
        "smlal2 v11.4s, v2.8h, v5.h[6] \n"
        "smlal v8.4s, v3.4h, v4.h[7] \n"
        "smlal v9.4s, v3.4h, v5.h[7] \n"
        "smlal2 v10.4s, v3.8h, v4.h[7] \n"
        "smlal2 v11.4s, v3.8h, v5.h[7] \n"
        "sub %[b], %[b], #64           \n"
        "add %[b], %[b], %[ldb]        \n"

        "smlal v12.4s, v2.4h, v6.h[6] \n"
        "smlal v13.4s, v2.4h, v7.h[6] \n"
        "subs %w[cnt], %w[cnt], #1        \n"
        "ld1 {v4.8h, v5.8h}, [%[b]], #32 \n"
        "smlal2 v14.4s, v2.8h, v6.h[6] \n"
        "smlal2 v15.4s, v2.8h, v7.h[6] \n"
        "smlal v12.4s, v3.4h, v6.h[7] \n"
        "smlal v13.4s, v3.4h, v7.h[7] \n"
        "smlal2 v14.4s, v3.8h, v6.h[7] \n"
        "smlal2 v15.4s, v3.8h, v7.h[7] \n"

        "bne 1b \n"
        "2: \n"

        "dup v20.4s, %w[beta]        \n"
        "mul v8.4s, v8.4s, v20.4s \n"
        "mul v9.4s, v9.4s, v20.4s \n"
        "mul v10.4s, v10.4s, v20.4s \n"
        "mul v11.4s, v11.4s, v20.4s \n"
        "mul v12.4s, v12.4s, v20.4s \n"
        "mul v13.4s, v13.4s, v20.4s \n"
        "mul v14.4s, v14.4s, v20.4s \n"
        "mul v15.4s, v15.4s, v20.4s \n"
        "stp q8, q10, [%[c]], #32 \n"
        "stp q9, q11, [%[c]], #32 \n"
        "stp q12, q14, [%[c]], #32 \n"
        "stp q13, q15, [%[c]], #32 \n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [ldb] "r" (ldb_byte),
          [beta] "r" (beta)
        : "v0", "v1", "v2", "v3", "v4","v5", "v6", "v7", "v8", "v9",
          "v10", "v11","v12", "v13", "v14", "v15", "cc", "memory"
      );
      // clang-format on
      b += 32;
    }
    for (; n > 0; --n) {
      int cnt = kcnt;
      const int16_t* a_ptr = A_packed;
      const int16_t* b_ptr = b;
      // clang-format off
      asm volatile(
        "ld1 {v0.8h, v1.8h}, [%[a]], #32 \n"
        "ld1 {v4.8h}, [%[b]], #16 \n"
        "ld1 {v2.8h, v3.8h}, [%[a]], #32 \n"
        "smull v5.4s, v0.4h, v4.h[0] \n"
        "smull2 v6.4s, v0.8h, v4.h[0] \n"
        "ld1 {v10.8h, v11.8h}, [%[a]], #32 \n"
        "smlal v5.4s, v1.4h, v4.h[1] \n"
        "smlal2 v6.4s, v1.8h, v4.h[1] \n"
        "ld1 {v12.8h, v13.8h}, [%[a]], #32 \n"
        "smlal v5.4s, v2.4h, v4.h[2] \n"
        "smlal2 v6.4s, v2.8h, v4.h[2] \n"
        "smlal v5.4s, v3.4h, v4.h[3] \n"
        "smlal2 v6.4s, v3.8h, v4.h[3] \n"
        "sub %[b], %[b], #16 \n"
        "add %[b], %[b], %[ldb] \n"
        "smlal v5.4s, v10.4h, v4.h[4] \n"
        "smlal2 v6.4s, v10.8h, v4.h[4] \n"
        "smlal v5.4s, v11.4h, v4.h[5] \n"
        "smlal2 v6.4s, v11.8h, v4.h[5] \n"
        "subs %w[cnt], %w[cnt], #1 \n"
        "ld1 {v0.8h, v1.8h}, [%[a]], #32 \n"
        "smlal v5.4s, v12.4h, v4.h[6] \n"
        "smlal2 v6.4s, v12.8h, v4.h[6] \n"
        "smlal v5.4s, v13.4h, v4.h[7] \n"
        "smlal2 v6.4s, v13.8h, v4.h[7] \n"

        "beq 2f \n"
        "1: \n"
        "ld1 {v4.8h}, [%[b]], #16 \n"
        "ld1 {v2.8h, v3.8h}, [%[a]], #32 \n"
        "smlal v5.4s, v0.4h, v4.h[0] \n"
        "smlal2 v6.4s, v0.8h, v4.h[0] \n"
        "ld1 {v10.8h, v11.8h}, [%[a]], #32 \n"
        "smlal v5.4s, v1.4h, v4.h[1] \n"
        "smlal2 v6.4s, v1.8h, v4.h[1] \n"
        "ld1 {v12.8h, v13.8h}, [%[a]], #32 \n"
        "smlal v5.4s, v2.4h, v4.h[2] \n"
        "smlal2 v6.4s, v2.8h, v4.h[2] \n"
        "smlal v5.4s, v3.4h, v4.h[3] \n"
        "smlal2 v6.4s, v3.8h, v4.h[3] \n"
        "sub %[b], %[b], #16 \n"
        "add %[b], %[b], %[ldb] \n"
        "smlal v5.4s, v10.4h, v4.h[4] \n"
        "smlal2 v6.4s, v10.8h, v4.h[4] \n"
        "smlal v5.4s, v11.4h, v4.h[5] \n"
        "smlal2 v6.4s, v11.8h, v4.h[5] \n"
        "subs %w[cnt], %w[cnt], #1 \n"
        "ld1 {v0.8h, v1.8h}, [%[a]], #32 \n"
        "smlal v5.4s, v12.4h, v4.h[6] \n"
        "smlal2 v6.4s, v12.8h, v4.h[6] \n"
        "smlal v5.4s, v13.4h, v4.h[7] \n"
        "smlal2 v6.4s, v13.8h, v4.h[7] \n"
        "bne 1b \n"

        "2: \n"
        "dup v10.4s, %w[beta]        \n"
        "mul v5.4s, v5.4s, v10.4s \n"
        "mul v6.4s, v6.4s, v10.4s \n"
        "st1 {v5.4s, v6.4s}, [%[c]], #32 \n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [ldb] "r" (ldb_byte),
          [beta] "r" (beta)
        : "v0", "v1", "v2", "v3", "v4","v5", "v6", "cc", "memory"
      );
      // clang-format on
      b += 8;
    }
#else
    for (; n > 3; n -= 4) {
      int cnt = kcnt;
      const int16_t* a_ptr = A_packed;
      const int16_t* b_ptr = b;
      // clang-format off
      asm volatile (
        "vld1.16 {d0-d3}, [%[b]]!  \n"
        "vld1.16 {d8-d11}, [%[a]]! \n"
        "vld1.16 {d4-d7}, [%[b]]! \n"
        "vmull.s16 q8, d8, d0[0] \n"
        "vmull.s16 q9, d8, d2[0] \n"
        "vld1.16 {d12-d15}, [%[a]]! \n"
        "vmull.s16 q10, d9, d0[0] \n"
        "vmull.s16 q11, d9, d2[0] \n"
        "vmlal.s16 q8, d10, d0[1] \n"
        "vmlal.s16 q9, d10, d2[1] \n"
        "vmlal.s16 q10, d11, d0[1] \n"
        "vmlal.s16 q11, d11, d2[1] \n"
        "vmull.s16 q12, d8, d4[0] \n"
        "vmull.s16 q13, d8, d6[0] \n"
        "vmull.s16 q14, d9, d4[0] \n"
        "vmull.s16 q15, d9, d6[0] \n"
        "vmlal.s16 q12, d10, d4[1] \n"
        "vmlal.s16 q13, d10, d6[1] \n"
        "vmlal.s16 q14, d11, d4[1] \n"
        "vmlal.s16 q15, d11, d6[1] \n"

        "vmlal.s16 q8, d12, d0[2] \n"
        "vmlal.s16 q9, d12, d2[2] \n"
        "vld1.16 {d8-d11}, [%[a]]! \n"
        "vmlal.s16 q10, d13, d0[2] \n"
        "vmlal.s16 q11, d13, d2[2] \n"
        "vmlal.s16 q8, d14, d0[3] \n"
        "vmlal.s16 q9, d14, d2[3] \n"
        "vmlal.s16 q10, d15, d0[3] \n"
        "vmlal.s16 q11, d15, d2[3] \n"

        "vmlal.s16 q12, d12, d4[2] \n"
        "vmlal.s16 q13, d12, d6[2] \n"
        "vmlal.s16 q14, d13, d4[2] \n"
        "vmlal.s16 q15, d13, d6[2] \n"
        "vmlal.s16 q12, d14, d4[3] \n"
        "vmlal.s16 q13, d14, d6[3] \n"
        "vmlal.s16 q14, d15, d4[3] \n"
        "vmlal.s16 q15, d15, d6[3] \n"

        "sub %[b], %[b], #64   \n"
        "add %[b], %[b], %[ldb]   \n"
        "vld1.16 {d12-d15}, [%[a]]! \n"
        "vmlal.s16 q8, d8, d1[0] \n"
        "vmlal.s16 q9, d8, d3[0] \n"
        "vmlal.s16 q10, d9, d1[0] \n"
        "vmlal.s16 q11, d9, d3[0] \n"
        "vmlal.s16 q8, d10, d1[1] \n"
        "vmlal.s16 q9, d10, d3[1] \n"
        "vmlal.s16 q10, d11, d1[1] \n"
        "vmlal.s16 q11, d11, d3[1] \n"
        "vmlal.s16 q8, d12, d1[2] \n"
        "vmlal.s16 q9, d12, d3[2] \n"
        "vmlal.s16 q10, d13, d1[2] \n"
        "vmlal.s16 q11, d13, d3[2] \n"
        "vmlal.s16 q8, d14, d1[3] \n"
        "vmlal.s16 q9, d14, d3[3] \n"
        "vmlal.s16 q10, d15, d1[3] \n"
        "vmlal.s16 q11, d15, d3[3] \n"
        "vld1.16 {d0-d3}, [%[b]]!  \n"
        "vmlal.s16 q12, d8, d5[0] \n"
        "vmlal.s16 q13, d8, d7[0] \n"
        "vmlal.s16 q14, d9, d5[0] \n"
        "vmlal.s16 q15, d9, d7[0] \n"
        "vmlal.s16 q12, d10, d5[1] \n"
        "vmlal.s16 q13, d10, d7[1] \n"
        "subs %[cnt], %[cnt], #1   \n"
        "vmlal.s16 q14, d11, d5[1] \n"
        "vmlal.s16 q15, d11, d7[1] \n"
        "vld1.16 {d8-d11}, [%[a]]! \n"
        "vmlal.s16 q12, d12, d5[2] \n"
        "vmlal.s16 q13, d12, d7[2] \n"
        "vmlal.s16 q14, d13, d5[2] \n"
        "vmlal.s16 q15, d13, d7[2] \n"
        "vmlal.s16 q12, d14, d5[3] \n"
        "vmlal.s16 q13, d14, d7[3] \n"
        "vmlal.s16 q14, d15, d5[3] \n"
        "vmlal.s16 q15, d15, d7[3] \n"

        "beq 2f \n"
        "1: \n"
        "vld1.16 {d4-d7}, [%[b]]! \n"
        "vmlal.s16 q8, d8, d0[0] \n"
        "vmlal.s16 q9, d8, d2[0] \n"
        "vld1.16 {d12-d15}, [%[a]]! \n"
        "vmlal.s16 q10, d9, d0[0] \n"
        "vmlal.s16 q11, d9, d2[0] \n"
        "vmlal.s16 q8, d10, d0[1] \n"
        "vmlal.s16 q9, d10, d2[1] \n"
        "vmlal.s16 q10, d11, d0[1] \n"
        "vmlal.s16 q11, d11, d2[1] \n"
        "vmlal.s16 q12, d8, d4[0] \n"
        "vmlal.s16 q13, d8, d6[0] \n"
        "vmlal.s16 q14, d9, d4[0] \n"
        "vmlal.s16 q15, d9, d6[0] \n"
        "vmlal.s16 q12, d10, d4[1] \n"
        "vmlal.s16 q13, d10, d6[1] \n"
        "vmlal.s16 q14, d11, d4[1] \n"
        "vmlal.s16 q15, d11, d6[1] \n"

        "vmlal.s16 q8, d12, d0[2] \n"
        "vmlal.s16 q9, d12, d2[2] \n"
        "vld1.16 {d8-d11}, [%[a]]! \n"
        "vmlal.s16 q10, d13, d0[2] \n"
        "vmlal.s16 q11, d13, d2[2] \n"
        "vmlal.s16 q8, d14, d0[3] \n"
        "vmlal.s16 q9, d14, d2[3] \n"
        "vmlal.s16 q10, d15, d0[3] \n"
        "vmlal.s16 q11, d15, d2[3] \n"

        "vmlal.s16 q12, d12, d4[2] \n"
        "vmlal.s16 q13, d12, d6[2] \n"
        "vmlal.s16 q14, d13, d4[2] \n"
        "vmlal.s16 q15, d13, d6[2] \n"
        "vmlal.s16 q12, d14, d4[3] \n"
        "vmlal.s16 q13, d14, d6[3] \n"
        "vmlal.s16 q14, d15, d4[3] \n"
        "vmlal.s16 q15, d15, d6[3] \n"

        "sub %[b], %[b], #64   \n"
        "add %[b], %[b], %[ldb]   \n"
        "vld1.16 {d12-d15}, [%[a]]! \n"
        "vmlal.s16 q8, d8, d1[0] \n"
        "vmlal.s16 q9, d8, d3[0] \n"
        "vmlal.s16 q10, d9, d1[0] \n"
        "vmlal.s16 q11, d9, d3[0] \n"
        "vmlal.s16 q8, d10, d1[1] \n"
        "vmlal.s16 q9, d10, d3[1] \n"
        "vmlal.s16 q10, d11, d1[1] \n"
        "vmlal.s16 q11, d11, d3[1] \n"
        "vmlal.s16 q8, d12, d1[2] \n"
        "vmlal.s16 q9, d12, d3[2] \n"
        "vmlal.s16 q10, d13, d1[2] \n"
        "vmlal.s16 q11, d13, d3[2] \n"
        "vmlal.s16 q8, d14, d1[3] \n"
        "vmlal.s16 q9, d14, d3[3] \n"
        "vmlal.s16 q10, d15, d1[3] \n"
        "vmlal.s16 q11, d15, d3[3] \n"
        "vld1.16 {d0-d3}, [%[b]]!  \n"
        "vmlal.s16 q12, d8, d5[0] \n"
        "vmlal.s16 q13, d8, d7[0] \n"
        "vmlal.s16 q14, d9, d5[0] \n"
        "vmlal.s16 q15, d9, d7[0] \n"
        "vmlal.s16 q12, d10, d5[1] \n"
        "vmlal.s16 q13, d10, d7[1] \n"
        "subs %[cnt], %[cnt], #1   \n"
        "vmlal.s16 q14, d11, d5[1] \n"
        "vmlal.s16 q15, d11, d7[1] \n"
        "vld1.16 {d8-d11}, [%[a]]! \n"
        "vmlal.s16 q12, d12, d5[2] \n"
        "vmlal.s16 q13, d12, d7[2] \n"
        "vmlal.s16 q14, d13, d5[2] \n"
        "vmlal.s16 q15, d13, d7[2] \n"
        "vmlal.s16 q12, d14, d5[3] \n"
        "vmlal.s16 q13, d14, d7[3] \n"
        "vmlal.s16 q14, d15, d5[3] \n"
        "vmlal.s16 q15, d15, d7[3] \n"

        "bne 1b \n"
        "2: \n"
        "vdup.32 q7, %[beta]          \n"
        "vmul.i32 q8, q8, q7          \n"
        "vmul.i32 q9, q9, q7          \n"
        "vmul.i32 q10, q10, q7        \n"
        "vmul.i32 q11, q11, q7        \n"
        "vmul.i32 q12, q12, q7        \n"
        "vmul.i32 q13, q13, q7        \n"
        "vmul.i32 q14, q14, q7        \n"
        "vmul.i32 q15, q15, q7        \n"
        "vst1.32 {d16-d17}, [%[c]]! \n"
        "vst1.32 {d20-d21}, [%[c]]! \n"
        "vst1.32 {d18-d19}, [%[c]]! \n"
        "vst1.32 {d22-d23}, [%[c]]! \n"
        "vst1.32 {d24-d25}, [%[c]]! \n"
        "vst1.32 {d28-d29}, [%[c]]! \n"
        "vst1.32 {d26-d27}, [%[c]]! \n"
        "vst1.32 {d30-d31}, [%[c]]! \n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [ldb] "r" (ldb_byte),
          [beta] "r" (beta)
        : "q0", "q1", "q2", "q3", "q4","q5", "q6", "q7", "q8",
          "q9", "q10", "q11", "q12", "q13", "q14", "q15", "cc", "memory"
      );
      // clang format on
      b += 32;
     }
    for (; n > 0; --n) {
      int cnt = kcnt;
      const int16_t* a_ptr = A_packed;
      const int16_t* b_ptr = b;
      // clang format off
      asm volatile (
        "vld1.16 {d0-d1}, [%[b]]! \n"
        "vld1.16 {d4-d7}, [%[a]]! \n"
        "vld1.16 {d8-d11}, [%[a]]! \n"
        "vmull.s16 q8, d4, d0[0] \n"
        "vmull.s16 q9, d5, d0[0] \n"
        "sub %[b], %[b], #16   \n"
        "vmlal.s16 q8, d6, d0[1] \n"
        "vmlal.s16 q9, d7, d0[1] \n"
        "add %[b], %[b], %[ldb]   \n"
        "subs %[cnt], %[cnt], #1   \n"

        "vld1.16 {d4-d7}, [%[a]]! \n"
        "vmlal.s16 q8, d8, d0[2] \n"
        "vmlal.s16 q9, d9, d0[2] \n"
        "vmlal.s16 q8, d10, d0[3] \n"
        "vmlal.s16 q9, d11, d0[3] \n"
        "vld1.16 {d8-d11}, [%[a]]! \n"

        "vmlal.s16 q8, d4, d1[0] \n"
        "vmlal.s16 q9, d5, d1[0] \n"
        "vmlal.s16 q8, d6, d1[1] \n"
        "vmlal.s16 q9, d7, d1[1] \n"
        "vld1.16 {d4-d7}, [%[a]]! \n"
        "vmlal.s16 q8, d8, d1[2] \n"
        "vmlal.s16 q9, d9, d1[2] \n"
        "vmlal.s16 q8, d10, d1[3] \n"
        "vmlal.s16 q9, d11, d1[3] \n"
        "beq 2f \n"
        "1:\n"
        "vld1.16 {d0-d1}, [%[b]]! \n"
        "vld1.16 {d8-d11}, [%[a]]! \n"
        "vmlal.s16 q8, d4, d0[0] \n"
        "vmlal.s16 q9, d5, d0[0] \n"
        "sub %[b], %[b], #16   \n"
        "vmlal.s16 q8, d6, d0[1] \n"
        "vmlal.s16 q9, d7, d0[1] \n"
        "add %[b], %[b], %[ldb]   \n"
        "subs %[cnt], %[cnt], #1   \n"

        "vld1.16 {d4-d7}, [%[a]]! \n"
        "vmlal.s16 q8, d8, d0[2] \n"
        "vmlal.s16 q9, d9, d0[2] \n"
        "vmlal.s16 q8, d10, d0[3] \n"
        "vmlal.s16 q9, d11, d0[3] \n"
        "vld1.16 {d8-d11}, [%[a]]! \n"

        "vmlal.s16 q8, d4, d1[0] \n"
        "vmlal.s16 q9, d5, d1[0] \n"
        "vmlal.s16 q8, d6, d1[1] \n"
        "vmlal.s16 q9, d7, d1[1] \n"
        "vld1.16 {d4-d7}, [%[a]]! \n"
        "vmlal.s16 q8, d8, d1[2] \n"
        "vmlal.s16 q9, d9, d1[2] \n"
        "vmlal.s16 q8, d10, d1[3] \n"
        "vmlal.s16 q9, d11, d1[3] \n"
        "bne 1b \n"
        "2: \n"
        "vdup.32 q7, %[beta]       \n"
        "vmul.i32 q8, q8, q7        \n"
        "vmul.i32 q9, q9, q7        \n"
        "vst1.32 {d16-d19}, [%[c]]! \n" 
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [ldb] "r" (ldb_byte),
          [beta] "r" (beta)
        : "q0", "q1", "q2", "q3", "q4","q5", "q6", "q7", "q8",
          "q9", "cc", "memory"
      );
      // clang-format on
      b += 8;
    }
#endif
    A_packed += lda;
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
  if (N > 16) {
    sgemm_prepack_c4_common(
        M, N, K, A_packed, B, C, bias, has_bias, has_relu, ctx);
  } else {
    sgemm_prepack_c4_small(
        M, N, K, A_packed, B, C, bias, has_bias, has_relu, ctx);
  }
}

void sgemm_prepack_c4_a35(int M,
                          int N,
                          int K,
                          const float* A_packed,
                          const float* B,
                          float* C,
                          const float* bias,
                          bool has_bias,
                          bool has_relu,
                          ARMContext* ctx) {
  if (N > 16) {
    sgemm_prepack_c4_common_a35(
        M, N, K, A_packed, B, C, bias, has_bias, has_relu, ctx);
  } else {
    sgemm_prepack_c4_small_a35(
        M, N, K, A_packed, B, C, bias, has_bias, has_relu, ctx);
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
