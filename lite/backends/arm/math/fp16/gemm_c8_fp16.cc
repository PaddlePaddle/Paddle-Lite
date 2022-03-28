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

#include "lite/backends/arm/math/fp16/gemm_c8_fp16.h"
#include <arm_neon.h>
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

void loadb_c8(float16_t* out,
              const float16_t* in,
              const int xstart,
              const int xend,
              const int k_round,
              const int n) {
  const int xlen = (xend - xstart + NBLOCK_C8 - 1) / NBLOCK_C8 * NBLOCK_C8;
  int xloop = xlen / NBLOCK_C8;
  bool flag_remain = n < (xstart + xlen);
  int remain = 0;
  int remain4 = 0;
  int remain1 = 0;
  if (!flag_remain) {
    remain = (n - xstart) - (xloop - 1) * NBLOCK_C8;
    remain4 = remain >> 2;
    remain1 = remain & 3;
    xloop -= 1;
  }
  const int ldo = NBLOCK_C8 * k_round;
  const int kloop = k_round >> 3;
  in += xstart * 8;
  if (xloop > 0) {
    LITE_PARALLEL_BEGIN(i, tid, kloop) {
      float16_t* out_ptr = out + 8 * NBLOCK_C8 * i;
      const float16_t* in_ptr = in + i * 8 * n;
      for (int j = 0; j < xloop; ++j) {
        float16_t* out_p = out_ptr + j * ldo;
        vst1q_f16(out_p, vld1q_f16(in_ptr));
        vst1q_f16(out_p + 8, vld1q_f16(in_ptr + 8));
        vst1q_f16(out_p + 16, vld1q_f16(in_ptr + 16));
        vst1q_f16(out_p + 24, vld1q_f16(in_ptr + 24));
        vst1q_f16(out_p + 32, vld1q_f16(in_ptr + 32));
        vst1q_f16(out_p + 40, vld1q_f16(in_ptr + 40));
        vst1q_f16(out_p + 48, vld1q_f16(in_ptr + 48));
        vst1q_f16(out_p + 56, vld1q_f16(in_ptr + 56));
        in_ptr += 64;
        out_p += 64;
      }
    }
    LITE_PARALLEL_END();
  }
  float16_t* out_remain4 = out + xloop * k_round * NBLOCK_C8;
  const float16_t* in_remain4 = in + xloop * NBLOCK_C8 * 8;
  if (remain4) {
    LITE_PARALLEL_BEGIN(i, tid, kloop) {
      float16_t* out_ptr = out_remain4 + 32 * i;
      const float16_t* in_ptr = in_remain4 + i * 8 * n;
      vst1q_f16(out_ptr, vld1q_f16(in_ptr));
      vst1q_f16(out_ptr + 8, vld1q_f16(in_ptr + 8));
      vst1q_f16(out_ptr + 16, vld1q_f16(in_ptr + 16));
      vst1q_f16(out_ptr + 24, vld1q_f16(in_ptr + 24));
      in_ptr += 32;
    }
    LITE_PARALLEL_END();
  }
  float16_t* out_remain1 = out_remain4 + remain4 * k_round * 4;
  const float16_t* in_remain1 = in_remain4 + remain4 * 32;
  if (remain1) {
    LITE_PARALLEL_BEGIN(i, tid, kloop) {
      float16_t* out_ptr = out_remain1 + 4 * remain1 * i;
      const float16_t* in_ptr = in_remain1 + i * 8 * n;
      for (int j = 0; j < remain1; ++j) {
        float16x8_t vin = vld1q_f16(in_ptr);
        in_ptr += 8;
        vst1q_f16(out_ptr, vin);
        out_ptr += 8;
      }
    }
    LITE_PARALLEL_END();
  }
}

void gemm_prepack_c8_fp16_common(int M,
                                 int N,
                                 int K,
                                 const float16_t* A_packed,
                                 const float16_t* B,
                                 float16_t* C,
                                 ARMContext* ctx) {
  const int m_round = (M + 7) / 8 * 8;
  const int k_round = (K + 7) / 8 * 8;
  size_t l2_cache = ctx->llc_size() > 0 ? ctx->llc_size() : 512 * 1024;
  int threads = ctx->threads();
  auto workspace = ctx->workspace_data<float16_t>();
  // l2 = MBLOCK_C8 * K + K * x_block + x_block * MBLOCK_C8;
  int bchunk_w = (l2_cache - k_round * MBLOCK_C8) /
                 ((k_round + MBLOCK_C8) * sizeof(float16_t));
  bchunk_w = bchunk_w > N ? N : bchunk_w;
  bchunk_w = bchunk_w / NBLOCK_C8 * NBLOCK_C8;
  bchunk_w = bchunk_w > NBLOCK_C8 ? bchunk_w : NBLOCK_C8;
  int bchunk_loop = (N + bchunk_w - 1) / bchunk_w;

  const int h_loop = m_round >> 3;
  const int kcnt = (k_round + KBLOCK_C8 - 1) / KBLOCK_C8;
  const int ldc = N << 3;
  const int lda = k_round << 3;
  // bchunk_loop
  float16_t* c = C;
  for (int n = 0; n < bchunk_loop; ++n) {
    int x_start = n * bchunk_w;
    int x_end = x_start + bchunk_w;
    int w_loop = bchunk_w / NBLOCK_C8;
    int flag_remain = 0;
    int w_loop4 = 0;
    int remain = 0;
    if (x_end > N) {
      w_loop = (N - x_start) / NBLOCK_C8;
      int w_loop_rem = (N - x_start) & (NBLOCK_C8 - 1);
      w_loop4 = w_loop_rem >> 2;
      remain = w_loop_rem & 3;
      x_end = N;
      flag_remain = 1;
    }
    float16_t* bchunk = workspace;
    loadb_c8(bchunk, B, x_start, x_end, k_round, N);
    float16_t* cchunk = c + n * bchunk_w * 8;
    int has_remain = (n == bchunk_loop - 1) && flag_remain;
    LITE_PARALLEL_BEGIN(h, tid, h_loop) {
      const float16_t* ablock = A_packed + h * lda;
      const float16_t* bblock = bchunk;
      float16_t* cblock = cchunk + h * ldc;
      for (int w = 0; w < w_loop; ++w) {
        int cnt = kcnt;
        const float16_t* ablock_ptr = ablock;
// clang-format off
#ifdef __aarch64__
        asm volatile(
            "prfm  pldl1keep, [%[a]]         \n"
            "prfm  pldl1keep, [%[b]]         \n"
            /* load a0a1 to v0-v1  */
            "ldp  q16, q17, [%[a]], #32      \n"
            "movi  v8.8h, #0                 \n"
            "movi  v9.8h, #0                 \n"
            "movi  v10.8h, #0                \n"
            "movi  v11.8h, #0                \n"
            /* load b0b1 to v4-v5  */
            "ldp  q0, q1, [%[b]], #32        \n"
            "movi  v12.8h, #0                 \n"
            "movi  v13.8h, #0                 \n"
            "movi  v14.8h, #0                \n"
            "movi  v15.8h, #0                \n"
            /* load a0a1 to v0-v1  */
            "ldp  q18, q19, [%[a]], #32      \n"
            /* load b0b1 to v4-v5  */
            "ldp  q2, q3, [%[b]], #32        \n"
            "1:\n"
            "prfm  pldl1keep, [%[a]]         \n"
            "prfm  pldl1keep, [%[b]]         \n"
            "ldp   q4, q5, [%[b]], #32       \n"
            "fmla  v8.8h,  v16.8h, v0.h[0]   \n"
            "fmla  v9.8h,  v16.8h, v1.h[0]   \n"
            "ldp   q6, q7, [%[b]], #32       \n"
            "fmla  v10.8h, v16.8h, v2.h[0]   \n"
            "fmla  v11.8h, v16.8h, v3.h[0]   \n"
            "fmla  v12.8h, v16.8h, v4.h[0]   \n"
            "fmla  v13.8h, v16.8h, v5.h[0]   \n"
            "fmla  v14.8h, v16.8h, v6.h[0]   \n"
            "fmla  v15.8h, v16.8h, v7.h[0]   \n"

            "fmla  v8.8h,  v17.8h, v0.h[1]   \n"
            "fmla  v9.8h,  v17.8h, v1.h[1]   \n"
            "fmla  v10.8h, v17.8h, v2.h[1]   \n"
            "fmla  v11.8h, v17.8h, v3.h[1]   \n"
            "fmla  v12.8h, v17.8h, v4.h[1]   \n"
            "fmla  v13.8h, v17.8h, v5.h[1]   \n"
            "fmla  v14.8h, v17.8h, v6.h[1]   \n"
            "fmla  v15.8h, v17.8h, v7.h[1]   \n"
            "ldp  q16, q17, [%[a]], #32      \n"

            "fmla  v8.8h,  v18.8h, v0.h[2]   \n"
            "fmla  v9.8h,  v18.8h, v1.h[2]   \n"
            "fmla  v10.8h, v18.8h, v2.h[2]   \n"
            "fmla  v11.8h, v18.8h, v3.h[2]   \n"
            "fmla  v12.8h, v18.8h, v4.h[2]   \n"
            "fmla  v13.8h, v18.8h, v5.h[2]   \n"
            "fmla  v14.8h, v18.8h, v6.h[2]   \n"
            "fmla  v15.8h, v18.8h, v7.h[2]   \n"
            "fmla  v8.8h,  v19.8h, v0.h[3]   \n"
            "fmla  v9.8h,  v19.8h, v1.h[3]   \n"
            "fmla  v10.8h, v19.8h, v2.h[3]   \n"
            "fmla  v11.8h, v19.8h, v3.h[3]   \n"
            "fmla  v12.8h, v19.8h, v4.h[3]   \n"
            "fmla  v13.8h, v19.8h, v5.h[3]   \n"
            "fmla  v14.8h, v19.8h, v6.h[3]   \n"
            "fmla  v15.8h, v19.8h, v7.h[3]   \n"
            "ldp  q18, q19, [%[a]], #32      \n"

            "fmla  v8.8h,  v16.8h, v0.h[4]   \n"
            "fmla  v9.8h,  v16.8h, v1.h[4]   \n"
            "fmla  v10.8h, v16.8h, v2.h[4]   \n"
            "fmla  v11.8h, v16.8h, v3.h[4]   \n"
            "fmla  v12.8h, v16.8h, v4.h[4]   \n"
            "fmla  v13.8h, v16.8h, v5.h[4]   \n"
            "fmla  v14.8h, v16.8h, v6.h[4]   \n"
            "fmla  v15.8h, v16.8h, v7.h[4]   \n"
            "fmla  v8.8h,  v17.8h, v0.h[5]   \n"
            "fmla  v9.8h,  v17.8h, v1.h[5]   \n"
            "fmla  v10.8h, v17.8h, v2.h[5]   \n"
            "fmla  v11.8h, v17.8h, v3.h[5]   \n"
            "fmla  v12.8h, v17.8h, v4.h[5]   \n"
            "fmla  v13.8h, v17.8h, v5.h[5]   \n"
            "fmla  v14.8h, v17.8h, v6.h[5]   \n"
            "fmla  v15.8h, v17.8h, v7.h[5]   \n"

            "fmla  v8.8h,  v18.8h, v0.h[6]   \n"
            "fmla  v9.8h,  v18.8h, v1.h[6]   \n"
            "ldp  q16, q17, [%[a]], #32      \n"
            "fmla  v10.8h, v18.8h, v2.h[6]   \n"
            "fmla  v11.8h, v18.8h, v3.h[6]   \n"
            "fmla  v12.8h, v18.8h, v4.h[6]   \n"
            "fmla  v13.8h, v18.8h, v5.h[6]   \n"
            "fmla  v14.8h, v18.8h, v6.h[6]   \n"
            "fmla  v15.8h, v18.8h, v7.h[6]   \n"
            "fmla  v8.8h,  v19.8h, v0.h[7]   \n"
            "fmla  v9.8h,  v19.8h, v1.h[7]   \n"
            "ldp  q0, q1, [%[b]], #32        \n"
            "fmla  v10.8h, v19.8h, v2.h[7]   \n"
            "fmla  v11.8h, v19.8h, v3.h[7]   \n"
            "subs  %w[cnt], %w[cnt], #1      \n"
            "fmla  v12.8h, v19.8h, v4.h[7]   \n"
            "fmla  v13.8h, v19.8h, v5.h[7]   \n"
            "ldp  q2, q3, [%[b]], #32        \n"
            "fmla  v14.8h, v19.8h, v6.h[7]   \n"
            "fmla  v15.8h, v19.8h, v7.h[7]   \n"
            "ldp  q18, q19, [%[a]], #32      \n"           
            "bne   1b\n"
            "2:\n"
            "st1   {v8.8h,  v9.8h,  v10.8h, v11.8h}, [%[c]], #64  \n"
            "st1   {v12.8h, v13.8h, v14.8h, v15.8h}, [%[c]], #64  \n"
            : [a] "+r"(ablock_ptr),
              [b] "+r"(bblock),
              [c] "+r"(cblock),
              [cnt] "+r"(cnt)
            : 
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", 
              "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", 
              "v17", "v18", "v19", "v20", "cc", "memory");
#else
        asm volatile(
          "1:\n"
          "vmov.u16 q8, #0\n"
          "vmov.u16 q9, #0\n"
          "vmov.u16 q10, #0\n"
          "vmov.u16 q11, #0\n"
          "vmov.u16 q12, #0\n"
          "vmov.u16 q13, #0\n"
          "vmov.u16 q14, #0\n"
          "vmov.u16 q15, #0\n"
          "0:\n"
          "pld [%[a_ptr]]\n"
          "pld [%[b_ptr]]\n"
          "vld1.16 {d8-d9}, [%[a_ptr]]!\n"
          "vldr d0, [%[b_ptr], #0x00]\n"
          "vldr d2, [%[b_ptr], #0x10]\n"
          "vldr d4, [%[b_ptr], #0x20]\n"
          "vldr d6, [%[b_ptr], #0x30]\n"
          "vld1.16  {d10-d11}, [%[a_ptr]]!\n"
          "vldr d1, [%[b_ptr], #0x40]\n"
          "vldr d3, [%[b_ptr], #0x50]\n"
          "pld      [%[a_ptr]]\n"
          "vldr d5, [%[b_ptr], #0x60]\n"
          "vldr d7, [%[b_ptr], #0x70]\n"
          "vld1.16  {d12-d13}, [%[a_ptr]]!\n"
          "vmla.f16 q8,  q4, d0[0]\n"
          "vmla.f16 q9,  q4, d2[0]\n"
          "vmla.f16 q10, q4, d4[0]\n"
          "vmla.f16 q11, q4, d6[0]\n"
          "vld1.16  {d14-d15}, [%[a_ptr]]!\n"
          "vmla.f16 q12, q4, d1[0]\n"
          "vmla.f16 q13, q4, d3[0]\n"
          "vmla.f16 q14, q4, d5[0]\n"
          "vmla.f16 q15, q4, d7[0]\n"
          "vld1.16 {d8-d9}, [%[a_ptr]]!\n"
          "vmla.f16 q8,  q5, d0[1]\n"
          "vmla.f16 q9,  q5, d2[1]\n"
          "vmla.f16 q10, q5, d4[1]\n"
          "vmla.f16 q11, q5, d6[1]\n"
          "vmla.f16 q12, q5, d1[1]\n"
          "vmla.f16 q13, q5, d3[1]\n"
          "vmla.f16 q14, q5, d5[1]\n"
          "vmla.f16 q15, q5, d7[1]\n"
          "vld1.16 {d10-d11}, [%[a_ptr]]!\n"

          "vmla.f16 q8,  q6, d0[2]\n"
          "vmla.f16 q9,  q6, d2[2]\n"
          "vmla.f16 q10, q6, d4[2]\n"
          "vmla.f16 q11, q6, d6[2]\n"
          "vmla.f16 q12, q6, d1[2]\n"
          "vmla.f16 q13, q6, d3[2]\n"
          "vmla.f16 q14, q6, d5[2]\n"
          "vmla.f16 q15, q6, d7[2]\n"
          "vld1.16 {d12-d13}, [%[a_ptr]]!\n"

          "vmla.f16 q8,  q7, d0[3]\n"
          "vmla.f16 q9,  q7, d2[3]\n"
          "vmla.f16 q10, q7, d4[3]\n"
          "vmla.f16 q11, q7, d6[3]\n"
          "vldr d0, [%[b_ptr], #0x08]\n"
          "vmla.f16 q12, q7, d1[3]\n"
          "vldr d2, [%[b_ptr], #0x18]\n"
          "vmla.f16 q13, q7, d3[3]\n"
          "vldr d4, [%[b_ptr], #0x28]\n"
          "vmla.f16 q14, q7, d5[3]\n"
          "vldr d6, [%[b_ptr], #0x38]\n"
          "vmla.f16 q15, q7, d7[3]\n"
          "vldr d1, [%[b_ptr], #0x48]\n"
          "vld1.16 {d14-d15}, [%[a_ptr]]!\n"
          "vldr d3, [%[b_ptr], #0x58]\n"

          "vmla.f16 q8,  q4, d0[0]\n"
          "vldr d5, [%[b_ptr], #0x68]\n"
          "vmla.f16 q9,  q4, d2[0]\n"
          "vldr d7, [%[b_ptr], #0x78]\n"
          "vmla.f16 q10, q4, d4[0]\n"
          "vmla.f16 q11, q4, d6[0]\n"
          "vmla.f16 q12, q4, d1[0]\n"
          "vmla.f16 q13, q4, d3[0]\n"
          "vmla.f16 q14, q4, d5[0]\n"
          "vmla.f16 q15, q4, d7[0]\n"

          "vmla.f16 q8,  q5, d0[1]\n"
          "vmla.f16 q9,  q5, d2[1]\n"
          "vmla.f16 q10, q5, d4[1]\n"
          "vmla.f16 q11, q5, d6[1]\n"
          "vmla.f16 q12, q5, d1[1]\n"
          "vmla.f16 q13, q5, d3[1]\n"
          "vmla.f16 q14, q5, d5[1]\n"
          "vmla.f16 q15, q5, d7[1]\n"

          "vmla.f16 q8,  q6, d0[2]\n"
          "vmla.f16 q9,  q6, d2[2]\n"
          "vmla.f16 q10, q6, d4[2]\n"
          "vmla.f16 q11, q6, d6[2]\n"
          "vmla.f16 q12, q6, d1[2]\n"
          "vmla.f16 q13, q6, d3[2]\n"
          "vmla.f16 q14, q6, d5[2]\n"
          "vmla.f16 q15, q6, d7[2]\n"

          "vmla.f16 q8,  q7, d0[3]\n"
          "vmla.f16 q9,  q7, d2[3]\n"
          "vmla.f16 q10, q7, d4[3]\n"
          "vmla.f16 q11, q7, d6[3]\n"
          "sub %[cnt], #1\n"
          "add  %[b_ptr], #128\n"
          "vmla.f16 q12, q7, d1[3]\n"
          "vmla.f16 q13, q7, d3[3]\n"
          "vmla.f16 q14, q7, d5[3]\n"
          "vmla.f16 q15, q7, d7[3]\n"
          "bne  0b\n"
          "2:\n"
          "vst1.16 {d16-d19}, [%[c]]!\n"
          "vst1.16 {d20-d23}, [%[c]]!\n"
          "vst1.16 {d24-d27}, [%[c]]!\n"
          "vst1.16 {d28-d31}, [%[c]]!\n"
          : [a_ptr] "+r"(ablock_ptr),
            [b_ptr] "+r"(bblock),
            [c] "+r"(cblock),
            [cnt] "+r"(cnt)
          :
          : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
            "q10", "q11", "q12", "q13", "q14", "q15", "cc", "memory"
        );
#endif
        // clang-format on
      }
      if (has_remain) {
        if (w_loop4 > 0) {
          int cnt = kcnt;
          const float16_t* ablock_ptr = ablock;
// clang-format off
#ifdef __aarch64__
          asm volatile(
            "prfm  pldl1keep, [%[a]]         \n"
            "prfm  pldl1keep, [%[b]]         \n"
            /* load a0a1 to v0-v1  */
            "ldp  q16, q17, [%[a]], #32      \n"
            /* load b0b1 to v4-v5  */
            "ldp  q0, q1, [%[b]], #32        \n"
            "movi  v8.8h, #0                 \n"
            "movi  v9.8h, #0                 \n"
            "movi  v10.8h, #0                \n"
            "movi  v11.8h, #0                \n"
            /* load a0a1 to v0-v1  */
            "ldp  q18, q19, [%[a]], #32      \n"
            /* load b0b1 to v4-v5  */
            "ldp  q2, q3, [%[b]], #32        \n"
            "1:\n"
            "prfm  pldl1keep, [%[a]]         \n"
            "prfm  pldl1keep, [%[b]]         \n"
            "fmla  v8.8h,  v16.8h, v0.h[0]   \n"
            "fmla  v9.8h,  v16.8h, v1.h[0]   \n"
            "fmla  v10.8h, v16.8h, v2.h[0]   \n"
            "fmla  v11.8h, v16.8h, v3.h[0]   \n"

            "fmla  v8.8h,  v17.8h, v0.h[1]   \n"
            "fmla  v9.8h,  v17.8h, v1.h[1]   \n"
            "fmla  v10.8h, v17.8h, v2.h[1]   \n"
            "fmla  v11.8h, v17.8h, v3.h[1]   \n"
            "ldp  q16, q17, [%[a]], #32      \n"

            "fmla  v8.8h,  v18.8h, v0.h[2]   \n"
            "fmla  v9.8h,  v18.8h, v1.h[2]   \n"
            "fmla  v10.8h, v18.8h, v2.h[2]   \n"
            "fmla  v11.8h, v18.8h, v3.h[2]   \n"
            "fmla  v8.8h,  v19.8h, v0.h[3]   \n"
            "fmla  v9.8h,  v19.8h, v1.h[3]   \n"
            "fmla  v10.8h, v19.8h, v2.h[3]   \n"
            "fmla  v11.8h, v19.8h, v3.h[3]   \n"
            "ldp  q18, q19, [%[a]], #32      \n"

            "fmla  v8.8h,  v16.8h, v0.h[4]   \n"
            "fmla  v9.8h,  v16.8h, v1.h[4]   \n"
            "fmla  v10.8h, v16.8h, v2.h[4]   \n"
            "fmla  v11.8h, v16.8h, v3.h[4]   \n"
            "fmla  v8.8h,  v17.8h, v0.h[5]   \n"
            "fmla  v9.8h,  v17.8h, v1.h[5]   \n"
            "fmla  v10.8h, v17.8h, v2.h[5]   \n"
            "fmla  v11.8h, v17.8h, v3.h[5]   \n"

            "fmla  v8.8h,  v18.8h, v0.h[6]   \n"
            "fmla  v9.8h,  v18.8h, v1.h[6]   \n"
            "ldp  q16, q17, [%[a]], #32      \n"
            "fmla  v10.8h, v18.8h, v2.h[6]   \n"
            "fmla  v11.8h, v18.8h, v3.h[6]   \n"
            "subs  %w[cnt], %w[cnt], #1      \n"
            "fmla  v8.8h,  v19.8h, v0.h[7]   \n"
            "fmla  v9.8h,  v19.8h, v1.h[7]   \n"
            "ldp  q0, q1, [%[b]], #32        \n"
            "fmla  v10.8h, v19.8h, v2.h[7]   \n"
            "fmla  v11.8h, v19.8h, v3.h[7]   \n"
            "ldp  q2, q3, [%[b]], #32        \n"
            "ldp  q18, q19, [%[a]], #32      \n"           
            "bne   1b\n"
            "2:\n"
            "stp q8,  q9,  [%[c]], #32  \n"
            "stp q10, q11, [%[c]], #32  \n"
            : [a] "+r"(ablock_ptr),
              [b] "+r"(bblock),
              [c] "+r"(cblock),
              [cnt] "+r"(cnt)
            : 
            : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13",
              "v14", "v15", "v16", "v17", "v18", "v19", "v20", "cc", "memory");
#else
          asm volatile(
            "1: \n"
            "vmov.u16 q8, #0\n"
            "vmov.u16 q9, #0\n"
            "vmov.u16 q10, #0\n"
            "vmov.u16 q11, #0\n"
            "0:\n"
            "pld [%[a_ptr]]\n"
            "pld [%[b_ptr]]\n"
            "vld1.16 {d8-d9}, [%[a_ptr]]!\n"
            "vldr d0, [%[b_ptr], #0x00]\n"
            "vldr d2, [%[b_ptr], #0x10]\n"
            "vldr d4, [%[b_ptr], #0x20]\n"
            "vldr d6, [%[b_ptr], #0x30]\n"
            "vld1.16 {d10-d11}, [%[a_ptr]]!\n"
            "vldr d1, [%[b_ptr], #0x08]\n"
            "vldr d3, [%[b_ptr], #0x18]\n"
            "pld [%[a_ptr]]\n"
            "vldr d5, [%[b_ptr], #0x28]\n"
            "vldr d7, [%[b_ptr], #0x38]\n"
            "vld1.16 {d12-d13}, [%[a_ptr]]!\n"
            "vmla.f16 q8,  q4, d0[0]\n"
            "vmla.f16 q9,  q4, d2[0]\n"
            "vmla.f16 q10, q4, d4[0]\n"
            "vmla.f16 q11, q4, d6[0]\n"
            "vld1.16 {d14-d15}, [%[a_ptr]]!\n"

            "vmla.f16 q8,  q5, d0[1]\n"
            "vmla.f16 q9,  q5, d2[1]\n"
            "vmla.f16 q10, q5, d4[1]\n"
            "vmla.f16 q11, q5, d6[1]\n"
            "vld1.16 {d8-d9}, [%[a_ptr]]!\n"

            "vmla.f16 q8,  q6, d0[2]\n"
            "vmla.f16 q9,  q6, d2[2]\n"
            "vmla.f16 q10, q6, d4[2]\n"
            "vmla.f16 q11, q6, d6[2]\n"
            "vld1.16 {d10-d11}, [%[a_ptr]]!\n"

            "vmla.f16 q8,  q7, d0[3]\n"
            "vmla.f16 q9,  q7, d2[3]\n"
            "vmla.f16 q10, q7, d4[3]\n"
            "vmla.f16 q11, q7, d6[3]\n"
            "vld1.16 {d12-d13}, [%[a_ptr]]!\n"

            "vmla.f16 q8,  q4, d1[0]\n"
            "vmla.f16 q9,  q4, d3[0]\n"
            "vmla.f16 q10, q4, d5[0]\n"
            "vmla.f16 q11, q4, d7[0]\n"

            "vmla.f16 q8,  q5, d1[1]\n"
            "vmla.f16 q9,  q5, d3[1]\n"
            "vmla.f16 q10, q5, d5[1]\n"
            "vmla.f16 q11, q5, d7[1]\n"

            "vmla.f16 q8,  q6, d1[2]\n"
            "vmla.f16 q9,  q6, d3[2]\n"
            "vmla.f16 q10, q6, d5[2]\n"
            "vmla.f16 q11, q6, d7[2]\n"
            "sub %[cnt], #1\n"
            "add  %[b_ptr], #64\n"

            "vmla.f16 q8,  q7, d1[3]\n"
            "vmla.f16 q9,  q7, d3[3]\n"
            "vmla.f16 q10, q7, d5[3]\n"
            "vmla.f16 q11, q7, d7[3]\n"
            "bne  0b\n"
            "2:\n"
            "vst1.16 {d16-d19}, [%[c]]!\n"
            "vst1.16 {d20-d23}, [%[c]]!\n"
            : [a_ptr] "+r"(ablock_ptr),
              [b_ptr] "+r"(bblock),
              [c] "+r"(cblock),
              [cnt] "+r"(cnt)
            : 
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
              "q10", "q11", "q12", "q13", "q14", "q15", "cc", "memory"
          );
#endif
          // clang-format on
        }
        if (remain > 0) {
          int cnt = kcnt;
          const float16_t* ablock_ptr = ablock;
// clang-format off
#ifdef __aarch64__
          asm volatile(
            "cmp   %w[remain], #3            \n"
            "prfm  pldl1keep, [%[a]]         \n"
            "prfm  pldl1keep, [%[b]]         \n"
            /* load b0b1 to v0  */
            "ldr   q0, [%[b]], #16           \n"
            /* load a0a1 to v16-v17  */
            "ldp   q16, q17, [%[a]], #32     \n"
            "movi  v8.8h, #0                 \n"
            "movi  v9.8h, #0                 \n"
            "movi  v10.8h, #0                \n"
            "movi  v11.8h, #0                \n"
            "beq   3f                        \n"
            "cmp   %w[remain], #2            \n"
            "ldr   q1, [%[b]], #16           \n"
            "prfm  pldl1keep, [%[b]]         \n"
            "movi  v12.8h, #0                 \n"
            "movi  v13.8h, #0                 \n"
            "movi  v14.8h, #0                \n"
            "movi  v15.8h, #0                \n"
            "beq   2f                        \n"
            /* remain = 1*/
            "0: \n"
            "ldp   q18, q19, [%[a]], #32     \n"
            "fmla  v8.8h,  v16.8h, v0.h[0]   \n"
            "fmla  v9.8h,  v17.8h, v0.h[1]   \n"
            "ldp   q16, q17, [%[a]], #32     \n"
            "fmla  v10.8h, v18.8h, v0.h[2]   \n"
            "fmla  v11.8h, v19.8h, v0.h[3]   \n"
            "ldp   q18, q19, [%[a]], #32     \n"
            "fmla  v8.8h,  v16.8h, v0.h[4]   \n"
            "fmla  v9.8h,  v17.8h, v0.h[5]   \n"
            "prfm  pldl1keep, [%[a]]         \n"
            "prfm  pldl1keep, [%[b]]         \n"
            "subs  %w[cnt], %w[cnt], #1      \n"
            "fmla  v10.8h, v18.8h, v0.h[6]   \n"
            "fmla  v11.8h, v19.8h, v0.h[7]   \n"
            "ldr   q0, [%[b]], #16           \n"
            "ldp   q16, q17, [%[a]], #32     \n"
            "bne 0b\n"
            "fadd  v18.8h, v8.8h,  v9.8h     \n"
            "fadd  v19.8h, v10.8h, v11.8h    \n"
            "fadd  v8.8h,  v18.8h, v19.8h    \n"
            "str   q8,     [%[c]], #16       \n"
            "b 1f\n"
            /* remain = 2*/
            "2: \n"
            "ldp   q18, q19, [%[a]], #32     \n"
            "fmla  v8.8h,  v16.8h, v0.h[0]   \n"
            "fmla  v9.8h,  v16.8h, v1.h[0]   \n"
            "fmla  v10.8h, v17.8h, v0.h[1]   \n"
            "fmla  v11.8h, v17.8h, v1.h[1]   \n"
            "ldp   q16, q17, [%[a]], #32     \n"
            "fmla  v8.8h,  v18.8h, v0.h[2]   \n"
            "fmla  v9.8h,  v18.8h, v1.h[2]   \n"
            "fmla  v10.8h, v19.8h, v0.h[3]   \n"
            "fmla  v11.8h, v19.8h, v1.h[3]   \n"
            "ldp   q18, q19, [%[a]], #32     \n"
            "fmla  v8.8h,  v16.8h, v0.h[4]   \n"
            "fmla  v9.8h,  v16.8h, v1.h[4]   \n"
            "fmla  v10.8h, v17.8h, v0.h[5]   \n"
            "fmla  v11.8h, v17.8h, v1.h[5]   \n"
            "prfm  pldl1keep, [%[a]]         \n"
            "prfm  pldl1keep, [%[b]]         \n"
            "fmla  v8.8h,  v18.8h, v0.h[6]   \n"
            "fmla  v9.8h,  v18.8h, v1.h[6]   \n"
            "subs  %w[cnt], %w[cnt], #1      \n"
            "fmla  v10.8h, v19.8h, v0.h[7]   \n"
            "fmla  v11.8h, v19.8h, v1.h[7]   \n"
            "ldp   q16, q17, [%[a]], #32     \n"
            "ldp   q0,  q1,  [%[b]], #32     \n"
            "bne 2b\n"
            "fadd  v18.8h, v8.8h,  v10.8h     \n"
            "fadd  v19.8h, v9.8h, v11.8h      \n"
            "stp   q18, q19, [%[c]], #32      \n"
            "b 1f\n"
            /* remain = 3*/
            "3: \n"
            "ldr q2, [%[b]], #16             \n"
            "ldp   q18, q19, [%[a]], #32     \n"
            "fmla  v8.8h,  v16.8h, v0.h[0]   \n"
            "fmla  v9.8h,  v16.8h, v1.h[0]   \n"
            "fmla  v10.8h, v16.8h, v2.h[0]   \n"
            "fmla  v11.8h, v17.8h, v0.h[1]   \n"
            "fmla  v12.8h, v17.8h, v1.h[1]   \n"
            "fmla  v13.8h, v17.8h, v2.h[1]   \n"
            "ldp   q16, q17, [%[a]], #32     \n"
            "fmla  v8.8h,  v18.8h, v0.h[2]   \n"
            "fmla  v9.8h,  v18.8h, v1.h[2]   \n"
            "fmla  v10.8h, v18.8h, v2.h[2]   \n"
            "fmla  v11.8h, v19.8h, v0.h[3]   \n"
            "fmla  v12.8h, v19.8h, v1.h[3]   \n"
            "fmla  v13.8h, v19.8h, v2.h[3]   \n"
            "ldp   q18, q19, [%[a]], #32     \n"
            "fmla  v8.8h,  v16.8h, v0.h[4]   \n"
            "fmla  v9.8h,  v16.8h, v1.h[4]   \n"
            "fmla  v10.8h, v16.8h, v2.h[4]   \n"
            "fmla  v11.8h, v17.8h, v0.h[5]   \n"
            "fmla  v12.8h, v17.8h, v1.h[5]   \n"
            "fmla  v13.8h, v17.8h, v2.h[5]   \n"
            "subs  %w[cnt], %w[cnt], #1      \n"
            "fmla  v8.8h,  v18.8h, v0.h[6]   \n"
            "fmla  v9.8h,  v18.8h, v1.h[6]   \n"
            "fmla  v10.8h, v18.8h, v2.h[6]   \n"
            "prfm  pldl1keep, [%[a]]         \n"
            "fmla  v11.8h, v19.8h, v0.h[7]   \n"
            "prfm  pldl1keep, [%[b]]         \n"
            "fmla  v12.8h, v19.8h, v1.h[7]   \n"
            "fmla  v13.8h, v19.8h, v2.h[7]   \n"
            "ldp   q16, q17, [%[a]], #32     \n"
            "ldp   q0,  q1,  [%[b]], #32     \n"
            "bne 3b\n"
            "fadd  v18.8h, v8.8h,  v11.8h     \n"
            "fadd  v19.8h, v9.8h,  v12.8h     \n"
            "fadd  v20.8h, v10.8h, v13.8h     \n"
            "st1   {v18.8h, v19.8h, v20.8h}, [%[c]], #48\n"
            "1: \n"
            : [a] "+r"(ablock_ptr),
              [b] "+r"(bblock),
              [c] "+r"(cblock),
              [cnt] "+r"(cnt)
            : [remain] "r"(remain)
            : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13",
              "v14", "v15", "v16", "v17", "v18", "v19", "v20", "cc", "memory");
#else
          asm volatile(
              "cmp  %[remain], #1\n"
              "vldr d0,  [%[b_ptr], #0x00]\n"
              "vld1.16 {d8-d11}, [%[a_ptr]]!\n"
              "vmov.u16 q8, #0\n"
              "vmov.u16 q9, #0\n"
              "vmov.u16 q10, #0\n"
              "vmov.u16 q11, #0\n"
              "beq  1f\n"
              "cmp  %[remain], #2\n"
              "vldr d2,  [%[b_ptr], #0x10]\n"
              "vld1.16 {d12-d15}, [%[a_ptr]]!\n"
              "vmov.u16 q12, #0\n"
              "vmov.u16 q13, #0\n"
              "vmov.u16 q14, #0\n"
              "vmov.u16 q15, #0\n"
              "beq  2f\n"
              // remain = 3
              "3:\n"
              "vldr d4, [%[b_ptr], #0x10]\n"
              "pld [%[a_ptr]]\n"
              "pld [%[b_ptr]]\n"
              "vldr d1, [%[b_ptr], #0x08]\n"
              "vldr d3, [%[b_ptr], #0x18]\n"
              "vldr d7, [%[b_ptr], #0x28]\n"
              "vmla.f16 q8,  q4, d0[0]\n"
              "vmla.f16 q9,  q5, d0[1]\n"
              "vmla.f16 q10, q4, d2[0]\n"
              "vmla.f16 q11, q5, d2[1]\n"
              "vmla.f16 q12, q4, d4[0]\n"
              "vmla.f16 q13, q5, d4[1]\n"
              "vld1.16 {d8-d11}, [%[a_ptr]]!\n"

              "vmla.f16 q8,  q6, d0[2]\n"
              "vmla.f16 q9,  q7, d0[3]\n"
              "vmla.f16 q10, q6, d2[2]\n"
              "vmla.f16 q11, q7, d2[3]\n"
              "vmla.f16 q12, q6, d4[2]\n"
              "vmla.f16 q13, q7, d4[2]\n"
              "add  %[b_ptr], #48\n"
              "vld1.16 {d12-d15}, [%[a_ptr]]!\n"

              "vmla.f16 q8,  q4, d1[0]\n"
              "vmla.f16 q9,  q5, d1[1]\n"
              "vmla.f16 q10, q4, d3[0]\n"
              "vmla.f16 q11, q5, d3[1]\n"
              "vmla.f16 q12, q4, d5[0]\n"
              "vmla.f16 q13, q5, d5[1]\n"
              "vldr d0,  [%[b_ptr], #0x00]\n"
              "vld1.16 {d8-d11}, [%[a_ptr]]!\n"
              "sub %[cnt], #1\n"
              "vldr d2,  [%[b_ptr], #0x10]\n"
              "vmla.f16 q8,  q6, d1[2]\n"
              "vmla.f16 q9,  q7, d1[3]\n"
              "vmla.f16 q10, q6, d3[2]\n"
              "vmla.f16 q11, q7, d3[3]\n"
              "vmla.f16 q12, q6, d5[2]\n"
              "vmla.f16 q13, q7, d5[3]\n"
              "vld1.16 {d12-d15}, [%[a_ptr]]!\n"
              "bne  1b\n"
              "vadd.f16 q0, q8, q9\n"
              "vadd.f16 q1, q10, q11\n"
              "vadd.f16 q2, q12, q13\n"
              "vst1.16 {d0-d3}, [%[c]]!\n"
              "vst1.16 {d4-d5}, [%[c]]!\n"
              "b 0f\n"
              // remain = 2
              "2:\n"
              "pld [%[a_ptr]]\n"
              "pld [%[b_ptr]]\n"
              "vldr d1, [%[b_ptr], #0x08]\n"
              "vldr d3, [%[b_ptr], #0x18]\n"
              "vmla.f16 q8,  q4, d0[0]\n"
              "vmla.f16 q9,  q5, d0[1]\n"
              "vmla.f16 q10, q4, d2[0]\n"
              "vmla.f16 q11, q5, d2[1]\n"
              "vld1.16 {d8-d11}, [%[a_ptr]]!\n"

              "vmla.f16 q8,  q6, d0[2]\n"
              "vmla.f16 q9,  q7, d0[3]\n"
              "vmla.f16 q10, q6, d2[2]\n"
              "vmla.f16 q11, q7, d2[3]\n"
              "add  %[b_ptr], #32\n"
              "vld1.16 {d12-d15}, [%[a_ptr]]!\n"

              "vmla.f16 q8,  q4, d1[0]\n"
              "vmla.f16 q9,  q5, d1[1]\n"
              "vmla.f16 q10, q4, d3[0]\n"
              "vmla.f16 q11, q5, d3[1]\n"
              "vldr d0,  [%[b_ptr], #0x00]\n"
              "vld1.16 {d8-d11}, [%[a_ptr]]!\n"
              "sub %[cnt], #1\n"
              "vldr d2,  [%[b_ptr], #0x10]\n"
              "vmla.f16 q8,  q6, d1[2]\n"
              "vmla.f16 q9,  q7, d1[3]\n"
              "vmla.f16 q10, q6, d3[2]\n"
              "vmla.f16 q11, q7, d3[3]\n"
              "vld1.16 {d12-d15}, [%[a_ptr]]!\n"
              "bne  1b\n"
              "vadd.f16 q0, q8, q9\n"
              "vadd.f16 q1, q10, q11\n"
              "vst1.16 {d0-d3}, [%[c]]!\n"
              "b 0f\n"
              // remain = 1
              "1:\n"
              "pld [%[a_ptr]]\n"
              "pld [%[b_ptr]]\n"
              "vldr d1, [%[b_ptr], #0x08]\n"
              "vld1.16 {d12-d15}, [%[a_ptr]]!\n"
              "vmla.f16 q8, q4, d0[0]\n"
              "vmla.f16 q9, q5, d0[1]\n"
              "vld1.16 {d8-d11}, [%[a_ptr]]!\n"
              "vmla.f16 q10, q6, d0[2]\n"
              "vmla.f16 q11, q7, d0[3]\n"
              "add  %[b_ptr], #16\n"
              "vld1.16 {d12-d15}, [%[a_ptr]]!\n"

              "sub %[cnt], #1\n"
              "vmla.f16 q8, q4, d1[0]\n"
              "vmla.f16 q9, q5, d1[1]\n"
              "vldr d4,  [%[b_ptr], #0x00]\n"
              "vmla.f16 q10, q6, d1[2]\n"
              "vmla.f16 q11, q7, d1[3]\n"
              "vld1.16 {d8-d11}, [%[a_ptr]]!\n"
              "bne  1b\n"
              "vadd.f16 q0, q8, q9\n"
              "vadd.f16 q1, q10, q11\n"
              "vadd.f16 q2, q0, q1\n"
              "vst1.16 {d4-d5}, [%[c]]!\n"
              "0: \n"
              : [a_ptr] "+r"(ablock_ptr),
                [b_ptr] "+r"(bblock),
                [c] "+r"(cblock),
                [cnt] "+r"(cnt)
              : [remain] "r"(remain)
              : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                "q10", "q11", "q12", "q13", "q14", "q15", "cc", "memory"
            );
#endif
          // clang-format on
        }
      }
    }
    LITE_PARALLEL_END();
  }
}

void gemm_prepack_c8_fp16_small(int M,
                                int N,
                                int K,
                                const float16_t* A_packed,
                                const float16_t* B,
                                float16_t* C,
                                ARMContext* ctx) {
  const int m_round = (M + 7) / 8 * 8;
  const int k_round = (K + 7) / 8 * 8;
  const int mloop = m_round >> 3;
  const int lda = k_round << 3;
  const int ldb_byte = 8 * N * sizeof(float16_t);
  const int kcnt = k_round >> 3;
  auto tmp = C;

  for (int m = 0; m < mloop; ++m) {
    const float16_t* b = B;
    int n = N;
#ifdef __aarch64__
    for (; n > 7; n -= 8) {
      int cnt = kcnt;
      const float16_t* a_ptr = A_packed;
      const float16_t* b_ptr = b;
      // clang-format off
      asm volatile(
        "0:\n"
        /* load a0, a1 */
        "ldp  q16, q17, [%[a]], #32\n"
        /* load b0, b1 */
        "ldp  q0,  q1,  [%[b]], #32\n"
        /* load b2, b3 */
        "ldp  q2,  q3,  [%[b]], #32\n"
        "prfm pldl1keep, [%[b]]    \n"
        /* load a2, a3 */
        "ldp  q18, q19, [%[a]], #32\n"
        /* load b4, b5 */
        "ldp  q4,  q5,  [%[b]], #32\n"
        "fmul v8.8h,  v16.8h, v0.h[0] \n"
        "fmul v9.8h,  v16.8h, v1.h[0] \n"
        /* load b6, b7 */
        "ldp  q6,  q7,  [%[b]], #32\n"
        "fmul v10.8h, v16.8h, v2.h[0] \n"
        "fmul v11.8h, v16.8h, v3.h[0] \n"
        "fmul v12.8h, v16.8h, v4.h[0] \n"
        "fmul v13.8h, v16.8h, v5.h[0] \n"
        "fmul v14.8h, v16.8h, v6.h[0] \n"
        "fmul v15.8h, v16.8h, v7.h[0] \n"
        "sub  %[b],   %[b],   #128    \n"
        "fmla v8.8h,  v17.8h, v0.h[1] \n"
        "fmla v9.8h,  v17.8h, v1.h[1] \n"
        "fmla v10.8h, v17.8h, v2.h[1] \n"
        "fmla v11.8h, v17.8h, v3.h[1] \n"
        "fmla v12.8h, v17.8h, v4.h[1] \n"
        "fmla v13.8h, v17.8h, v5.h[1] \n"
        "fmla v14.8h, v17.8h, v6.h[1] \n"
        "fmla v15.8h, v17.8h, v7.h[1] \n"
        "add  %[b],   %[b],   %[ldb]  \n"
        "fmla v8.8h,  v18.8h, v0.h[2] \n"
        "fmla v9.8h,  v18.8h, v1.h[2] \n"
        "fmla v10.8h, v18.8h, v2.h[2] \n"
        "fmla v11.8h, v18.8h, v3.h[2] \n"
        "fmla v12.8h, v18.8h, v4.h[2] \n"
        "fmla v13.8h, v18.8h, v5.h[2] \n"
        "fmla v14.8h, v18.8h, v6.h[2] \n"
        "fmla v15.8h, v18.8h, v7.h[2] \n"
        "ldp  q16, q17, [%[a]], #32\n"
        "fmla v8.8h,  v19.8h, v0.h[3] \n"
        "fmla v9.8h,  v19.8h, v1.h[3] \n"
        "fmla v10.8h, v19.8h, v2.h[3] \n"
        "fmla v11.8h, v19.8h, v3.h[3] \n"
        "fmla v12.8h, v19.8h, v4.h[3] \n"
        "fmla v13.8h, v19.8h, v5.h[3] \n"
        "fmla v14.8h, v19.8h, v6.h[3] \n"
        "fmla v15.8h, v19.8h, v7.h[3] \n"
        "ldp  q18, q19, [%[a]], #32\n"
        "fmla v8.8h,  v16.8h, v0.h[4] \n"
        "fmla v9.8h,  v16.8h, v1.h[4] \n"
        "fmla v10.8h, v16.8h, v2.h[4] \n"
        "fmla v11.8h, v16.8h, v3.h[4] \n"
        "fmla v12.8h, v16.8h, v4.h[4] \n"
        "fmla v13.8h, v16.8h, v5.h[4] \n"
        "fmla v14.8h, v16.8h, v6.h[4] \n"
        "fmla v15.8h, v16.8h, v7.h[4] \n"
        "fmla v8.8h,  v17.8h, v0.h[5] \n"
        "fmla v9.8h,  v17.8h, v1.h[5] \n"
        "fmla v10.8h, v17.8h, v2.h[5] \n"
        "fmla v11.8h, v17.8h, v3.h[5] \n"
        "fmla v12.8h, v17.8h, v4.h[5] \n"
        "fmla v13.8h, v17.8h, v5.h[5] \n"
        "fmla v14.8h, v17.8h, v6.h[5] \n"
        "fmla v15.8h, v17.8h, v7.h[5] \n"
        "fmla v8.8h,  v18.8h, v0.h[6] \n"
        "fmla v9.8h,  v18.8h, v1.h[6] \n"
        /* load a0, a1 */
        "ldp  q16, q17, [%[a]], #32\n"
        "fmla v10.8h, v18.8h, v2.h[6] \n"
        "fmla v11.8h, v18.8h, v3.h[6] \n"
        "prfm pldl1keep, [%[b]]    \n"
        "fmla v12.8h, v18.8h, v4.h[6] \n"
        "fmla v13.8h, v18.8h, v5.h[6] \n"
        "fmla v14.8h, v18.8h, v6.h[6] \n"
        "fmla v15.8h, v18.8h, v7.h[6] \n"
        "fmla v8.8h,  v19.8h, v0.h[7] \n"
        "fmla v9.8h,  v19.8h, v1.h[7] \n"
        "fmla v10.8h, v19.8h, v2.h[7] \n"
        "fmla v11.8h, v19.8h, v3.h[7] \n"
        /* load b0, b1 */
        "ldp  q0,  q1,  [%[b]], #32\n"
        "subs %w[cnt], %w[cnt], #1    \n"
        "fmla v12.8h, v19.8h, v4.h[7] \n"
        "fmla v13.8h, v19.8h, v5.h[7] \n"
        /* load b2, b3 */
        "ldp  q2,  q3,  [%[b]], #32\n"
        "fmla v14.8h, v19.8h, v6.h[7] \n"
        "fmla v15.8h, v19.8h, v7.h[7] \n"
        "beq  2f                      \n"
        "1:\n"
        /* load b4, b5 */
        "ldp  q4,  q5,  [%[b]], #32\n"
        /* load a2, a3 */
        "ldp  q18, q19, [%[a]], #32\n"
        "fmla v8.8h,  v16.8h, v0.h[0] \n"
        "fmla v9.8h,  v16.8h, v1.h[0] \n"
        /* load b6, b7 */
        "ldp  q6,  q7,  [%[b]], #32\n"
        "fmla v10.8h, v16.8h, v2.h[0] \n"
        "fmla v11.8h, v16.8h, v3.h[0] \n"
        "fmla v12.8h, v16.8h, v4.h[0] \n"
        "fmla v13.8h, v16.8h, v5.h[0] \n"
        "fmla v14.8h, v16.8h, v6.h[0] \n"
        "fmla v15.8h, v16.8h, v7.h[0] \n"

        "sub  %[b],   %[b],   #128    \n"
        "fmla v8.8h,  v17.8h, v0.h[1] \n"
        "fmla v9.8h,  v17.8h, v1.h[1] \n"
        "fmla v10.8h, v17.8h, v2.h[1] \n"
        "fmla v11.8h, v17.8h, v3.h[1] \n"
        "add  %[b],   %[b],   %[ldb]  \n"
        "fmla v12.8h, v17.8h, v4.h[1] \n"
        "fmla v13.8h, v17.8h, v5.h[1] \n"
        "fmla v14.8h, v17.8h, v6.h[1] \n"
        "fmla v15.8h, v17.8h, v7.h[1] \n"
        
        "ldp  q16, q17, [%[a]], #32\n"
        "fmla v8.8h,  v18.8h, v0.h[2] \n"
        "fmla v9.8h,  v18.8h, v1.h[2] \n"
        "prfm pldl1keep, [%[b]]       \n"
        "fmla v10.8h, v18.8h, v2.h[2] \n"
        "fmla v11.8h, v18.8h, v3.h[2] \n"
        "fmla v12.8h, v18.8h, v4.h[2] \n"
        "fmla v13.8h, v18.8h, v5.h[2] \n"
        "fmla v14.8h, v18.8h, v6.h[2] \n"
        "fmla v15.8h, v18.8h, v7.h[2] \n"
        "fmla v8.8h,  v19.8h, v0.h[3] \n"
        "fmla v9.8h,  v19.8h, v1.h[3] \n"
        "fmla v10.8h, v19.8h, v2.h[3] \n"
        "fmla v11.8h, v19.8h, v3.h[3] \n"
        "fmla v12.8h, v19.8h, v4.h[3] \n"
        "fmla v13.8h, v19.8h, v5.h[3] \n"
        "fmla v14.8h, v19.8h, v6.h[3] \n"
        "fmla v15.8h, v19.8h, v7.h[3] \n"

        "ldp  q18, q19, [%[a]], #32\n"
        "fmla v8.8h,  v16.8h, v0.h[4] \n"
        "fmla v9.8h,  v16.8h, v1.h[4] \n"
        "fmla v10.8h, v16.8h, v2.h[4] \n"
        "fmla v11.8h, v16.8h, v3.h[4] \n"
        "fmla v12.8h, v16.8h, v4.h[4] \n"
        "fmla v13.8h, v16.8h, v5.h[4] \n"
        "fmla v14.8h, v16.8h, v6.h[4] \n"
        "fmla v15.8h, v16.8h, v7.h[4] \n"
        "fmla v8.8h,  v17.8h, v0.h[5] \n"
        "fmla v9.8h,  v17.8h, v1.h[5] \n"
        "fmla v10.8h, v17.8h, v2.h[5] \n"
        "fmla v11.8h, v17.8h, v3.h[5] \n"
        "fmla v12.8h, v17.8h, v4.h[5] \n"
        "fmla v13.8h, v17.8h, v5.h[5] \n"
        "fmla v14.8h, v17.8h, v6.h[5] \n"
        "fmla v15.8h, v17.8h, v7.h[5] \n"
        /* load a0, a1 */
        "ldp  q16, q17, [%[a]], #32\n"
        "fmla v8.8h,  v18.8h, v0.h[6] \n"
        "fmla v9.8h,  v18.8h, v1.h[6] \n"
        "fmla v10.8h, v18.8h, v2.h[6] \n"
        "fmla v11.8h, v18.8h, v3.h[6] \n"
        "fmla v12.8h, v18.8h, v4.h[6] \n"
        "fmla v13.8h, v18.8h, v5.h[6] \n"
        "fmla v14.8h, v18.8h, v6.h[6] \n"
        "fmla v15.8h, v18.8h, v7.h[6] \n"

        "fmla v8.8h,  v19.8h, v0.h[7] \n"
        "fmla v9.8h,  v19.8h, v1.h[7] \n"
        "fmla v10.8h, v19.8h, v2.h[7] \n"
        /* load b0, b1 */
        "ldp  q0,  q1,  [%[b]], #32\n"
        "subs %w[cnt], %w[cnt], #1    \n"
        "fmla v11.8h, v19.8h, v3.h[7] \n"
        "fmla v12.8h, v19.8h, v4.h[7] \n"
        /* load b2, b3 */
        "ldp  q2,  q3,  [%[b]], #32\n"
        "fmla v13.8h, v19.8h, v5.h[7] \n"
        "fmla v14.8h, v19.8h, v6.h[7] \n"
        "fmla v15.8h, v19.8h, v7.h[7] \n"
        "bne  1b                      \n"
        "2:\n"
        "st1  {v8.8h,  v9.8h,  v10.8h, v11.8h}, [%[c]], #64 \n"
        "st1  {v12.8h, v13.8h, v14.8h, v15.8h}, [%[c]], #64 \n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [ldb]  "r" (ldb_byte)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
          "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
          "v19", "cc", "memory"
      );
      // clang-format on
      b += 64;
    }
    if (n > 3) {
      int cnt = kcnt;
      const float16_t* a_ptr = A_packed;
      const float16_t* b_ptr = b;
      // clang-format off
      asm volatile(
        "0:\n"
        /* load a0, a1 */
        "ldp  q16, q17, [%[a]], #32\n"
        /* load b0, b1 */
        "ldp  q0,  q1,  [%[b]], #32\n"
        /* load b2, b3 */
        "ldp  q2,  q3,  [%[b]], #32\n"
        "prfm pldl1keep, [%[b]]    \n"
        /* load a2, a3 */
        "ldp  q18, q19, [%[a]], #32\n"
        "fmul v8.8h,  v16.8h, v0.h[0] \n"
        "fmul v9.8h,  v16.8h, v1.h[0] \n"
        "fmul v10.8h, v16.8h, v2.h[0] \n"
        "fmul v11.8h, v16.8h, v3.h[0] \n"
        "sub  %[b],   %[b],   #64     \n"
        "fmla v8.8h,  v17.8h, v0.h[1] \n"
        "fmla v9.8h,  v17.8h, v1.h[1] \n"
        "fmla v10.8h, v17.8h, v2.h[1] \n"
        "fmla v11.8h, v17.8h, v3.h[1] \n"
        "add  %[b],   %[b],   %[ldb]  \n"
        "fmla v8.8h,  v18.8h, v0.h[2] \n"
        "fmla v9.8h,  v18.8h, v1.h[2] \n"
        "fmla v10.8h, v18.8h, v2.h[2] \n"
        "fmla v11.8h, v18.8h, v3.h[2] \n"
        "ldp  q16, q17, [%[a]], #32\n"
        "fmla v8.8h,  v19.8h, v0.h[3] \n"
        "fmla v9.8h,  v19.8h, v1.h[3] \n"
        "fmla v10.8h, v19.8h, v2.h[3] \n"
        "fmla v11.8h, v19.8h, v3.h[3] \n"
        "ldp  q18, q19, [%[a]], #32\n"
        "fmla v8.8h,  v16.8h, v0.h[4] \n"
        "fmla v9.8h,  v16.8h, v1.h[4] \n"
        "fmla v10.8h, v16.8h, v2.h[4] \n"
        "fmla v11.8h, v16.8h, v3.h[4] \n"

        "fmla v8.8h,  v17.8h, v0.h[5] \n"
        "fmla v9.8h,  v17.8h, v1.h[5] \n"
        "fmla v10.8h, v17.8h, v2.h[5] \n"
        "fmla v11.8h, v17.8h, v3.h[5] \n"
        "fmla v8.8h,  v18.8h, v0.h[6] \n"
        "fmla v9.8h,  v18.8h, v1.h[6] \n"
        /* load a0, a1 */
        "ldp  q16, q17, [%[a]], #32\n"
        "fmla v10.8h, v18.8h, v2.h[6] \n"
        "fmla v11.8h, v18.8h, v3.h[6] \n"
        "prfm pldl1keep, [%[b]]    \n"
        "subs %w[cnt], %w[cnt], #1    \n"
        "fmla v8.8h,  v19.8h, v0.h[7] \n"
        "fmla v9.8h,  v19.8h, v1.h[7] \n"
        /* load b0, b1 */
        "ldp  q0,  q1,  [%[b]], #32\n"
        "fmla v10.8h, v19.8h, v2.h[7] \n"
        "fmla v11.8h, v19.8h, v3.h[7] \n"
        /* load b2, b3 */
        "ldp  q2,  q3,  [%[b]], #32\n"
        "beq  2f                      \n"
        "1:\n"
        /* load a2, a3 */
        "ldp  q18, q19, [%[a]], #32\n"
        "fmla v8.8h,  v16.8h, v0.h[0] \n"
        "fmla v9.8h,  v16.8h, v1.h[0] \n"
        "fmla v10.8h, v16.8h, v2.h[0] \n"
        "fmla v11.8h, v16.8h, v3.h[0] \n"

        "sub  %[b],   %[b],   #64     \n"
        "fmla v8.8h,  v17.8h, v0.h[1] \n"
        "fmla v9.8h,  v17.8h, v1.h[1] \n"
        "fmla v10.8h, v17.8h, v2.h[1] \n"
        "fmla v11.8h, v17.8h, v3.h[1] \n"
        "add  %[b],   %[b],   %[ldb]  \n"
        
        "ldp  q16, q17, [%[a]], #32\n"
        "fmla v8.8h,  v18.8h, v0.h[2] \n"
        "fmla v9.8h,  v18.8h, v1.h[2] \n"
        "prfm pldl1keep, [%[b]]       \n"
        "fmla v10.8h, v18.8h, v2.h[2] \n"
        "fmla v11.8h, v18.8h, v3.h[2] \n"

        "fmla v8.8h,  v19.8h, v0.h[3] \n"
        "fmla v9.8h,  v19.8h, v1.h[3] \n"
        "fmla v10.8h, v19.8h, v2.h[3] \n"
        "fmla v11.8h, v19.8h, v3.h[3] \n"

        "ldp  q18, q19, [%[a]], #32\n"
        "fmla v8.8h,  v16.8h, v0.h[4] \n"
        "fmla v9.8h,  v16.8h, v1.h[4] \n"
        "fmla v10.8h, v16.8h, v2.h[4] \n"
        "fmla v11.8h, v16.8h, v3.h[4] \n"

        "fmla v8.8h,  v17.8h, v0.h[5] \n"
        "fmla v9.8h,  v17.8h, v1.h[5] \n"
        "fmla v10.8h, v17.8h, v2.h[5] \n"
        "fmla v11.8h, v17.8h, v3.h[5] \n"
        /* load a0, a1 */
        "ldp  q16, q17, [%[a]], #32\n"
        "fmla v8.8h,  v18.8h, v0.h[6] \n"
        "fmla v9.8h,  v18.8h, v1.h[6] \n"
        "fmla v10.8h, v18.8h, v2.h[6] \n"
        "fmla v11.8h, v18.8h, v3.h[6] \n"

        "subs %w[cnt], %w[cnt], #1    \n"
        "fmla v8.8h,  v19.8h, v0.h[7] \n"
        "fmla v9.8h,  v19.8h, v1.h[7] \n"
        /* load b0, b1 */
        "ldp  q0,  q1,  [%[b]], #32\n"
        "fmla v10.8h, v19.8h, v2.h[7] \n"
        "fmla v11.8h, v19.8h, v3.h[7] \n"
        /* load b2, b3 */
        "ldp  q2,  q3,  [%[b]], #32\n"
        "bne  1b                      \n"
        "2:\n"
        "stp q8, q9, [%[c]], #32      \n"
        "stp q10, q11, [%[c]], #32      \n"
        : [a] "+r" (a_ptr),
          [b] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [ldb]  "r" (ldb_byte)
        : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13",
          "v14", "v15", "v16", "v17", "v18", "v19", "cc", "memory"
      );
      // clang-format on
      b += 32;
      n -= 4;
    }
    for (; n > 0; n--) {
      int cnt = kcnt;
      const float16_t* a_ptr = A_packed;
      const float16_t* b_ptr = b;
      asm volatile(
          "0:\n"
          /* load a0, a1 */
          "ldp  q16, q17, [%[a]], #32  \n"
          /* load b0 */
          "ldr  q0,  [%[b]], #16       \n"
          /* load a2, a3 */
          "ldp  q18, q19, [%[a]], #32   \n"
          "sub  %[b],   %[b],   #16     \n"
          "subs %w[cnt], %w[cnt], #1    \n"
          "fmul v8.8h,  v16.8h, v0.h[0] \n"
          "fmul v9.8h,  v17.8h, v0.h[1] \n"
          "ldp  q16, q17, [%[a]], #32\n"
          "fmul v10.8h, v18.8h, v0.h[2] \n"
          "fmul v11.8h, v19.8h, v0.h[3] \n"
          "ldp  q18, q19, [%[a]], #32\n"
          "add  %[b],   %[b],   %[ldb]  \n"
          "fmla v8.8h,  v16.8h, v0.h[4] \n"
          "fmla v9.8h,  v17.8h, v0.h[5] \n"
          /* load a0, a1 */
          "ldp  q16, q17, [%[a]], #32   \n"
          "fmla v10.8h, v18.8h, v0.h[6] \n"
          "fmla v11.8h, v19.8h, v0.h[7] \n"
          /* load b0 */
          "ldr  q0,  [%[b]], #16        \n"
          /* load a2, a3 */
          "ldp  q18, q19, [%[a]], #32   \n"
          "beq  2f                      \n"
          "1:\n"
          "sub  %[b],   %[b],   #16     \n"
          "subs %w[cnt], %w[cnt], #1    \n"
          "fmla v8.8h,  v16.8h, v0.h[0] \n"
          "fmla v9.8h,  v17.8h, v0.h[1] \n"
          "ldp  q16, q17, [%[a]], #32   \n"
          "fmla v10.8h, v18.8h, v0.h[2] \n"
          "fmla v11.8h, v19.8h, v0.h[3] \n"
          "ldp  q18, q19, [%[a]], #32   \n"
          "add  %[b],   %[b],   %[ldb]  \n"
          "fmla v8.8h,  v16.8h, v0.h[4] \n"
          "fmla v9.8h,  v17.8h, v0.h[5] \n"
          /* load a0, a1 */
          "ldp  q16, q17, [%[a]], #32   \n"
          "fmla v10.8h, v18.8h, v0.h[6] \n"
          "fmla v11.8h, v19.8h, v0.h[7] \n"
          /* load b0 */
          "ldr  q0,  [%[b]], #16        \n"
          /* load a2, a3 */
          "ldp  q18, q19, [%[a]], #32   \n"
          "bne  1b                      \n"
          "2:\n"
          "fadd v16.8h,  v8.8h,  v9.8h  \n"
          "fadd v17.8h,  v10.8h, v11.8h \n"
          "fadd v8.8h,  v16.8h,  v17.8h \n"
          "st1  {v8.8h}, [%[c]], #16    \n"
          : [a] "+r"(a_ptr), [b] "+r"(b_ptr), [c] "+r"(C), [cnt] "+r"(cnt)
          : [ldb] "r"(ldb_byte)
          : "v0",
            "v8",
            "v9",
            "v10",
            "v11",
            "v16",
            "v17",
            "v18",
            "v19",
            "cc",
            "memory");
      b += 8;
    }
#else
    for (; n > 7; n -= 8) {
      int cnt = kcnt;
      const float16_t* a_ptr = A_packed;
      const float16_t* b_ptr = b;
      // clang-format off
      asm volatile(
        "1:\n"
        "vmov.u16 q8, #0\n"
        "vmov.u16 q9, #0\n"
        "vmov.u16 q10, #0\n"
        "vmov.u16 q11, #0\n"
        "vmov.u16 q12, #0\n"
        "vmov.u16 q13, #0\n"
        "vmov.u16 q14, #0\n"
        "vmov.u16 q15, #0\n"
        "0:\n"
        "pld [%[a_ptr]]\n"
        "pld [%[b_ptr]]\n"
        "vld1.16 {d8-d9}, [%[a_ptr]]!\n"
        "vldr d0, [%[b_ptr], #0x00]\n"
        "vldr d2, [%[b_ptr], #0x10]\n"
        "vldr d4, [%[b_ptr], #0x20]\n"
        "vldr d6, [%[b_ptr], #0x30]\n"
        "vld1.16 {d10-d11}, [%[a_ptr]]!\n"
        "vldr d1, [%[b_ptr], #0x40]\n"
        "vldr d3, [%[b_ptr], #0x50]\n"
        "pld [%[a_ptr]]\n"
        "vldr d5, [%[b_ptr], #0x60]\n"
        "vldr d7, [%[b_ptr], #0x70]\n"
        "vld1.16 {d12-d13}, [%[a_ptr]]!\n"
        "vmla.f16 q8,  q4, d0[0]\n"
        "vmla.f16 q9,  q4, d2[0]\n"
        "vmla.f16 q10, q4, d4[0]\n"
        "vmla.f16 q11, q4, d6[0]\n"
        "vld1.16 {d14-d15}, [%[a_ptr]]!\n"
        "vmla.f16 q12, q4, d1[0]\n"
        "vmla.f16 q13, q4, d3[0]\n"
        "vmla.f16 q14, q4, d5[0]\n"
        "vmla.f16 q15, q4, d7[0]\n"
        "vld1.16 {d8-d9}, [%[a_ptr]]!\n"
        "vmla.f16 q8,  q5, d0[1]\n"
        "vmla.f16 q9,  q5, d2[1]\n"
        "vmla.f16 q10, q5, d4[1]\n"
        "vmla.f16 q11, q5, d6[1]\n"
        "vmla.f16 q12, q5, d1[1]\n"
        "vmla.f16 q13, q5, d3[1]\n"
        "vmla.f16 q14, q5, d5[1]\n"
        "vmla.f16 q15, q5, d7[1]\n"
        "vld1.16 {d10-d11}, [%[a_ptr]]!\n"

        "vmla.f16 q8,  q6, d0[2]\n"
        "vmla.f16 q9,  q6, d2[2]\n"
        "vmla.f16 q10, q6, d4[2]\n"
        "vmla.f16 q11, q6, d6[2]\n"
        "vmla.f16 q12, q6, d1[2]\n"
        "vmla.f16 q13, q6, d3[2]\n"
        "vmla.f16 q14, q6, d5[2]\n"
        "vmla.f16 q15, q6, d7[2]\n"
        "vld1.16 {d12-d13}, [%[a_ptr]]!\n"

        "vmla.f16 q8,  q7, d0[3]\n"
        "vmla.f16 q9,  q7, d2[3]\n"
        "vmla.f16 q10, q7, d4[3]\n"
        "vmla.f16 q11, q7, d6[3]\n"
        "vldr d0, [%[b_ptr], #0x08]\n"
        "vmla.f16 q12, q7, d1[3]\n"
        "vldr d2, [%[b_ptr], #0x18]\n"
        "vmla.f16 q13, q7, d3[3]\n"
        "vldr d4, [%[b_ptr], #0x28]\n"
        "vmla.f16 q14, q7, d5[3]\n"
        "vldr d6, [%[b_ptr], #0x38]\n"
        "vmla.f16 q15, q7, d7[3]\n"
        "vldr d1, [%[b_ptr], #0x48]\n"
        "vld1.16 {d14-d15}, [%[a_ptr]]!\n"
        "vldr d3, [%[b_ptr], #0x58]\n"

        "vmla.f16 q8,  q4, d0[0]\n"
        "vldr d5, [%[b_ptr], #0x68]\n"
        "vmla.f16 q9,  q4, d2[0]\n"
        "vldr d7, [%[b_ptr], #0x78]\n"
        "vmla.f16 q10, q4, d4[0]\n"
        "vmla.f16 q11, q4, d6[0]\n"
        "vmla.f16 q12, q4, d1[0]\n"
        "vmla.f16 q13, q4, d3[0]\n"
        "vmla.f16 q14, q4, d5[0]\n"
        "vmla.f16 q15, q4, d7[0]\n"

        "vmla.f16 q8,  q5, d0[1]\n"
        "vmla.f16 q9,  q5, d2[1]\n"
        "vmla.f16 q10, q5, d4[1]\n"
        "vmla.f16 q11, q5, d6[1]\n"
        "vmla.f16 q12, q5, d1[1]\n"
        "vmla.f16 q13, q5, d3[1]\n"
        "vmla.f16 q14, q5, d5[1]\n"
        "vmla.f16 q15, q5, d7[1]\n"

        "vmla.f16 q8,  q6, d0[2]\n"
        "vmla.f16 q9,  q6, d2[2]\n"
        "vmla.f16 q10, q6, d4[2]\n"
        "vmla.f16 q11, q6, d6[2]\n"
        "vmla.f16 q12, q6, d1[2]\n"
        "vmla.f16 q13, q6, d3[2]\n"
        "vmla.f16 q14, q6, d5[2]\n"
        "vmla.f16 q15, q6, d7[2]\n"

        "vmla.f16 q8,  q7, d0[3]\n"
        "vmla.f16 q9,  q7, d2[3]\n"
        "vmla.f16 q10, q7, d4[3]\n"
        "vmla.f16 q11, q7, d6[3]\n"
        "subs %[cnt], #1\n"
        "add  %[b_ptr], %[ldb]\n"
        "vmla.f16 q12, q7, d1[3]\n"
        "vmla.f16 q13, q7, d3[3]\n"
        "vmla.f16 q14, q7, d5[3]\n"
        "vmla.f16 q15, q7, d7[3]\n"
        "bne  0b\n"
        "2:\n"
        "vst1.16 {d16-d19}, [%[c]]!\n"
        "vst1.16 {d20-d23}, [%[c]]!\n"
        "vst1.16 {d24-d27}, [%[c]]!\n"
        "vst1.16 {d28-d31}, [%[c]]!\n"
        : [a_ptr] "+r" (a_ptr),
          [b_ptr] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [ldb]  "r" (ldb_byte)
        : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
          "q10", "q11", "q12", "q13", "q14", "q15", "cc", "memory"
      );
      // clang-format on
      b += 64;
    }
    if (n > 3) {
      int cnt = kcnt;
      const float16_t* a_ptr = A_packed;
      const float16_t* b_ptr = b;
      // clang-format off
      asm volatile(
        "1: \n"
        "vmov.u16 q8, #0\n"
        "vmov.u16 q9, #0\n"
        "vmov.u16 q10, #0\n"
        "vmov.u16 q11, #0\n"
        "0:\n"
        "pld [%[a_ptr]]\n"
        "pld [%[b_ptr]]\n"
        "vld1.16 {d8-d9}, [%[a_ptr]]!\n"
        "vldr d0, [%[b_ptr], #0x00]\n"
        "vldr d2, [%[b_ptr], #0x10]\n"
        "vldr d4, [%[b_ptr], #0x20]\n"
        "vldr d6, [%[b_ptr], #0x30]\n"
        "vld1.16 {d10-d11}, [%[a_ptr]]!\n"
        "vldr d1, [%[b_ptr], #0x08]\n"
        "vldr d3, [%[b_ptr], #0x18]\n"
        "pld [%[a_ptr]]\n"
        "vldr d5, [%[b_ptr], #0x28]\n"
        "vldr d7, [%[b_ptr], #0x38]\n"
        "vld1.16 {d12-d13}, [%[a_ptr]]!\n"
        "vmla.f16 q8,  q4, d0[0]\n"
        "vmla.f16 q9,  q4, d2[0]\n"
        "vmla.f16 q10, q4, d4[0]\n"
        "vmla.f16 q11, q4, d6[0]\n"
        "vld1.16 {d14-d15}, [%[a_ptr]]!\n"

        "vmla.f16 q8,  q5, d0[1]\n"
        "vmla.f16 q9,  q5, d2[1]\n"
        "vmla.f16 q10, q5, d4[1]\n"
        "vmla.f16 q11, q5, d6[1]\n"
        "vld1.16 {d8-d9}, [%[a_ptr]]!\n"

        "vmla.f16 q8,  q6, d0[2]\n"
        "vmla.f16 q9,  q6, d2[2]\n"
        "vmla.f16 q10, q6, d4[2]\n"
        "vmla.f16 q11, q6, d6[2]\n"
        "vld1.16 {d10-d11}, [%[a_ptr]]!\n"

        "vmla.f16 q8,  q7, d0[3]\n"
        "vmla.f16 q9,  q7, d2[3]\n"
        "vmla.f16 q10, q7, d4[3]\n"
        "vmla.f16 q11, q7, d6[3]\n"
        "vld1.16 {d12-d13}, [%[a_ptr]]!\n"

        "vmla.f16 q8,  q4, d1[0]\n"
        "vmla.f16 q9,  q4, d3[0]\n"
        "vmla.f16 q10, q4, d5[0]\n"
        "vmla.f16 q11, q4, d7[0]\n"
        "vld1.16 {d14-d15}, [%[a_ptr]]!\n"

        "vmla.f16 q8,  q5, d1[1]\n"
        "vmla.f16 q9,  q5, d3[1]\n"
        "vmla.f16 q10, q5, d5[1]\n"
        "vmla.f16 q11, q5, d7[1]\n"

        "vmla.f16 q8,  q6, d1[2]\n"
        "vmla.f16 q9,  q6, d3[2]\n"
        "vmla.f16 q10, q6, d5[2]\n"
        "vmla.f16 q11, q6, d7[2]\n"
        "subs %[cnt], #1\n"
        "add  %[b_ptr], %[ldb]\n"

        "vmla.f16 q8,  q7, d1[3]\n"
        "vmla.f16 q9,  q7, d3[3]\n"
        "vmla.f16 q10, q7, d5[3]\n"
        "vmla.f16 q11, q7, d7[3]\n"
        "bne  0b\n"
        "2:\n"
        "vst1.16 {d16-d19}, [%[c]]!\n"
        "vst1.16 {d20-d23}, [%[c]]!\n"
        : [a_ptr] "+r" (a_ptr),
          [b_ptr] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [ldb]  "r" (ldb_byte)
        : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
          "q10", "q11", "q12", "q13", "q14", "q15", "cc", "memory"
      );
      // clang-format on
      b += 32;
      n -= 4;
    }
    for (; n > 0; n--) {
      int cnt = kcnt;
      const float16_t* a_ptr = A_packed;
      const float16_t* b_ptr = b;
      // clang-format off
      asm volatile(
        "1: \n"
        "vmov.u16 q8, #0\n"
        "vmov.u16 q9, #0\n"
        "vmov.u16 q10, #0\n"
        "vmov.u16 q11, #0\n"
        "0:\n"
        "pld [%[a_ptr]]\n"
        "pld [%[b_ptr]]\n"
        "vld1.16 {d8-d11}, [%[a_ptr]]!\n"
        "vldr d0, [%[b_ptr], #0x00]\n"
        "vldr d2, [%[b_ptr], #0x10]\n"
        "vldr d4, [%[b_ptr], #0x20]\n"
        "vldr d6, [%[b_ptr], #0x30]\n"
        "vld1.16 {d12-d15}, [%[a_ptr]]!\n"
        "vldr d1, [%[b_ptr], #0x08]\n"
        "vldr d3, [%[b_ptr], #0x18]\n"
        "pld [%[a_ptr]]\n"
        "vmla.f16 q8,  q4, d0[0]\n"
        "vmla.f16 q9,  q5, d0[1]\n"
        "vld1.16 {d8-d11}, [%[a_ptr]]!\n"
        "vmla.f16 q10, q6, d0[2]\n"
        "vmla.f16 q11, q7, d0[3]\n"
        "add  %[b_ptr], %[ldb]\n"
        "vld1.16 {d12-d15}, [%[a_ptr]]!\n"

        "subs %[cnt], #1\n"
        "vmla.f16 q8,  q4, d1[0]\n"
        "vmla.f16 q9,  q5, d1[1]\n"
        "vmla.f16 q10, q6, d1[2]\n"
        "vmla.f16 q11, q7, d1[3]\n"
        "bne  0b\n"
        "2:\n"
        "vadd.f16 q0, q8, q9\n"
        "vadd.f16 q1, q10, q11\n"
        "vadd.f16 q2, q0, q1\n"
        "vst1.16 {d4-d5}, [%[c]]!\n"
        : [a_ptr] "+r" (a_ptr),
          [b_ptr] "+r" (b_ptr),
          [c] "+r" (C),
          [cnt] "+r" (cnt)
        : [ldb]  "r" (ldb_byte)
        : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
          "q10", "q11", "q12", "q13", "q14", "q15", "cc", "memory"
      );
      // clang-format on
      b += 8;
    }
#endif
    A_packed += lda;
  }
}

void gemm_prepack_c8_fp16(int M,
                          int N,
                          int K,
                          const float16_t* A_packed,
                          const float16_t* B,
                          float16_t* C,
                          ARMContext* ctx) {
  if (N > 16) {
    gemm_prepack_c8_fp16_common(M, N, K, A_packed, B, C, ctx);
  } else {
    gemm_prepack_c8_fp16_small(M, N, K, A_packed, B, C, ctx);
  }
}

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
