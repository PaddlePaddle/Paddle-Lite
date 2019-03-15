/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "operators/math/gemm.h"
#include <string.h>
#include "common/log.h"
#include "memory/t_malloc.h"
#if __ARM_NEON
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {
namespace math {

#if __ARM_NEON
inline float32x4_t vandq_f32(float32x4_t x, uint32x4_t mask) {
  return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), mask));
}
#endif

void Gemm::PackMatrixA_6r(int m, int k, int m_tail, const float *A, int lda,
                          float *buffer, const bool parallel) {
  uint32_t mask[8] = {0, 1, 2, 3, 4, 5, 4, 5};
  int remain_k = k & 0x3;
  uint32x4_t vzero = vdupq_n_u32(0);
  uint32x4_t vmask1 = vcltq_u32(vld1q_u32(mask), vdupq_n_u32(remain_k));

  #pragma omp parallel for if (parallel)
  // num_threads(framework::threads())
  for (int i = 0; i < m - 5; i += 6) {
    const float *a0 = A + i * lda;
    const float *a1 = A + (i + 1) * lda;
    const float *a2 = A + (i + 2) * lda;
    const float *a3 = A + (i + 3) * lda;
    const float *a4 = A + (i + 4) * lda;
    const float *a5 = A + (i + 5) * lda;
    float *out_ptr = buffer + i * k;

    int loops = k >> 2;
    if (loops > 0) {
#if __aarch64__
      for (int l = 0; l < loops; ++l) {
        float32x4_t _d0 = vld1q_f32(a0);
        float32x4_t _d1 = vld1q_f32(a1);
        float32x4_t _d2 = vld1q_f32(a2);
        float32x4_t _d3 = vld1q_f32(a3);
        float32x4_t _d4 = vld1q_f32(a4);
        float32x4_t _d5 = vld1q_f32(a5);

        float32x4x2_t _q0 = vtrnq_f32(_d0, _d1);
        float32x4x2_t _q1 = vtrnq_f32(_d2, _d3);
        float32x4x2_t _q3 = vtrnq_f32(_d4, _d5);
        _d0 = vcombine_f32(vget_low_f32(_q0.val[0]), vget_low_f32(_q1.val[0]));
        _d1 = vcombine_f32(vget_low_f32(_q0.val[1]), vget_low_f32(_q1.val[1]));
        _d2 =
            vcombine_f32(vget_high_f32(_q0.val[0]), vget_high_f32(_q1.val[0]));
        _d3 =
            vcombine_f32(vget_high_f32(_q0.val[1]), vget_high_f32(_q1.val[1]));

        vst1q_f32(out_ptr, _d0);
        vst1_f32(out_ptr + 4, vget_low_f32(_q3.val[0]));
        vst1q_f32(out_ptr + 6, _d1);
        vst1_f32(out_ptr + 10, vget_low_f32(_q3.val[1]));
        vst1q_f32(out_ptr + 12, _d2);
        vst1_f32(out_ptr + 16, vget_high_f32(_q3.val[0]));
        vst1q_f32(out_ptr + 18, _d3);
        vst1_f32(out_ptr + 22, vget_high_f32(_q3.val[1]));

        a0 += 4;
        a1 += 4;
        a2 += 4;
        a3 += 4;
        a4 += 4;
        a5 += 4;
        out_ptr += 24;
      }
#else
      asm volatile(
          "loop_4k_%=:                        \n"
          "vld1.32    {d0-d1}, [%[a0]]!       \n"
          "vld1.32    {d2-d3}, [%[a1]]!       \n"
          "vld1.32    {d4-d5}, [%[a2]]!       \n"
          "vld1.32    {d6-d7}, [%[a3]]!       \n"
          "vld1.32    {d8-d9}, [%[a4]]!       \n"
          "vld1.32    {d10-d11}, [%[a5]]!     \n"
          "vtrn.32    q0, q1                  \n"
          "vtrn.32    q2, q3                  \n"
          "vtrn.32    q4, q5                  \n"
          "vswp.32    d1, d4                  \n"
          "vswp.32    d3, d6                  \n"

          "vst1.32    {q0}, [%[out]]!         \n"
          "vst1.32    {d8}, [%[out]]!         \n"
          "vst1.32    {q1}, [%[out]]!         \n"
          "vst1.32    {d10}, [%[out]]!        \n"
          "vst1.32    {q2}, [%[out]]!         \n"
          "vst1.32    {d9}, [%[out]]!         \n"
          "vst1.32    {q3}, [%[out]]!         \n"
          "vst1.32    {d11}, [%[out]]!        \n"

          "subs       %[loops], #1            \n"
          "bne        loop_4k_%=              \n"
          : [out] "+r"(out_ptr), [a0] "+r"(a0), [a1] "+r"(a1), [a2] "+r"(a2),
            [a3] "+r"(a3), [a4] "+r"(a4), [a5] "+r"(a5), [loops] "+r"(loops)
          :
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5");
#endif
    }

    if (remain_k > 0) {
      float32x4_t _d0 = vld1q_f32(a0);
      float32x4_t _d1 = vld1q_f32(a1);
      float32x4_t _d2 = vld1q_f32(a2);
      float32x4_t _d3 = vld1q_f32(a3);
      float32x4_t _d4 = vld1q_f32(a4);
      float32x4_t _d5 = vld1q_f32(a5);

      _d0 = vandq_f32(_d0, vmask1);
      _d1 = vandq_f32(_d1, vmask1);
      _d2 = vandq_f32(_d2, vmask1);
      _d3 = vandq_f32(_d3, vmask1);
      _d4 = vandq_f32(_d4, vmask1);
      _d5 = vandq_f32(_d5, vmask1);

      float32x4x2_t _q0 = vtrnq_f32(_d0, _d1);
      float32x4x2_t _q1 = vtrnq_f32(_d2, _d3);
      float32x4x2_t _q3 = vtrnq_f32(_d4, _d5);
      _d0 = vcombine_f32(vget_low_f32(_q0.val[0]), vget_low_f32(_q1.val[0]));
      _d1 = vcombine_f32(vget_low_f32(_q0.val[1]), vget_low_f32(_q1.val[1]));
      _d2 = vcombine_f32(vget_high_f32(_q0.val[0]), vget_high_f32(_q1.val[0]));

      switch (remain_k) {
        case 3:
          vst1q_f32(out_ptr + 12, _d2);
          vst1_f32(out_ptr + 16, vget_high_f32(_q3.val[0]));
        case 2:
          vst1q_f32(out_ptr + 6, _d1);
          vst1_f32(out_ptr + 10, vget_low_f32(_q3.val[1]));
        case 1:
          vst1q_f32(out_ptr, _d0);
          vst1_f32(out_ptr + 4, vget_low_f32(_q3.val[0]));
        default:
          break;
      }
    }
  }

  int remain_m = m % 6;
  if (remain_m) {
    int remain_m_start = m - remain_m;
    const float *a0 = A + remain_m_start * lda;
    const float *a1 = a0 + lda;
    const float *a2 = a0 + 2 * lda;
    const float *a3 = a0 + 3 * lda;
    const float *a4 = a0 + 4 * lda;
    const float *a5 = a0 + 5 * lda;
    float *out_ptr = buffer + remain_m_start * k;

    uint32x4_t vmask2 = vcltq_u32(vld1q_u32(mask), vdupq_n_u32(remain_m));
    uint32x4_t vmask3 = vcltq_u32(vld1q_u32(mask + 4), vdupq_n_u32(remain_m));

    int loops = k >> 2;
    if (loops > 0) {
#if __aarch64__
      for (int l = 0; l < loops; ++l) {
        float32x4_t _d0 = vld1q_f32(a0);
        float32x4_t _d1 = vld1q_f32(a1);
        float32x4_t _d2 = vld1q_f32(a2);
        float32x4_t _d3 = vld1q_f32(a3);
        float32x4_t _d4 = vld1q_f32(a4);
        float32x4_t _d5 = vld1q_f32(a5);

        float32x4x2_t _q0 = vtrnq_f32(_d0, _d1);
        float32x4x2_t _q1 = vtrnq_f32(_d2, _d3);
        float32x4x2_t _q3 = vtrnq_f32(_d4, _d5);
        _d0 = vcombine_f32(vget_low_f32(_q0.val[0]), vget_low_f32(_q1.val[0]));
        _d1 = vcombine_f32(vget_low_f32(_q0.val[1]), vget_low_f32(_q1.val[1]));
        _d2 =
            vcombine_f32(vget_high_f32(_q0.val[0]), vget_high_f32(_q1.val[0]));
        _d3 =
            vcombine_f32(vget_high_f32(_q0.val[1]), vget_high_f32(_q1.val[1]));

        _d0 = vandq_f32(_d0, vmask2);
        _d1 = vandq_f32(_d1, vmask2);
        _d2 = vandq_f32(_d2, vmask2);
        _d3 = vandq_f32(_d3, vmask2);
        _d4 = vandq_f32(_q3.val[0], vmask3);
        _d5 = vandq_f32(_q3.val[1], vmask3);

        vst1q_f32(out_ptr, _d0);
        vst1_f32(out_ptr + 4, vget_low_f32(_d4));
        vst1q_f32(out_ptr + 6, _d1);
        vst1_f32(out_ptr + 10, vget_low_f32(_d5));
        vst1q_f32(out_ptr + 12, _d2);
        vst1_f32(out_ptr + 16, vget_high_f32(_d4));
        vst1q_f32(out_ptr + 18, _d3);
        vst1_f32(out_ptr + 22, vget_high_f32(_d5));

        a0 += 4;
        a1 += 4;
        a2 += 4;
        a3 += 4;
        a4 += 4;
        a5 += 4;
        out_ptr += 24;
      }
#else
      asm volatile(
          "loop_4k_%=:                        \n"
          "vld1.32    {d0-d1}, [%[a0]]!       \n"
          "vld1.32    {d2-d3}, [%[a1]]!       \n"
          "vld1.32    {d4-d5}, [%[a2]]!       \n"
          "vld1.32    {d6-d7}, [%[a3]]!       \n"
          "vld1.32    {d8-d9}, [%[a4]]!       \n"
          "vld1.32    {d10-d11}, [%[a5]]!     \n"
          "vtrn.32    q0, q1                  \n"
          "vtrn.32    q2, q3                  \n"
          "vtrn.32    q4, q5                  \n"
          "vswp.32    d1, d4                  \n"
          "vswp.32    d3, d6                  \n"

          "vbif       q0, %q[vzero], %q[vmask2] \n"
          "vbif       q1, %q[vzero], %q[vmask2] \n"
          "vbif       q2, %q[vzero], %q[vmask2] \n"
          "vbif       q3, %q[vzero], %q[vmask2] \n"
          "vbif       q4, %q[vzero], %q[vmask3] \n"
          "vbif       q5, %q[vzero], %q[vmask3] \n"

          "vst1.32    {q0}, [%[out]]!         \n"
          "vst1.32    {d8}, [%[out]]!         \n"
          "vst1.32    {q1}, [%[out]]!         \n"
          "vst1.32    {d10}, [%[out]]!        \n"
          "vst1.32    {q2}, [%[out]]!         \n"
          "vst1.32    {d9}, [%[out]]!         \n"
          "vst1.32    {q3}, [%[out]]!         \n"
          "vst1.32    {d11}, [%[out]]!        \n"

          "subs       %[loops], #1            \n"
          "bne        loop_4k_%=              \n"
          : [out] "+r"(out_ptr), [a0] "+r"(a0), [a1] "+r"(a1), [a2] "+r"(a2),
            [a3] "+r"(a3), [a4] "+r"(a4), [a5] "+r"(a5), [loops] "+r"(loops)
          : [vmask2] "w"(vmask2), [vmask3] "w"(vmask3), [vzero] "w"(vzero)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5");
#endif
    }

    if (remain_k > 0) {
      float32x4_t _d0 = vld1q_f32(a0);
      float32x4_t _d1 = vld1q_f32(a1);
      float32x4_t _d2 = vld1q_f32(a2);
      float32x4_t _d3 = vld1q_f32(a3);
      float32x4_t _d4 = vld1q_f32(a4);
      float32x4_t _d5 = vld1q_f32(a5);

      _d0 = vandq_f32(_d0, vmask1);
      _d1 = vandq_f32(_d1, vmask1);
      _d2 = vandq_f32(_d2, vmask1);
      _d3 = vandq_f32(_d3, vmask1);
      _d4 = vandq_f32(_d4, vmask1);
      _d5 = vandq_f32(_d5, vmask1);

      float32x4x2_t _q0 = vtrnq_f32(_d0, _d1);
      float32x4x2_t _q1 = vtrnq_f32(_d2, _d3);
      float32x4x2_t _q3 = vtrnq_f32(_d4, _d5);
      _d0 = vcombine_f32(vget_low_f32(_q0.val[0]), vget_low_f32(_q1.val[0]));
      _d1 = vcombine_f32(vget_low_f32(_q0.val[1]), vget_low_f32(_q1.val[1]));
      _d2 = vcombine_f32(vget_high_f32(_q0.val[0]), vget_high_f32(_q1.val[0]));
      // _d3 = vcombine_f32(vget_high_f32(_q0.val[1]),
      // vget_high_f32(_q1.val[1]));

      _d0 = vandq_f32(_d0, vmask2);
      _d1 = vandq_f32(_d1, vmask2);
      _d2 = vandq_f32(_d2, vmask2);
      // _d3 = vandq_f32(_d3, vmask2);
      _d4 = vandq_f32(_q3.val[0], vmask3);
      _d5 = vandq_f32(_q3.val[1], vmask3);

      switch (remain_k) {
        case 3:
          vst1q_f32(out_ptr + 12, _d2);
          vst1_f32(out_ptr + 16, vget_high_f32(_d4));
        case 2:
          vst1q_f32(out_ptr + 6, _d1);
          vst1_f32(out_ptr + 10, vget_low_f32(_d5));
        case 1:
          vst1q_f32(out_ptr, _d0);
          vst1_f32(out_ptr + 4, vget_low_f32(_d4));
        default:
          break;
      }
    }
  }
}

// 将B矩阵分块复制到连续内存(RowMajor)
void Gemm::PackMatrixB_8c(int k, int n, int n_tail, const float *B, int ldb,
                          float *buffer, const bool parallel) {
  const int j_length = n - n_tail;

  #pragma omp parallel for if (parallel)
  // num_threads(framework::threads())
  for (int i = 0; i < k; ++i) {
    int j = 0;
    for (; j < j_length - 31; j += 32) {
      float *local_buffer0 = buffer + j * k + i * NR;
      float *local_buffer1 = buffer + (j + 8) * k + i * NR;
      float *local_buffer2 = buffer + (j + 16) * k + i * NR;
      float *local_buffer3 = buffer + (j + 24) * k + i * NR;
      const float *b0 = B + i * ldb + j;
#if __aarch64__
      asm volatile(
          "prfm   pldl1keep,       [%[b0]]                 \n"
          "ld1    {v0.4s, v1.4s},  [%[b0]], #32            \n"
          "ld1    {v2.4s, v3.4s},  [%[b0]], #32            \n"
          "ld1    {v4.4s, v5.4s},  [%[b0]], #32            \n"
          "ld1    {v6.4s, v7.4s},  [%[b0]]                 \n"
          "st1    {v0.4s, v1.4s},  [%[local_buffer0]], #32 \n"
          "st1    {v2.4s, v3.4s},  [%[local_buffer1]], #32 \n"
          "st1    {v4.4s, v5.4s},  [%[local_buffer2]], #32 \n"
          "st1    {v6.4s, v7.4s},  [%[local_buffer3]], #32 \n"
          : [local_buffer0] "+r"(local_buffer0),
            [local_buffer1] "+r"(local_buffer1),
            [local_buffer2] "+r"(local_buffer2),
            [local_buffer3] "+r"(local_buffer3), [b0] "+r"(b0)
          :
          : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else
      asm volatile(
          // "pld      [%[b]]                          \n"
          "vld1.32  {q0, q1},   [%[b0]]!             \n"
          "vld1.32  {q2, q3},   [%[b0]]!             \n"
          "vld1.32  {q4, q5},   [%[b0]]!             \n"
          "vld1.32  {q6, q7},   [%[b0]]!             \n"
          "vst1.32  {q0, q1},   [%[local_buffer0]]!  \n"
          "vst1.32  {q2, q3},   [%[local_buffer1]]!  \n"
          "vst1.32  {q4, q5},   [%[local_buffer2]]!  \n"
          "vst1.32  {q6, q7},   [%[local_buffer3]]!  \n"
          : [local_buffer0] "+r"(local_buffer0),
            [local_buffer1] "+r"(local_buffer1),
            [local_buffer2] "+r"(local_buffer2),
            [local_buffer3] "+r"(local_buffer3), [b0] "+r"(b0)
          :
          : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
#endif  // __aarch64__
    }
    for (; j < j_length - 15; j += 16) {
      float *local_buffer0 = buffer + j * k + i * NR;
      float *local_buffer1 = buffer + (j + 8) * k + i * NR;
      const float *b0 = &B(i, j);
#if __ARM_NEON
#if __aarch64__
      asm volatile(
          "prfm   pldl1keep,        [%[b0]]            \n"
          "ld1    {v0.4s, v1.4s},   [%[b0]], #32       \n"
          "ld1    {v2.4s, v3.4s},   [%[b0]]            \n"
          "st1    {v0.4s, v1.4s},   [%[local_buffer0]],  #32 \n"
          "st1    {v2.4s, v3.4s},   [%[local_buffer1]],  #32 \n"
          : [local_buffer0] "+r"(local_buffer0),
            [local_buffer1] "+r"(local_buffer1), [b0] "+r"(b0)
          :
          : "memory", "v0", "v1", "v2", "v3");
#else
      asm volatile(
          //          "pld        [%[b0]]                     \n"
          "vld1.32    {q0, q1},   [%[b0]]!               \n"
          "vld1.32    {q2, q3},   [%[b0]]                \n"
          "vst1.32    {q0, q1},   [%[local_buffer0]]!    \n"
          "vst1.32    {q2, q3},   [%[local_buffer1]]!    \n"
          : [local_buffer0] "+r"(local_buffer0),
            [local_buffer1] "+r"(local_buffer1), [b0] "+r"(b0)
          :
          : "memory", "q0", "q1", "q2", "q3");
#endif  // __aarch64__
#endif  // __ARM_NEON
    }
    for (; j < j_length; j += NR) {
      float *local_buffer = buffer + j * k + i * NR;
      const float *b0 = &B(i, j);
#if __aarch64__
      asm volatile(
          "prfm     pldl1keep,       [%[b0]]            \n"
          "ld1      {v0.4s, v1.4s},  [%[b0]]            \n"
          "st1      {v0.4s, v1.4s},  [%[local_buffer]], #32 \n"
          : [local_buffer] "+r"(local_buffer)
          : [b0] "r"(b0)
          : "memory", "v0", "v1");
#else
      asm volatile(
          // "pld      [%[b]]                          \n"
          "vld1.32  {q0, q1},   [%[b0]]              \n"
          "vst1.32  {q0, q1},   [%[local_buffer]]        \n"
          : [local_buffer] "+r"(local_buffer)
          : [b0] "r"(b0)
          : "memory", "q0", "q1");
#endif  // __aarch64__
    }
  }
  if (n_tail != 0) {
    uint32_t mask[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint32x4_t vzero = vdupq_n_u32(0);
    uint32x4_t vmask1 = vcltq_u32(vld1q_u32(mask), vdupq_n_u32(n_tail));
    uint32x4_t vmask2 = vcltq_u32(vld1q_u32(mask + 4), vdupq_n_u32(n_tail));

    float *local_buffer = buffer + j_length * k;
    for (int i = 0; i < k; ++i) {
      const float *b0 = &B(i, j_length);
#if __aarch64__
      asm volatile(
          "prfm   pldl1keep,       [%[b0]]            \n"
          "ld1    {v0.4s, v1.4s},  [%[b0]]            \n"
          "BIF    v0.8b, %[vzero].8b, %[vmask1].8b    \n"
          "BIF    v1.8b, %[vzero].8b, %[vmask2].8b    \n"
          "st1      {v0.4s, v1.4s},  [%[local_buffer]], #32 \n"
          : [local_buffer] "+r"(local_buffer)
          : [vmask1] "w"(vmask1), [vmask2] "w"(vmask2), [vzero] "w"(vzero),
            [b0] "r"(b0)
          : "memory", "v0", "v1");
#else
      asm volatile(
          "vld1.32  {q0, q1},   [%[b0]]              \n"
          "vbif     q0, %q[vzero], %q[vmask1]        \n"
          "vbif     q1, %q[vzero], %q[vmask2]        \n"
          "vst1.32  {q0, q1},   [%[local_buffer]]!   \n"
          : [local_buffer] "+r"(local_buffer)
          : [vmask1] "w"(vmask1), [vmask2] "w"(vmask2), [vzero] "w"(vzero),
            [b0] "r"(b0)
          : "memory", "q0", "q1");
#endif
    }
  }
}

#if __ARM_NEON
#if __aarch64__
void Gemm::PackMatrixB_12c(int k, int n, int n_tail, const float *B, int ldb,
                           float *buffer, const bool parallel) {
  const int j_length = n - n_tail;

  #pragma omp parallel for if (parallel)
  // num_threads(framework::threads())
  for (int j = 0; j < j_length; j += NR) {
    float *local_buffer = buffer + j * k;
    for (int i = 0; i < k; ++i) {
      const float *b0 = &B(i, j);
      asm volatile(
          "prfm   pldl2keep,        [%[b0], #64]           \n\t"
          "ld1    {v0.4s, v1.4s, v2.4s},   [%[b0]]           \n\t"
          "st1    {v0.4s, v1.4s, v2.4s},   [%[local_buffer]],  #48 \n\t"
          : [local_buffer] "+r"(local_buffer)
          : [b0] "r"(b0)
          : "memory", "v0", "v1", "v2");
    }
  }
  if (n_tail != 0) {
    float *local_buffer = buffer + j_length * k;
    for (int i = 0; i < k; ++i) {
      const float *b0 = &B(i, j_length);
      for (int j = j_length; j < n; ++j) {
        *local_buffer++ = *b0++;
      }
      for (int j = n; j < j_length + NR; ++j) {
        *local_buffer++ = 0;
      }
    }
  }
}

void Gemm::PackMatrixB_16c(int k, int n, int n_tail, const float *B, int ldb,
                           float *buffer, const bool parallel) {
  const int j_length = n - n_tail;

  #pragma omp parallel for if (parallel)
  // num_threads(framework::threads())
  for (int j = 0; j < n - n_tail; j += NR) {
    float *local_buffer = buffer + j * k;
    for (int i = 0; i < k; ++i) {
      const float *b0 = &B(i, j);
      asm volatile(
          "prfm   pldl2keep,        [%[b0], #64]           \n\t"
          "ld1    {v0.4s, v1.4s, v2.4s, v3.4s},   [%[b0]]           \n\t"
          "st1    {v0.4s, v1.4s, v2.4s, v3.4s},   [%[local_buffer]],  #64 \n\t"
          : [local_buffer] "+r"(local_buffer)
          : [b0] "r"(b0)
          : "memory", "v0", "v1", "v2", "v3");
    }
  }
  if (n_tail != 0) {
    float *local_buffer = buffer + j_length * k;
    for (int i = 0; i < k; ++i) {
      const float *b0 = &B(i, j_length);
      for (int j = j_length; j < n; ++j) {
        *local_buffer++ = *b0++;
      }
      for (int j = n; j < j_length + NR; ++j) {
        *local_buffer++ = 0;
      }
    }
  }
}
#endif  // __aarch64__
#endif  // __ARM_NEON

// 分块矩阵乘法
void Gemm::InnerKernel(int mc, int nc, float alpha, const float *a,
                       const float *b, float beta, float *c, float *C, int ldc,
                       bool relu) {
#pragma omp parallel for num_threads(framework::threads())
  for (int j = 0; j < nc; j += NR) {
    for (int i = 0; i < mc; i += MR) {
#if __aarch64__
      // AddDot8x12(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
      AddDot6x16(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
#else
      // AddDot4x4(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
      // AddDot4x8(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
      AddDot6x8(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
#endif
    }
  }

  if (alpha != 1) {
    WriteWithAlphaBeta(mc, nc, c, C, ldc);
    return;
  }
  if (beta == 0) {
    WriteBasic(mc, nc, c, C, ldc);
    return;
  }
  if (beta == 1 && !relu) {
    WriteWithAdd(mc, nc, c, C, ldc);
    return;
  }
  if (beta == 1 && relu) {
    WriteWithAddRelu(mc, nc, c, C, ldc);
    return;
  }
}

// 分块矩阵乘法
void Gemm::InnerKernelWithBias(int mc, int nc, float alpha, const float *a,
                               const float *b, float beta, float *c, float *C,
                               int ldc, bool relu, float *bias) {
#pragma omp parallel for num_threads(framework::threads())
  for (int j = 0; j < nc; j += NR) {
    for (int i = 0; i < mc; i += MR) {
#if __aarch64__
      // AddDot8x12(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
      AddDot6x16(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
#else
      // AddDot4x4(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
      // AddDot4x8(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
      AddDot6x8(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
#endif
    }
  }

  if (alpha != 1) {
    WriteWithAlphaBeta(mc, nc, c, C, ldc);
    return;
  }
  if (beta == 0) {
    WriteBasic(mc, nc, c, C, ldc);
    return;
  }
  if (beta == 1 && !relu) {
    if (bias == nullptr) {
      WriteWithAdd(mc, nc, c, C, ldc);
    } else {
      WriteWithAddV1(mc, nc, c, C, ldc, bias);
    }
    return;
  }
  if (beta == 1 && relu) {
    if (bias == nullptr) {
      WriteWithAddRelu(mc, nc, c, C, ldc);
    } else {
      WriteWithAddReluV1(mc, nc, c, C, ldc, bias);
    }
    return;
  }
}

// 分块矩阵乘法
void Gemm::InnerKernelWithBn(int mc, int nc, float alpha, const float *a,
                             const float *b, float beta, float *c, float *C,
                             int ldc, bool relu, float *new_scale,
                             float *new_bias) {
#pragma omp parallel for num_threads(framework::threads())
  for (int j = 0; j < nc; j += NR) {
    for (int i = 0; i < mc; i += MR) {
#if __aarch64__
      // AddDot8x12(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
      AddDot6x16(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
#else
      // AddDot4x4(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
      // AddDot4x8(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
      AddDot6x8(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
#endif
    }
  }

  if (relu) {
    WriteWithBnRelu(mc, nc, c, C, ldc, new_scale, new_bias);
  } else {
    WriteWithBn(mc, nc, c, C, ldc, new_scale, new_bias);
  }
}

// 分块矩阵乘法
void Gemm::InnerKernelWithBnAdd(int mc, int nc, float alpha, const float *a,
                                const float *b, float beta, float *c, float *C,
                                int ldc, bool relu, float *new_scale,
                                float *new_bias, float *bias) {
#pragma omp parallel for num_threads(framework::threads())
  for (int j = 0; j < nc; j += NR) {
    for (int i = 0; i < mc; i += MR) {
#if __aarch64__
      // AddDot8x12(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
      AddDot6x16(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
#else
      // AddDot4x4(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
      // AddDot4x8(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
      AddDot6x8(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
#endif
    }
  }
  WriteWithBnAddRelu(mc, nc, c, C, ldc, new_scale, new_bias, bias);
}

void Gemm::InnerKernelWithPRelu(int mc, int nc, const float *a, const float *b,
                                float *c, float *C, int ldc, float *p,
                                std::string mode, float *bias, float *bias1) {
#pragma omp parallel for num_threads(framework::threads())
  for (int j = 0; j < nc; j += NR) {
    for (int i = 0; i < mc; i += MR) {
#if __aarch64__
      // AddDot8x12(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
      AddDot6x16(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
#else
      // AddDot4x4(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
      // AddDot4x8(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
      AddDot6x8(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
#endif
    }
  }
  WriteWithAddPRelu(mc, nc, c, C, ldc, p, mode, bias, bias1);
}

#if __ARM_NEON
#if __aarch64__

void Gemm::AddDot6x8(int k, const float *a, const float *b, float *c, int ldc) {
  // init C
  float32x4_t cv0 = vdupq_n_f32(0.0);
  float32x4_t cv1 = vdupq_n_f32(0.0);
  float32x4_t cv2 = vdupq_n_f32(0.0);
  float32x4_t cv3 = vdupq_n_f32(0.0);
  float32x4_t cv4 = vdupq_n_f32(0.0);
  float32x4_t cv5 = vdupq_n_f32(0.0);
  float32x4_t cv6 = vdupq_n_f32(0.0);
  float32x4_t cv7 = vdupq_n_f32(0.0);
  float32x4_t cv8 = vdupq_n_f32(0.0);
  float32x4_t cv9 = vdupq_n_f32(0.0);
  float32x4_t cv10 = vdupq_n_f32(0.0);
  float32x4_t cv11 = vdupq_n_f32(0.0);

  float32x4_t av;
  float32x4_t bv0;
  float32x4_t bv1;

  float32x2_t av01;
  float32x2_t av23;
  float32x2_t av45;

  for (int p = 0; p < k; p += 1) {
    av = vld1q_f32(a);
    av01 = vget_low_f32(av);
    av23 = vget_high_f32(av);
    av45 = vld1_f32(a + 4);
    bv0 = vld1q_f32(b);
    bv1 = vld1q_f32(b + 4);

    cv0 = vmlaq_lane_f32(cv0, bv0, av01, 0);
    cv1 = vmlaq_lane_f32(cv1, bv1, av01, 0);
    cv2 = vmlaq_lane_f32(cv2, bv0, av01, 1);
    cv3 = vmlaq_lane_f32(cv3, bv1, av01, 1);

    cv4 = vmlaq_lane_f32(cv4, bv0, av23, 0);
    cv5 = vmlaq_lane_f32(cv5, bv1, av23, 0);
    cv6 = vmlaq_lane_f32(cv6, bv0, av23, 1);
    cv7 = vmlaq_lane_f32(cv7, bv1, av23, 1);

    cv8 = vmlaq_lane_f32(cv8, bv0, av45, 0);
    cv9 = vmlaq_lane_f32(cv9, bv1, av45, 0);
    cv10 = vmlaq_lane_f32(cv10, bv0, av45, 1);
    cv11 = vmlaq_lane_f32(cv11, bv1, av45, 1);

    a += MR;
    b += NR;
  }

  vst1q_f32(c, cv0);
  vst1q_f32(c + 4, cv1);
  vst1q_f32(c + ldc, cv2);
  vst1q_f32(c + ldc + 4, cv3);
  vst1q_f32(c + 2 * ldc, cv4);
  vst1q_f32(c + 2 * ldc + 4, cv5);
  vst1q_f32(c + 3 * ldc, cv6);
  vst1q_f32(c + 3 * ldc + 4, cv7);
  vst1q_f32(c + 4 * ldc, cv8);
  vst1q_f32(c + 4 * ldc + 4, cv9);
  vst1q_f32(c + 5 * ldc, cv10);
  vst1q_f32(c + 5 * ldc + 4, cv11);
}

void Gemm::AddDot8x12(int k, const float *a, const float *b, float *c,
                      int ldc) {
  const float *a_ptr, *b_ptr;
  a_ptr = a;
  b_ptr = b;
  int kc1 = k;
  int step = 4 * ldc;
  asm volatile(
      "dup      v5.4s,     wzr     \n\t"
      "dup      v6.4s,     wzr     \n\t"
      "dup      v7.4s,     wzr     \n\t"
      "dup      v8.4s,     wzr     \n\t"
      "dup      v9.4s,     wzr     \n\t"
      "dup      v10.4s,    wzr     \n\t"
      "dup      v11.4s,    wzr     \n\t"
      "dup      v12.4s,    wzr     \n\t"
      "dup      v13.4s,    wzr     \n\t"
      "dup      v14.4s,    wzr     \n\t"
      "dup      v15.4s,    wzr     \n\t"
      "dup      v16.4s,    wzr     \n\t"

      "dup      v17.4s,    wzr     \n\t"
      "dup      v18.4s,    wzr     \n\t"
      "dup      v19.4s,    wzr     \n\t"
      "dup      v20.4s,    wzr     \n\t"
      "dup      v21.4s,    wzr     \n\t"
      "dup      v22.4s,    wzr     \n\t"
      "dup      v23.4s,    wzr     \n\t"
      "dup      v24.4s,    wzr     \n\t"
      "dup      v25.4s,    wzr     \n\t"
      "dup      v26.4s,    wzr     \n\t"
      "dup      v27.4s,    wzr     \n\t"
      "dup      v28.4s,    wzr     \n\t"

      "subs       %[kc1], %[kc1], #1    \n\t"
      "blt        2f                    \n\t"
      "1:                               \n\t"

      "prfm     pldl1keep,         [%[a_ptr],   #32]  \n\t"
      "prfm     pldl1keep,         [%[b_ptr],   #48]  \n\t"

      "ld1      {v0.4s, v1.4s},         [%[a_ptr]],   #32   \n\t"
      "ld1      {v2.4s, v3.4s, v4.4s},  [%[b_ptr]],   #48   \n\t"

      "fmla     v5.4s,    v2.4s,   v0.s[0]       \n\t"
      "fmla     v6.4s,    v3.4s,   v0.s[0]       \n\t"
      "fmla     v7.4s,    v4.4s,   v0.s[0]       \n\t"
      "fmla     v8.4s,    v2.4s,   v0.s[1]       \n\t"
      "fmla     v9.4s,    v3.4s,   v0.s[1]       \n\t"
      "fmla     v10.4s,   v4.4s,   v0.s[1]       \n\t"
      "fmla     v11.4s,   v2.4s,   v0.s[2]       \n\t"
      "fmla     v12.4s,   v3.4s,   v0.s[2]       \n\t"
      "fmla     v13.4s,   v4.4s,   v0.s[2]       \n\t"
      "fmla     v14.4s,   v2.4s,   v0.s[3]       \n\t"
      "fmla     v15.4s,   v3.4s,   v0.s[3]       \n\t"
      "fmla     v16.4s,   v4.4s,   v0.s[3]       \n\t"

      "fmla     v17.4s,   v2.4s,   v1.s[0]       \n\t"
      "fmla     v18.4s,   v3.4s,   v1.s[0]       \n\t"
      "fmla     v19.4s,   v4.4s,   v1.s[0]       \n\t"
      "fmla     v20.4s,   v2.4s,   v1.s[1]       \n\t"
      "fmla     v21.4s,   v3.4s,   v1.s[1]       \n\t"
      "fmla     v22.4s,   v4.4s,   v1.s[1]       \n\t"
      "fmla     v23.4s,   v2.4s,   v1.s[2]       \n\t"
      "fmla     v24.4s,   v3.4s,   v1.s[2]       \n\t"
      "fmla     v25.4s,   v4.4s,   v1.s[2]       \n\t"
      "fmla     v26.4s,   v2.4s,   v1.s[3]       \n\t"
      "fmla     v27.4s,   v3.4s,   v1.s[3]       \n\t"
      "fmla     v28.4s,   v4.4s,   v1.s[3]       \n\t"

      "subs       %[kc1], %[kc1], #1      \n\t"
      "bge        1b                      \n\t"
      "2:                                 \n\t"

      "st1      {v5.4s,   v6.4s,  v7.4s},    [%[c]],   %[step]   \n\t"
      "st1      {v8.4s,   v9.4s,  v10.4s},   [%[c]],   %[step]   \n\t"
      "st1      {v11.4s,  v12.4s, v13.4s},   [%[c]],   %[step]   \n\t"
      "st1      {v14.4s,  v15.4s, v16.4s},   [%[c]],   %[step]   \n\t"
      "st1      {v17.4s,  v18.4s, v19.4s},   [%[c]],   %[step]   \n\t"
      "st1      {v20.4s,  v21.4s, v22.4s},   [%[c]],   %[step]   \n\t"
      "st1      {v23.4s,  v24.4s, v25.4s},   [%[c]],   %[step]   \n\t"
      "st1      {v26.4s,  v27.4s, v28.4s},   [%[c]],   %[step]   \n\t"
      :
      : [a_ptr] "r"(a_ptr), [b_ptr] "r"(b_ptr), [c] "r"(c), [kc1] "r"(kc1),
        [step] "r"(step)
      : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
        "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
        "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28");
}

void Gemm::AddDot6x16(int k, const float *a, const float *b, float *c,
                      int ldc) {
  const float *a_ptr, *b_ptr;
  a_ptr = a;
  b_ptr = b;
  int kc1 = k;
  int step = 4 * ldc;
  int step1 = 4 * 6;
  asm volatile(

      "dup      v6.4s,     wzr     \n\t"
      "dup      v7.4s,     wzr     \n\t"
      "dup      v8.4s,     wzr     \n\t"
      "dup      v9.4s,     wzr     \n\t"
      "dup      v10.4s,    wzr     \n\t"
      "dup      v11.4s,    wzr     \n\t"
      "dup      v12.4s,    wzr     \n\t"
      "dup      v13.4s,    wzr     \n\t"

      "dup      v14.4s,    wzr     \n\t"
      "dup      v15.4s,    wzr     \n\t"
      "dup      v16.4s,    wzr     \n\t"
      "dup      v17.4s,    wzr     \n\t"
      "dup      v18.4s,    wzr     \n\t"
      "dup      v19.4s,    wzr     \n\t"
      "dup      v20.4s,    wzr     \n\t"
      "dup      v21.4s,    wzr     \n\t"

      "dup      v22.4s,    wzr     \n\t"
      "dup      v23.4s,    wzr     \n\t"
      "dup      v24.4s,    wzr     \n\t"
      "dup      v25.4s,    wzr     \n\t"
      "dup      v26.4s,    wzr     \n\t"
      "dup      v27.4s,    wzr     \n\t"
      "dup      v28.4s,    wzr     \n\t"
      "dup      v29.4s,    wzr     \n\t"

      "subs       %[kc1], %[kc1], #1    \n\t"
      "blt        2f                    \n\t"
      "1:                               \n\t"

      "prfm   pldl1keep,  [%[a_ptr],  #24]  \n\t"
      "prfm   pldl1keep,  [%[b_ptr],  #64]  \n\t"

      "ld1      {v0.4s, v1.4s},  [%[a_ptr]],   %[step1]       \n\t"
      "ld1      {v2.4s, v3.4s, v4.4s, v5.4s},  [%[b_ptr]],    #64   \n\t"

      "fmla     v6.4s,    v2.4s,   v0.s[0]       \n\t"
      "fmla     v7.4s,    v3.4s,   v0.s[0]       \n\t"
      "fmla     v8.4s,    v4.4s,   v0.s[0]       \n\t"
      "fmla     v9.4s,    v5.4s,   v0.s[0]       \n\t"

      "fmla     v10.4s,   v2.4s,   v0.s[1]       \n\t"
      "fmla     v11.4s,   v3.4s,   v0.s[1]       \n\t"
      "fmla     v12.4s,   v4.4s,   v0.s[1]       \n\t"
      "fmla     v13.4s,   v5.4s,   v0.s[1]       \n\t"

      "fmla     v14.4s,   v2.4s,   v0.s[2]       \n\t"
      "fmla     v15.4s,   v3.4s,   v0.s[2]       \n\t"
      "fmla     v16.4s,   v4.4s,   v0.s[2]       \n\t"
      "fmla     v17.4s,   v5.4s,   v0.s[2]       \n\t"

      "fmla     v18.4s,   v2.4s,   v0.s[3]       \n\t"
      "fmla     v19.4s,   v3.4s,   v0.s[3]       \n\t"
      "fmla     v20.4s,   v4.4s,   v0.s[3]       \n\t"
      "fmla     v21.4s,   v5.4s,   v0.s[3]       \n\t"

      "fmla     v22.4s,   v2.4s,   v1.s[0]       \n\t"
      "fmla     v23.4s,   v3.4s,   v1.s[0]       \n\t"
      "fmla     v24.4s,   v4.4s,   v1.s[0]       \n\t"
      "fmla     v25.4s,   v5.4s,   v1.s[0]       \n\t"

      "fmla     v26.4s,   v2.4s,   v1.s[1]       \n\t"
      "fmla     v27.4s,   v3.4s,   v1.s[1]       \n\t"
      "fmla     v28.4s,   v4.4s,   v1.s[1]       \n\t"
      "fmla     v29.4s,   v5.4s,   v1.s[1]       \n\t"

      "subs       %[kc1], %[kc1], #1      \n\t"
      "bge        1b                      \n\t"
      "2:                                 \n\t"

      "st1      {v6.4s,  v7.4s,  v8.4s,  v9.4s},    [%[c]],   %[step]   \n\t"
      "st1      {v10.4s, v11.4s, v12.4s, v13.4s},   [%[c]],   %[step]   \n\t"
      "st1      {v14.4s, v15.4s, v16.4s, v17.4s},   [%[c]],   %[step]   \n\t"
      "st1      {v18.4s, v19.4s, v20.4s, v21.4s},   [%[c]],   %[step]   \n\t"
      "st1      {v22.4s, v23.4s, v24.4s, v25.4s},   [%[c]],   %[step]   \n\t"
      "st1      {v26.4s, v27.4s, v28.4s, v29.4s},   [%[c]],   %[step]   \n\t"
      :
      : [a_ptr] "r"(a_ptr), [b_ptr] "r"(b_ptr), [c] "r"(c), [kc1] "r"(kc1),
        [step] "r"(step), [step1] "r"(step1)
      : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
        "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
        "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29");
}

#else

void Gemm::AddDot4x4(int k, const float *a, const float *b, float *c, int ldc) {
  const float *a_ptr, *b_ptr;
  a_ptr = a;
  b_ptr = b;
  int kc1 = k / 4;
  int kc2 = k % 4;
  int step = 4 * ldc;
  asm volatile(
      "pld        [%[a_ptr]]          \n\t"
      "pld        [%[b_ptr]]          \n\t"
      "vmov.f32   q10,    #0.0        \n\t"
      "vmov.f32   q11,    #0.0        \n\t"
      "vmov.f32   q12,    #0.0        \n\t"
      "vmov.f32   q13,    #0.0        \n\t"

      "subs       %[kc1], %[kc1], #1  \n\t"
      "blt        end_kc1_%=          \n\t"
      "loop_kc1_%=:                   \n\t"
      "pld        [%[a_ptr], #64]     \n\t"
      "pld        [%[b_ptr], #64]     \n\t"
      "vld1.32    {q0, q1}, [%[a_ptr]]!   \n\t"
      "vld1.32    {q2, q3}, [%[b_ptr]]!   \n\t"
      "vmla.f32   q10, q2, d0[0]      \n\t"
      "vmla.f32   q11, q2, d0[1]      \n\t"
      "vmla.f32   q12, q2, d1[0]      \n\t"
      "vmla.f32   q13, q2, d1[1]      \n\t"
      "vmla.f32   q10, q3, d2[0]      \n\t"
      "vmla.f32   q11, q3, d2[1]      \n\t"
      "vmla.f32   q12, q3, d3[0]      \n\t"
      "vmla.f32   q13, q3, d3[1]      \n\t"
      "vld1.32    {q4, q5}, [%[a_ptr]]!   \n\t"
      "vld1.32    {q6, q7}, [%[b_ptr]]!   \n\t"
      "vmla.f32   q10, q6, d8[0]      \n\t"
      "vmla.f32   q11, q6, d8[1]      \n\t"
      "vmla.f32   q12, q6, d9[0]      \n\t"
      "vmla.f32   q13, q6, d9[1]      \n\t"
      "vmla.f32   q10, q7, d10[0]     \n\t"
      "vmla.f32   q11, q7, d10[1]     \n\t"
      "vmla.f32   q12, q7, d11[0]     \n\t"
      "vmla.f32   q13, q7, d11[1]     \n\t"
      "subs       %[kc1], %[kc1], #1  \n\t"
      "bge        loop_kc1_%=         \n\t"
      "end_kc1_%=:                    \n\t"

      "subs       %[kc2], %[kc2], #1  \n\t"
      "blt        end_kc2_%=          \n\t"
      "loop_kc2_%=:                   \n\t"
      "vld1.32    {q0}, [%[a_ptr]]!   \n\t"
      "vld1.32    {q1}, [%[b_ptr]]!   \n\t"
      "vmla.f32   q10, q1, d0[0]      \n\t"
      "vmla.f32   q11, q1, d0[1]      \n\t"
      "vmla.f32   q12, q1, d1[0]      \n\t"
      "vmla.f32   q13, q1, d1[1]      \n\t"
      "subs       %[kc2], %[kc2], #1  \n\t"
      "bge        loop_kc2_%=         \n\t"
      "end_kc2_%=:                    \n\t"

      "mov        r5,     %[c]        \n\t"
      "mov        r6,     %[step]     \n\t"
      "vst1.32    {q10}, [r5], r6     \n\t"
      "vst1.32    {q11}, [r5], r6     \n\t"
      "vst1.32    {q12}, [r5], r6     \n\t"
      "vst1.32    {q13}, [r5]         \n\t"
      :
      : [a_ptr] "r"(a_ptr), [b_ptr] "r"(b_ptr), [c] "r"(c), [kc1] "r"(kc1),
        [kc2] "r"(kc2), [step] "r"(step)
      : "memory", "r5", "r6", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
        "q10", "q11", "q12", "q13");
}

void Gemm::AddDot4x8(int k, const float *a, const float *b, float *c, int ldc) {
  const float *a_ptr, *b_ptr;
  a_ptr = a;
  b_ptr = b;
  int kc1 = k / 4;
  int kc2 = k % 4;
  int step = 4 * ldc;
  asm volatile(
      "pld        [%[a_ptr]]          \n\t"
      "pld        [%[b_ptr]]          \n\t"

      "vmov.f32   q8,     #0.0        \n\t"
      "vmov.f32   q9,     #0.0        \n\t"
      "vmov.f32   q10,    #0.0        \n\t"
      "vmov.f32   q11,    #0.0        \n\t"
      "vmov.f32   q12,    #0.0        \n\t"
      "vmov.f32   q13,    #0.0        \n\t"
      "vmov.f32   q14,    #0.0        \n\t"
      "vmov.f32   q15,    #0.0        \n\t"

      "subs       %[kc1], %[kc1], #1  \n\t"
      "blt        end_kc1_%=          \n\t"
      "loop_kc1_%=:                   \n\t"

      "pld        [%[a_ptr], #64]     \n\t"
      "pld        [%[b_ptr], #64]     \n\t"

      "vld1.32    {q0, q1}, [%[a_ptr]]!   \n\t"
      "vld1.32    {q2, q3}, [%[b_ptr]]!   \n\t"
      "vld1.32    {q4, q5}, [%[b_ptr]]!   \n\t"

      "vmla.f32   q8,   q2,   d0[0]      \n\t"
      "vmla.f32   q9,   q3,   d0[0]      \n\t"
      "vmla.f32   q10,  q2,   d0[1]      \n\t"
      "vmla.f32   q11,  q3,   d0[1]      \n\t"
      "vmla.f32   q12,  q2,   d1[0]      \n\t"
      "vmla.f32   q13,  q3,   d1[0]      \n\t"
      "vmla.f32   q14,  q2,   d1[1]      \n\t"
      "vmla.f32   q15,  q3,   d1[1]      \n\t"

      "vmla.f32   q8,   q4,   d2[0]      \n\t"
      "vmla.f32   q9,   q5,   d2[0]      \n\t"
      "vmla.f32   q10,  q4,   d2[1]      \n\t"
      "vmla.f32   q11,  q5,   d2[1]      \n\t"
      "vmla.f32   q12,  q4,   d3[0]      \n\t"
      "vmla.f32   q13,  q5,   d3[0]      \n\t"
      "vmla.f32   q14,  q4,   d3[1]      \n\t"
      "vmla.f32   q15,  q5,   d3[1]      \n\t"

      "pld        [%[b_ptr], #64]     \n\t"

      "vld1.32    {q0, q1}, [%[a_ptr]]!   \n\t"
      "vld1.32    {q2, q3}, [%[b_ptr]]!   \n\t"
      "vld1.32    {q4, q5}, [%[b_ptr]]!   \n\t"

      "vmla.f32   q8,   q2,   d0[0]      \n\t"
      "vmla.f32   q9,   q3,   d0[0]      \n\t"
      "vmla.f32   q10,  q2,   d0[1]      \n\t"
      "vmla.f32   q11,  q3,   d0[1]      \n\t"
      "vmla.f32   q12,  q2,   d1[0]      \n\t"
      "vmla.f32   q13,  q3,   d1[0]      \n\t"
      "vmla.f32   q14,  q2,   d1[1]      \n\t"
      "vmla.f32   q15,  q3,   d1[1]      \n\t"

      "vmla.f32   q8,   q4,   d2[0]      \n\t"
      "vmla.f32   q9,   q5,   d2[0]      \n\t"
      "vmla.f32   q10,  q4,   d2[1]      \n\t"
      "vmla.f32   q11,  q5,   d2[1]      \n\t"
      "vmla.f32   q12,  q4,   d3[0]      \n\t"
      "vmla.f32   q13,  q5,   d3[0]      \n\t"
      "vmla.f32   q14,  q4,   d3[1]      \n\t"
      "vmla.f32   q15,  q5,   d3[1]      \n\t"

      "subs       %[kc1], %[kc1], #1  \n\t"
      "bge        loop_kc1_%=         \n\t"
      "end_kc1_%=:                    \n\t"

      "subs       %[kc2], %[kc2], #1  \n\t"
      "blt        end_kc2_%=          \n\t"
      "loop_kc2_%=:                   \n\t"
      "vld1.32    {q0},     [%[a_ptr]]!   \n\t"
      "vld1.32    {q2, q3}, [%[b_ptr]]!   \n\t"
      "vmla.f32   q8,   q2,   d0[0]      \n\t"
      "vmla.f32   q9,   q3,   d0[0]      \n\t"
      "vmla.f32   q10,  q2,   d0[1]      \n\t"
      "vmla.f32   q11,  q3,   d0[1]      \n\t"
      "vmla.f32   q12,  q2,   d1[0]      \n\t"
      "vmla.f32   q13,  q3,   d1[0]      \n\t"
      "vmla.f32   q14,  q2,   d1[1]      \n\t"
      "vmla.f32   q15,  q3,   d1[1]      \n\t"
      "subs       %[kc2], %[kc2], #1  \n\t"
      "bge        loop_kc2_%=         \n\t"
      "end_kc2_%=:                    \n\t"

      "mov        r5,     %[c]        \n\t"
      "mov        r6,     %[step]     \n\t"
      "vst1.32    {q8, q9},   [r5], r6     \n\t"
      "vst1.32    {q10, q11}, [r5], r6     \n\t"
      "vst1.32    {q12, q13}, [r5], r6     \n\t"
      "vst1.32    {q14, q15}, [r5]         \n\t"
      :
      : [a_ptr] "r"(a_ptr), [b_ptr] "r"(b_ptr), [c] "r"(c), [kc1] "r"(kc1),
        [kc2] "r"(kc2), [step] "r"(step)
      : "memory", "r5", "r6", "q0", "q1", "q2", "q3", "q4", "q5", "q8", "q9",
        "q10", "q11", "q12", "q13", "q14", "q15");
}

void Gemm::AddDot6x8(int k, const float *a, const float *b, float *c, int ldc) {
  const float *a_ptr, *b_ptr;
  a_ptr = a;
  b_ptr = b;
  int kc1 = k / 8;
  int kc2 = k % 8;
  int step = sizeof(float) * ldc;
  asm volatile(
      "pld        [%[a_ptr]]            \n\t"
      "pld        [%[a_ptr],  #64]      \n\t"
      "pld        [%[b_ptr]]            \n\t"
      "pld        [%[b_ptr],  #64]      \n\t"

      "vmov.f32   q4,     #0.0          \n\t"
      "vmov.f32   q5,     #0.0          \n\t"
      "vmov.f32   q6,     #0.0          \n\t"
      "vmov.f32   q7,     #0.0          \n\t"
      "vmov.f32   q8,     #0.0          \n\t"
      "vmov.f32   q9,     #0.0          \n\t"
      "vmov.f32   q10,    #0.0          \n\t"
      "vmov.f32   q11,    #0.0          \n\t"
      "vmov.f32   q12,    #0.0          \n\t"
      "vmov.f32   q13,    #0.0          \n\t"
      "vmov.f32   q14,    #0.0          \n\t"
      "vmov.f32   q15,    #0.0          \n\t"

      "subs       %[kc1], %[kc1], #1    \n\t"
      "blt        2f                    \n\t"
      "1:                               \n\t"

      "pld        [%[a_ptr], #128]       \n\t"
      "pld        [%[b_ptr], #128]       \n\t"

      "vld1.32    {d0-d2},  [%[a_ptr]]!   \n\t"
      "vld1.32    {q2, q3}, [%[b_ptr]]!   \n\t"

      "vmla.f32   q4,   q2,   d0[0]       \n\t"
      "vmla.f32   q5,   q3,   d0[0]       \n\t"
      "vmla.f32   q6,   q2,   d0[1]       \n\t"
      "vmla.f32   q7,   q3,   d0[1]       \n\t"
      "vmla.f32   q8,   q2,   d1[0]       \n\t"
      "vmla.f32   q9,   q3,   d1[0]       \n\t"
      "vmla.f32   q10,  q2,   d1[1]       \n\t"
      "vmla.f32   q11,  q3,   d1[1]       \n\t"
      "vmla.f32   q12,  q2,   d2[0]       \n\t"
      "vmla.f32   q13,  q3,   d2[0]       \n\t"
      "vmla.f32   q14,  q2,   d2[1]       \n\t"
      "vmla.f32   q15,  q3,   d2[1]       \n\t"

      "vld1.32    {d0-d2},  [%[a_ptr]]!   \n\t"
      "vld1.32    {q2, q3}, [%[b_ptr]]!   \n\t"

      "vmla.f32   q4,   q2,   d0[0]       \n\t"
      "vmla.f32   q5,   q3,   d0[0]       \n\t"
      "vmla.f32   q6,   q2,   d0[1]       \n\t"
      "vmla.f32   q7,   q3,   d0[1]       \n\t"
      "vmla.f32   q8,   q2,   d1[0]       \n\t"
      "vmla.f32   q9,   q3,   d1[0]       \n\t"
      "vmla.f32   q10,  q2,   d1[1]       \n\t"
      "vmla.f32   q11,  q3,   d1[1]       \n\t"
      "vmla.f32   q12,  q2,   d2[0]       \n\t"
      "vmla.f32   q13,  q3,   d2[0]       \n\t"
      "vmla.f32   q14,  q2,   d2[1]       \n\t"
      "vmla.f32   q15,  q3,   d2[1]       \n\t"

      "pld        [%[a_ptr], #128]       \n\t"
      "pld        [%[b_ptr], #128]       \n\t"

      "vld1.32    {d0-d2},  [%[a_ptr]]!   \n\t"
      "vld1.32    {q2, q3}, [%[b_ptr]]!   \n\t"

      "vmla.f32   q4,   q2,   d0[0]       \n\t"
      "vmla.f32   q5,   q3,   d0[0]       \n\t"
      "vmla.f32   q6,   q2,   d0[1]       \n\t"
      "vmla.f32   q7,   q3,   d0[1]       \n\t"
      "vmla.f32   q8,   q2,   d1[0]       \n\t"
      "vmla.f32   q9,   q3,   d1[0]       \n\t"
      "vmla.f32   q10,  q2,   d1[1]       \n\t"
      "vmla.f32   q11,  q3,   d1[1]       \n\t"
      "vmla.f32   q12,  q2,   d2[0]       \n\t"
      "vmla.f32   q13,  q3,   d2[0]       \n\t"
      "vmla.f32   q14,  q2,   d2[1]       \n\t"
      "vmla.f32   q15,  q3,   d2[1]       \n\t"

      "vld1.32    {d0-d2},  [%[a_ptr]]!   \n\t"
      "vld1.32    {q2, q3}, [%[b_ptr]]!   \n\t"

      "vmla.f32   q4,   q2,   d0[0]       \n\t"
      "vmla.f32   q5,   q3,   d0[0]       \n\t"
      "vmla.f32   q6,   q2,   d0[1]       \n\t"
      "vmla.f32   q7,   q3,   d0[1]       \n\t"
      "vmla.f32   q8,   q2,   d1[0]       \n\t"
      "vmla.f32   q9,   q3,   d1[0]       \n\t"
      "vmla.f32   q10,  q2,   d1[1]       \n\t"
      "vmla.f32   q11,  q3,   d1[1]       \n\t"
      "vmla.f32   q12,  q2,   d2[0]       \n\t"
      "vmla.f32   q13,  q3,   d2[0]       \n\t"
      "vmla.f32   q14,  q2,   d2[1]       \n\t"
      "vmla.f32   q15,  q3,   d2[1]       \n\t"

      "pld        [%[a_ptr], #128]       \n\t"
      "pld        [%[b_ptr], #128]       \n\t"

      "vld1.32    {d0-d2},  [%[a_ptr]]!   \n\t"
      "vld1.32    {q2, q3}, [%[b_ptr]]!   \n\t"

      "vmla.f32   q4,   q2,   d0[0]       \n\t"
      "vmla.f32   q5,   q3,   d0[0]       \n\t"
      "vmla.f32   q6,   q2,   d0[1]       \n\t"
      "vmla.f32   q7,   q3,   d0[1]       \n\t"
      "vmla.f32   q8,   q2,   d1[0]       \n\t"
      "vmla.f32   q9,   q3,   d1[0]       \n\t"
      "vmla.f32   q10,  q2,   d1[1]       \n\t"
      "vmla.f32   q11,  q3,   d1[1]       \n\t"
      "vmla.f32   q12,  q2,   d2[0]       \n\t"
      "vmla.f32   q13,  q3,   d2[0]       \n\t"
      "vmla.f32   q14,  q2,   d2[1]       \n\t"
      "vmla.f32   q15,  q3,   d2[1]       \n\t"

      "vld1.32    {d0-d2},  [%[a_ptr]]!   \n\t"
      "vld1.32    {q2, q3}, [%[b_ptr]]!   \n\t"

      "vmla.f32   q4,   q2,   d0[0]       \n\t"
      "vmla.f32   q5,   q3,   d0[0]       \n\t"
      "vmla.f32   q6,   q2,   d0[1]       \n\t"
      "vmla.f32   q7,   q3,   d0[1]       \n\t"
      "vmla.f32   q8,   q2,   d1[0]       \n\t"
      "vmla.f32   q9,   q3,   d1[0]       \n\t"
      "vmla.f32   q10,  q2,   d1[1]       \n\t"
      "vmla.f32   q11,  q3,   d1[1]       \n\t"
      "vmla.f32   q12,  q2,   d2[0]       \n\t"
      "vmla.f32   q13,  q3,   d2[0]       \n\t"
      "vmla.f32   q14,  q2,   d2[1]       \n\t"
      "vmla.f32   q15,  q3,   d2[1]       \n\t"

      "pld        [%[a_ptr], #128]       \n\t"
      "pld        [%[b_ptr], #128]       \n\t"

      "vld1.32    {d0-d2},  [%[a_ptr]]!   \n\t"
      "vld1.32    {q2, q3}, [%[b_ptr]]!   \n\t"

      "vmla.f32   q4,   q2,   d0[0]       \n\t"
      "vmla.f32   q5,   q3,   d0[0]       \n\t"
      "vmla.f32   q6,   q2,   d0[1]       \n\t"
      "vmla.f32   q7,   q3,   d0[1]       \n\t"
      "vmla.f32   q8,   q2,   d1[0]       \n\t"
      "vmla.f32   q9,   q3,   d1[0]       \n\t"
      "vmla.f32   q10,  q2,   d1[1]       \n\t"
      "vmla.f32   q11,  q3,   d1[1]       \n\t"
      "vmla.f32   q12,  q2,   d2[0]       \n\t"
      "vmla.f32   q13,  q3,   d2[0]       \n\t"
      "vmla.f32   q14,  q2,   d2[1]       \n\t"
      "vmla.f32   q15,  q3,   d2[1]       \n\t"

      "vld1.32    {d0-d2},  [%[a_ptr]]!   \n\t"
      "vld1.32    {q2, q3}, [%[b_ptr]]!   \n\t"

      "vmla.f32   q4,   q2,   d0[0]       \n\t"
      "vmla.f32   q5,   q3,   d0[0]       \n\t"
      "vmla.f32   q6,   q2,   d0[1]       \n\t"
      "vmla.f32   q7,   q3,   d0[1]       \n\t"
      "vmla.f32   q8,   q2,   d1[0]       \n\t"
      "vmla.f32   q9,   q3,   d1[0]       \n\t"
      "vmla.f32   q10,  q2,   d1[1]       \n\t"
      "vmla.f32   q11,  q3,   d1[1]       \n\t"
      "vmla.f32   q12,  q2,   d2[0]       \n\t"
      "vmla.f32   q13,  q3,   d2[0]       \n\t"
      "vmla.f32   q14,  q2,   d2[1]       \n\t"
      "vmla.f32   q15,  q3,   d2[1]       \n\t"

      "subs       %[kc1], %[kc1], #1      \n\t"
      "bge        1b                      \n\t"
      "2:                                 \n\t"

      "subs       %[kc2], %[kc2], #1      \n\t"
      "blt        4f                      \n\t"
      "3:                                 \n\t"

      "vld1.32    {d0-d2},  [%[a_ptr]]!   \n\t"
      "vld1.32    {q2, q3}, [%[b_ptr]]!   \n\t"

      "vmla.f32   q4,   q2,   d0[0]       \n\t"
      "vmla.f32   q5,   q3,   d0[0]       \n\t"
      "vmla.f32   q6,   q2,   d0[1]       \n\t"
      "vmla.f32   q7,   q3,   d0[1]       \n\t"
      "vmla.f32   q8,   q2,   d1[0]       \n\t"
      "vmla.f32   q9,   q3,   d1[0]       \n\t"
      "vmla.f32   q10,  q2,   d1[1]       \n\t"
      "vmla.f32   q11,  q3,   d1[1]       \n\t"
      "vmla.f32   q12,  q2,   d2[0]       \n\t"
      "vmla.f32   q13,  q3,   d2[0]       \n\t"
      "vmla.f32   q14,  q2,   d2[1]       \n\t"
      "vmla.f32   q15,  q3,   d2[1]       \n\t"

      "subs       %[kc2], %[kc2], #1      \n\t"
      "bge        3b                      \n\t"
      "4:                                 \n\t"

      "mov        r5,     %[c]            \n\t"
      "mov        r6,     %[step]         \n\t"
      "vst1.32    {q4, q5},   [r5], r6    \n\t"
      "vst1.32    {q6, q7},   [r5], r6    \n\t"
      "vst1.32    {q8, q9},   [r5], r6    \n\t"
      "vst1.32    {q10, q11}, [r5], r6    \n\t"
      "vst1.32    {q12, q13}, [r5], r6    \n\t"
      "vst1.32    {q14, q15}, [r5]        \n\t"

      :
      : [a_ptr] "r"(a_ptr), [b_ptr] "r"(b_ptr), [c] "r"(c), [kc1] "r"(kc1),
        [kc2] "r"(kc2), [step] "r"(step)
      : "cc", "memory", "r5", "r6", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
        "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}

#endif  // __aarch64__
#endif  // __ARM_NEON

#if __ARM_NEON
#if __aarch64__

// 分块矩阵乘法结果回写
// C = A * B
void Gemm::WriteBasic(int mc, int nc, float *c, float *C, int ldc) {
  int nc1 = nc / 4;
  int _nc1 = nc % 4;

  float *c_ptr, *C_ptr;
  float32x4_t cv;
  for (int i = 0; i < mc; ++i) {
    c_ptr = c + i * NC;
    C_ptr = C + i * ldc;
    for (int j = 0; j < nc1; ++j) {
      cv = vld1q_f32(c_ptr);
      vst1q_f32(C_ptr, cv);
      c_ptr += 4;
      C_ptr += 4;
    }
    if (_nc1 != 0) {
      cv = vld1q_f32(c_ptr);
      if (_nc1 >= 1) {
        vst1q_lane_f32(C_ptr, cv, 0);
        C_ptr++;
      }
      if (_nc1 >= 2) {
        vst1q_lane_f32(C_ptr, cv, 1);
        C_ptr++;
      }
      if (_nc1 >= 3) {
        vst1q_lane_f32(C_ptr, cv, 2);
      }
    }
  }
}

// C = alpha * A * B + beta * C
void Gemm::WriteWithAlphaBeta(int mc, int nc, float *c, float *C, int ldc) {}

// C = A * B + C
void Gemm::WriteWithAdd(int mc, int nc, float *c, float *C, int ldc) {
  int nc1 = nc / 4;
  int _nc1 = nc % 4;

  float *c_ptr, *C_ptr;
  float32x4_t cv;
  float32x4_t cv1;
  for (int i = 0; i < mc; ++i) {
    c_ptr = c + i * NC;
    C_ptr = C + i * ldc;
    for (int j = 0; j < nc1; ++j) {
      cv = vld1q_f32(c_ptr);
      cv1 = vld1q_f32(C_ptr);
      cv = vaddq_f32(cv, cv1);
      vst1q_f32(C_ptr, cv);
      c_ptr += 4;
      C_ptr += 4;
    }
    if (_nc1 != 0) {
      cv = vld1q_f32(c_ptr);
      cv1 = vld1q_f32(C_ptr);
      cv = vaddq_f32(cv, cv1);
      if (_nc1 >= 1) {
        vst1q_lane_f32(C_ptr, cv, 0);
        C_ptr++;
      }
      if (_nc1 >= 2) {
        vst1q_lane_f32(C_ptr, cv, 1);
        C_ptr++;
      }
      if (_nc1 >= 3) {
        vst1q_lane_f32(C_ptr, cv, 2);
      }
    }
  }
}
// C = A * B + bias
void Gemm::WriteWithAddV1(int mc, int nc, float *c, float *C, int ldc,
                          float *bias) {
  int nc1 = nc / 4;
  int _nc1 = nc % 4;

  float *c_ptr, *C_ptr;
  float32x4_t cv;
  float32x4_t biasv;
  for (int i = 0; i < mc; ++i) {
    c_ptr = c + i * NC;
    C_ptr = C + i * ldc;
    biasv = vld1q_dup_f32(bias + i);
    for (int j = 0; j < nc1; ++j) {
      cv = vld1q_f32(c_ptr);
      cv = vaddq_f32(cv, biasv);
      vst1q_f32(C_ptr, cv);
      c_ptr += 4;
      C_ptr += 4;
    }
    if (_nc1 != 0) {
      cv = vld1q_f32(c_ptr);
      cv = vaddq_f32(cv, biasv);
      if (_nc1 >= 1) {
        vst1q_lane_f32(C_ptr, cv, 0);
        C_ptr++;
      }
      if (_nc1 >= 2) {
        vst1q_lane_f32(C_ptr, cv, 1);
        C_ptr++;
      }
      if (_nc1 >= 3) {
        vst1q_lane_f32(C_ptr, cv, 2);
        C_ptr++;
      }
    }
  }
}

// C = A * B + C, relu(C)
void Gemm::WriteWithAddRelu(int mc, int nc, float *c, float *C, int ldc) {
  int nc1 = nc / 4;
  int _nc1 = nc % 4;

  float *c_ptr, *C_ptr;
  float32x4_t cv;
  float32x4_t cv1;
  float32x4_t zero = vdupq_n_f32(0.0);
  for (int i = 0; i < mc; ++i) {
    c_ptr = c + i * NC;
    C_ptr = C + i * ldc;
    for (int j = 0; j < nc1; ++j) {
      cv = vld1q_f32(c_ptr);
      cv1 = vld1q_f32(C_ptr);
      cv = vaddq_f32(cv, cv1);
      cv = vmaxq_f32(cv, zero);
      vst1q_f32(C_ptr, cv);
      c_ptr += 4;
      C_ptr += 4;
    }
    if (_nc1 != 0) {
      cv = vld1q_f32(c_ptr);
      cv1 = vld1q_f32(C_ptr);
      cv = vaddq_f32(cv, cv1);
      cv = vmaxq_f32(cv, zero);
      if (_nc1 >= 1) {
        vst1q_lane_f32(C_ptr, cv, 0);
        C_ptr++;
      }
      if (_nc1 >= 2) {
        vst1q_lane_f32(C_ptr, cv, 1);
        C_ptr++;
      }
      if (_nc1 >= 3) {
        vst1q_lane_f32(C_ptr, cv, 2);
      }
    }
  }
}

// C = A * B + bias, relu(C)
void Gemm::WriteWithAddReluV1(int mc, int nc, float *c, float *C, int ldc,
                              float *bias) {
  int nc1 = nc / 4;
  int _nc1 = nc % 4;

  float *c_ptr, *C_ptr;
  float32x4_t cv;
  float32x4_t biasv;
  float32x4_t zero = vdupq_n_f32(0.0);
  for (int i = 0; i < mc; ++i) {
    c_ptr = c + i * NC;
    C_ptr = C + i * ldc;
    biasv = vld1q_dup_f32(bias + i);
    for (int j = 0; j < nc1; ++j) {
      cv = vld1q_f32(c_ptr);
      cv = vaddq_f32(cv, biasv);
      cv = vmaxq_f32(cv, zero);
      vst1q_f32(C_ptr, cv);
      c_ptr += 4;
      C_ptr += 4;
    }
    if (_nc1 != 0) {
      cv = vld1q_f32(c_ptr);
      cv = vaddq_f32(cv, biasv);
      cv = vmaxq_f32(cv, zero);
      if (_nc1 >= 1) {
        vst1q_lane_f32(C_ptr, cv, 0);
        C_ptr++;
      }
      if (_nc1 >= 2) {
        vst1q_lane_f32(C_ptr, cv, 1);
        C_ptr++;
      }
      if (_nc1 >= 3) {
        vst1q_lane_f32(C_ptr, cv, 2);
        C_ptr++;
      }
    }
  }
}

// C = A * B + C,prelu(C)
void Gemm::WriteWithAddPRelu(int mc, int nc, float *c, float *C, int ldc,
                             float *p, std::string mode, float *bias,
                             float *bias1) {
  int nc1 = nc / 4;
  int _nc1 = nc % 4;

  float *c_ptr, *C_ptr;
  float32x4_t cv;
  float32x4_t cv1;
  float32x4_t biasv;
  float32x4_t biasv1;
  float32x4_t zero = vdupq_n_f32(0.0);
  float32x4_t pv;
  float *ptr = p;
  for (int i = 0; i < mc; ++i) {
    c_ptr = c + i * NC;
    C_ptr = C + i * ldc;
    biasv = vld1q_dup_f32(bias + i);
    if (bias1 == nullptr) {
      biasv1 = zero;
    } else {
      biasv1 = vld1q_dup_f32(bias1 + i);
    }

    for (int j = 0; j < nc1; ++j) {
      cv = vld1q_f32(c_ptr);
      cv = vaddq_f32(cv, biasv);
      cv = vaddq_f32(cv, biasv1);
      cv = vmaxq_f32(cv, zero);
      cv1 = vminq_f32(cv, zero);
      if (mode == "channel") {
        cv1 = vmulq_n_f32(cv1, ptr[i]);
      } else if (mode == "element") {
        pv = vld1q_f32(ptr);
        cv1 = vmulq_f32(cv1, pv);
        ptr = ptr + 4;
      } else {
        cv1 = vmulq_n_f32(cv1, ptr[0]);
      }
      cv = vaddq_f32(cv, cv1);
      vst1q_f32(C_ptr, cv);
      c_ptr += 4;
      C_ptr += 4;
    }
    if (_nc1 != 0) {
      cv = vld1q_f32(c_ptr);
      cv = vaddq_f32(cv, biasv);
      cv = vaddq_f32(cv, biasv1);
      cv = vmaxq_f32(cv, zero);
      cv1 = vminq_f32(cv, zero);
      if (mode == "channel") {
        cv1 = vmulq_n_f32(cv1, ptr[i]);
      } else if (mode == "element") {
        pv = vld1q_f32(ptr);
        cv1 = vmulq_f32(cv1, pv);
        ptr = ptr + 4;
      } else {
        cv1 = vmulq_n_f32(cv1, ptr[0]);
      }
      cv = vaddq_f32(cv, cv1);
      if (_nc1 >= 1) {
        vst1q_lane_f32(C_ptr, cv, 0);
        C_ptr++;
      }
      if (_nc1 >= 2) {
        vst1q_lane_f32(C_ptr, cv, 1);
        C_ptr++;
      }
      if (_nc1 >= 3) {
        vst1q_lane_f32(C_ptr, cv, 2);
        C_ptr++;
      }
    }
  }
}

// C = A * B, batchnorm(C)
void Gemm::WriteWithBn(int mc, int nc, float *c, float *C, int ldc,
                       float *new_scale, float *new_bias) {
  int nc1 = nc / 4;
  int _nc1 = nc % 4;

  float *c_ptr, *C_ptr;
  float32x4_t cv;
  float32x4_t cv1;
  float32x4_t bias;
  float32x2_t scale;
  for (int i = 0; i < mc; ++i) {
    c_ptr = c + i * NC;
    C_ptr = C + i * ldc;
    bias = vld1q_dup_f32(new_bias);
    scale = vld1_dup_f32(new_scale);
    new_bias++;
    new_scale++;
    float scale0 = vget_lane_f32(scale, 0);
    for (int j = 0; j < nc1; ++j) {
      cv = vld1q_f32(c_ptr);
      cv = vmlaq_n_f32(bias, cv, scale0);
      vst1q_f32(C_ptr, cv);
      c_ptr += 4;
      C_ptr += 4;
    }
    if (_nc1 != 0) {
      cv = vld1q_f32(c_ptr);
      cv = vmlaq_n_f32(bias, cv, scale0);
      if (_nc1 >= 1) {
        vst1q_lane_f32(C_ptr, cv, 0);
        C_ptr++;
      }
      if (_nc1 >= 2) {
        vst1q_lane_f32(C_ptr, cv, 1);
        C_ptr++;
      }
      if (_nc1 >= 3) {
        vst1q_lane_f32(C_ptr, cv, 2);
        C_ptr++;
      }
    }
  }
}

// C = A * B, batchnorm(C), relu(C)
void Gemm::WriteWithBnRelu(int mc, int nc, float *c, float *C, int ldc,
                           float *new_scale, float *new_bias) {
  int nc1 = nc / 4;
  int _nc1 = nc % 4;

  float *c_ptr, *C_ptr;
  float32x4_t cv;
  float32x4_t bias;
  float32x2_t scale;
  float32x4_t zero = vdupq_n_f32(0.0);
  for (int i = 0; i < mc; ++i) {
    c_ptr = c + i * NC;
    C_ptr = C + i * ldc;
    bias = vld1q_dup_f32(new_bias);
    scale = vld1_dup_f32(new_scale);
    new_bias++;
    new_scale++;
    float scale0 = vget_lane_f32(scale, 0);
    for (int j = 0; j < nc1; ++j) {
      cv = vld1q_f32(c_ptr);
      cv = vmlaq_n_f32(bias, cv, scale0);
      cv = vmaxq_f32(cv, zero);
      vst1q_f32(C_ptr, cv);
      c_ptr += 4;
      C_ptr += 4;
    }
    if (_nc1 != 0) {
      cv = vld1q_f32(c_ptr);
      cv = vmlaq_n_f32(bias, cv, scale0);
      cv = vmaxq_f32(cv, zero);
      if (_nc1 >= 1) {
        vst1q_lane_f32(C_ptr, cv, 0);
        C_ptr++;
      }
      if (_nc1 >= 2) {
        vst1q_lane_f32(C_ptr, cv, 1);
        C_ptr++;
      }
      if (_nc1 >= 3) {
        vst1q_lane_f32(C_ptr, cv, 2);
      }
    }
  }
}

// C = A * B, batchnorm(C),C = C + bias; relu(C)
void Gemm::WriteWithBnAddRelu(int mc, int nc, float *c, float *C, int ldc,
                              float *new_scale, float *new_bias, float *bias) {
  int nc1 = nc / 4;
  int _nc1 = nc % 4;

  float *c_ptr, *C_ptr, *bias_ptr;
  float32x4_t cv;
  float32x4_t nbias;
  float32x2_t scale;
  float32x4_t biasv;
  float32x4_t zero = vdupq_n_f32(0.0);
  for (int i = 0; i < mc; ++i) {
    c_ptr = c + i * NC;
    C_ptr = C + i * ldc;
    bias_ptr = bias + i * ldc;
    nbias = vld1q_dup_f32(new_bias);
    scale = vld1_dup_f32(new_scale);
    new_bias++;
    new_scale++;
    float scale0 = vget_lane_f32(scale, 0);
    for (int j = 0; j < nc1; ++j) {
      cv = vld1q_f32(c_ptr);
      biasv = vld1q_f32(bias_ptr);
      cv = vmlaq_n_f32(nbias, cv, scale0);
      cv = vaddq_f32(cv, biasv);
      cv = vmaxq_f32(cv, zero);
      vst1q_f32(C_ptr, cv);
      c_ptr += 4;
      C_ptr += 4;
      bias_ptr += 4;
    }
    if (_nc1 != 0) {
      cv = vld1q_f32(c_ptr);
      biasv = vld1q_f32(bias_ptr);
      cv = vmlaq_n_f32(nbias, cv, scale0);
      cv = vaddq_f32(cv, biasv);
      cv = vmaxq_f32(cv, zero);
      if (_nc1 >= 1) {
        vst1q_lane_f32(C_ptr, cv, 0);
        C_ptr++;
      }
      if (_nc1 >= 2) {
        vst1q_lane_f32(C_ptr, cv, 1);
        C_ptr++;
      }
      if (_nc1 >= 3) {
        vst1q_lane_f32(C_ptr, cv, 2);
      }
    }
  }
}

#else

void Gemm::VectorKernel(int m, int n, int k, float alpha, const float *A,
                        int lda, const float *B, int ldb, float beta, float *C,
                        int ldc, bool relu) {
  float *bufferC = static_cast<float *>(memory::Alloc(sizeof(float) * n));

  const float *a0, *b0, *b1, *b2, *b3;
  float *c0, *C0;

  int volatile kc1 = k / 4;
  int volatile kc2 = k % 4;
  int volatile nc1 = n / 16;
  int _nc1 = n % 16;
  int volatile nc2 = _nc1 / 4;
  int volatile nc3 = _nc1 % 4;
  for (int i = 0; i < kc1; i++) {
    a0 = A + i * 4;
    b0 = B + i * 4 * ldb;
    b1 = b0 + ldb;
    b2 = b1 + ldb;
    b3 = b2 + ldb;
    c0 = bufferC;
    asm volatile(
        "pld        [%[a0], #16]          \n\t"
        "vld1.32    {q0}, [%[a0]]         \n\t"

        "subs       %[nc1], %[nc1], #1    \n\t"
        "blt        end_nc1_%=            \n\t"
        "loop_nc1_%=:                     \n\t"

        "cmp        %[i],       #0        \n\t"
        "beq        i_eq0_%=              \n\t"
        "bne        i_ne0_%=              \n\t"

        "i_eq0_%=:                        \n\t"
        "vmov.f32   q10,    #0.0          \n\t"
        "vmov.f32   q11,    #0.0          \n\t"
        "vmov.f32   q12,    #0.0          \n\t"
        "vmov.f32   q13,    #0.0          \n\t"
        "b          gemm_nc1_%=           \n\t"

        "i_ne0_%=:                        \n\t"
        "pld        [%[c0], #64]          \n\t"
        "vld1.32    {q10, q11}, [%[c0]]!  \n\t"
        "vld1.32    {q12, q13}, [%[c0]]   \n\t"
        "sub        %[c0], %[c0], #32     \n\t"

        "gemm_nc1_%=:                     \n\t"
        "pld        [%[b0], #64]          \n\t"
        "vld1.32    {q2, q3}, [%[b0]]!    \n\t"
        "vld1.32    {q4, q5}, [%[b0]]!    \n\t"
        "vmla.f32   q10, q2, d0[0]        \n\t"
        "vmla.f32   q11, q3, d0[0]        \n\t"
        "vmla.f32   q12, q4, d0[0]        \n\t"
        "vmla.f32   q13, q5, d0[0]        \n\t"

        "pld        [%[b1], #64]          \n\t"
        "vld1.32    {q2, q3}, [%[b1]]!    \n\t"
        "vld1.32    {q4, q5}, [%[b1]]!    \n\t"
        "vmla.f32   q10, q2, d0[1]        \n\t"
        "vmla.f32   q11, q3, d0[1]        \n\t"
        "vmla.f32   q12, q4, d0[1]        \n\t"
        "vmla.f32   q13, q5, d0[1]        \n\t"

        "pld        [%[b2], #64]          \n\t"
        "vld1.32    {q2, q3}, [%[b2]]!    \n\t"
        "vld1.32    {q4, q5}, [%[b2]]!    \n\t"
        "vmla.f32   q10, q2, d1[0]        \n\t"
        "vmla.f32   q11, q3, d1[0]        \n\t"
        "vmla.f32   q12, q4, d1[0]        \n\t"
        "vmla.f32   q13, q5, d1[0]        \n\t"

        "pld        [%[b3], #64]          \n\t"
        "vld1.32    {q2, q3}, [%[b3]]!    \n\t"
        "vld1.32    {q4, q5}, [%[b3]]!    \n\t"
        "vmla.f32   q10, q2, d1[1]        \n\t"
        "vmla.f32   q11, q3, d1[1]        \n\t"
        "vmla.f32   q12, q4, d1[1]        \n\t"
        "vmla.f32   q13, q5, d1[1]        \n\t"

        "vst1.32    {q10, q11}, [%[c0]]!  \n\t"
        "vst1.32    {q12, q13}, [%[c0]]!  \n\t"

        "subs       %[nc1], %[nc1], #1    \n\t"
        "bge        loop_nc1_%=           \n\t"
        "end_nc1_%=:                      \n\t"

        "subs       %[nc2], %[nc2], #1    \n\t"
        "blt        end_nc2_%=            \n\t"
        "loop_nc2_%=:                     \n\t"

        "cmp        %[i],       #0        \n\t"
        "beq        ii_eq0_%=             \n\t"
        "bne        ii_ne0_%=             \n\t"

        "ii_eq0_%=:                       \n\t"
        "vmov.f32   q10,    #0.0          \n\t"
        "b          gemm_nc2_%=           \n\t"

        "ii_ne0_%=:                       \n\t"
        "pld        [%[c0], #16]          \n\t"
        "vld1.32    {q10}, [%[c0]]        \n\t"

        "gemm_nc2_%=:                     \n\t"
        "pld        [%[b0], #16]          \n\t"
        "vld1.32    {q2}, [%[b0]]!        \n\t"
        "vmla.f32   q10, q2, d0[0]        \n\t"

        "pld        [%[b1], #16]          \n\t"
        "vld1.32    {q3}, [%[b1]]!        \n\t"
        "vmla.f32   q10, q3, d0[1]        \n\t"

        "pld        [%[b2], #16]          \n\t"
        "vld1.32    {q4}, [%[b2]]!        \n\t"
        "vmla.f32   q10, q4, d1[0]        \n\t"

        "pld        [%[b3], #16]          \n\t"
        "vld1.32    {q5}, [%[b3]]!        \n\t"
        "vmla.f32   q10, q5, d1[1]        \n\t"

        "vst1.32    {q10}, [%[c0]]!       \n\t"

        "subs       %[nc2], %[nc2], #1    \n\t"
        "bge        loop_nc2_%=           \n\t"
        "end_nc2_%=:                      \n\t"

        : [b0] "+r"(b0), [b1] "+r"(b1), [b2] "+r"(b2), [b3] "+r"(b3),
          [c0] "+r"(c0)
        : [a0] "r"(a0), [i] "r"(i), [nc1] "r"(nc1), [nc2] "r"(nc2)
        : "memory", "q0", "q2", "q3", "q4", "q5", "q10", "q11", "q12", "q13");

    for (int j = 0; j < nc3; j++) {
      if (i == 0) {
        *c0 = (*a0) * (*b0++);
      } else {
        *c0 += (*a0) * (*b0++);
      }
      *c0 += (*(a0 + 1)) * (*b1++);
      *c0 += (*(a0 + 2)) * (*b2++);
      *c0 += (*(a0 + 3)) * (*b3++);
      c0++;
    }
  }

  for (int i = 0; i < kc2; ++i) {
    a0 = A + 4 * kc1 + i;
    b0 = B + (4 * kc1 + i) * ldb;
    c0 = bufferC;
    asm volatile(
        "pld        [%[a0], #16]          \n\t"
        "vld1.32    {d0}, [%[a0]]         \n\t"

        "subs       %[nc1], %[nc1], #1    \n\t"
        "blt        end_nc1_%=            \n\t"
        "loop_nc1_%=:                     \n\t"

        "pld        [%[c0], #64]          \n\t"
        "vld1.32    {q10, q11}, [%[c0]]!  \n\t"
        "vld1.32    {q12, q13}, [%[c0]]   \n\t"
        "sub        %[c0], %[c0], #32     \n\t"

        "gemm_nc1_%=:                     \n\t"
        "pld        [%[b0], #64]          \n\t"
        "vld1.32    {q2, q3}, [%[b0]]!    \n\t"
        "vld1.32    {q4, q5}, [%[b0]]!    \n\t"
        "vmla.f32   q10, q2, d0[0]        \n\t"
        "vmla.f32   q11, q3, d0[0]        \n\t"
        "vmla.f32   q12, q4, d0[0]        \n\t"
        "vmla.f32   q13, q5, d0[0]        \n\t"

        "vst1.32    {q10, q11}, [%[c0]]!  \n\t"
        "vst1.32    {q12, q13}, [%[c0]]!  \n\t"

        "subs       %[nc1], %[nc1], #1    \n\t"
        "bge        loop_nc1_%=           \n\t"
        "end_nc1_%=:                      \n\t"

        "subs       %[nc2], %[nc2], #1    \n\t"
        "blt        end_nc2_%=            \n\t"
        "loop_nc2_%=:                     \n\t"

        "pld        [%[c0], #16]          \n\t"
        "vld1.32    {q10}, [%[c0]]        \n\t"

        "gemm_nc2_%=:                     \n\t"
        "vld1.32    {q2}, [%[b0]]!        \n\t"
        "vmla.f32   q10, q2, d0[0]        \n\t"

        "vst1.32    {q10}, [%[c0]]!       \n\t"

        "subs       %[nc2], %[nc2], #1    \n\t"
        "bge        loop_nc2_%=           \n\t"
        "end_nc2_%=:                      \n\t"

        : [b0] "+r"(b0), [b1] "+r"(b1), [b2] "+r"(b2), [b3] "+r"(b3),
          [c0] "+r"(c0)
        : [a0] "r"(a0), [nc1] "r"(nc1), [nc2] "r"(nc2)
        : "memory", "q0", "q2", "q3", "q4", "q5", "q10", "q11", "q12", "q13");

    for (int j = 0; j < nc3; j++) {
      *c0 += (*a0) * (*b0++);
      c0++;
    }
  }

  if (alpha != 1) {
    VecWriteWithAlphaBeta(n, bufferC, C, ldc);
    return;
  }
  if (beta == 0) {
    VecWriteBasic(n, bufferC, C, ldc);
    return;
  }
  if (beta == 1 && !relu) {
    VecWriteWithAdd(n, bufferC, C, ldc);
    return;
  }
  if (beta == 1 && relu) {
    VecWriteWithAddRelu(n, bufferC, C, ldc);
    return;
  }
}

void Gemm::VectorKernelWithBn(int m, int n, int k, float alpha, const float *A,
                              int lda, const float *B, int ldb, float beta,
                              float *C, int ldc, bool relu, float *new_scale,
                              float *new_bias) {
  float *bufferC = static_cast<float *>(memory::Alloc(sizeof(float) * n));

  const float *a0, *b0, *b1, *b2, *b3;
  float *c0, *C0;

  int volatile kc1 = k / 4;
  int volatile kc2 = k % 4;
  int volatile nc1 = n / 16;
  int _nc1 = n % 16;
  int volatile nc2 = _nc1 / 4;
  int volatile nc3 = _nc1 % 4;
  for (int i = 0; i < kc1; i++) {
    a0 = A + i * 4;
    b0 = B + i * 4 * ldb;
    b1 = b0 + ldb;
    b2 = b1 + ldb;
    b3 = b2 + ldb;
    c0 = bufferC;
    asm volatile(
        "pld        [%[a0], #16]          \n\t"
        "vld1.32    {q0}, [%[a0]]         \n\t"

        "subs       %[nc1], %[nc1], #1    \n\t"
        "blt        end_nc1_%=            \n\t"
        "loop_nc1_%=:                     \n\t"

        "cmp        %[i],       #0        \n\t"
        "beq        i_eq0_%=              \n\t"
        "bne        i_ne0_%=              \n\t"

        "i_eq0_%=:                        \n\t"
        "vmov.f32   q10,    #0.0          \n\t"
        "vmov.f32   q11,    #0.0          \n\t"
        "vmov.f32   q12,    #0.0          \n\t"
        "vmov.f32   q13,    #0.0          \n\t"
        "b          gemm_nc1_%=           \n\t"

        "i_ne0_%=:                        \n\t"
        "pld        [%[c0], #64]          \n\t"
        "vld1.32    {q10, q11}, [%[c0]]!  \n\t"
        "vld1.32    {q12, q13}, [%[c0]]   \n\t"
        "sub        %[c0], %[c0], #32     \n\t"

        "gemm_nc1_%=:                     \n\t"
        "pld        [%[b0], #64]          \n\t"
        "vld1.32    {q2, q3}, [%[b0]]!    \n\t"
        "vld1.32    {q4, q5}, [%[b0]]!    \n\t"
        "vmla.f32   q10, q2, d0[0]        \n\t"
        "vmla.f32   q11, q3, d0[0]        \n\t"
        "vmla.f32   q12, q4, d0[0]        \n\t"
        "vmla.f32   q13, q5, d0[0]        \n\t"

        "pld        [%[b1], #64]          \n\t"
        "vld1.32    {q2, q3}, [%[b1]]!    \n\t"
        "vld1.32    {q4, q5}, [%[b1]]!    \n\t"
        "vmla.f32   q10, q2, d0[1]        \n\t"
        "vmla.f32   q11, q3, d0[1]        \n\t"
        "vmla.f32   q12, q4, d0[1]        \n\t"
        "vmla.f32   q13, q5, d0[1]        \n\t"

        "pld        [%[b2], #64]          \n\t"
        "vld1.32    {q2, q3}, [%[b2]]!    \n\t"
        "vld1.32    {q4, q5}, [%[b2]]!    \n\t"
        "vmla.f32   q10, q2, d1[0]        \n\t"
        "vmla.f32   q11, q3, d1[0]        \n\t"
        "vmla.f32   q12, q4, d1[0]        \n\t"
        "vmla.f32   q13, q5, d1[0]        \n\t"

        "pld        [%[b3], #64]          \n\t"
        "vld1.32    {q2, q3}, [%[b3]]!    \n\t"
        "vld1.32    {q4, q5}, [%[b3]]!    \n\t"
        "vmla.f32   q10, q2, d1[1]        \n\t"
        "vmla.f32   q11, q3, d1[1]        \n\t"
        "vmla.f32   q12, q4, d1[1]        \n\t"
        "vmla.f32   q13, q5, d1[1]        \n\t"

        "vst1.32    {q10, q11}, [%[c0]]!  \n\t"
        "vst1.32    {q12, q13}, [%[c0]]!  \n\t"

        "subs       %[nc1], %[nc1], #1    \n\t"
        "bge        loop_nc1_%=           \n\t"
        "end_nc1_%=:                      \n\t"

        "subs       %[nc2], %[nc2], #1    \n\t"
        "blt        end_nc2_%=            \n\t"
        "loop_nc2_%=:                     \n\t"

        "cmp        %[i],       #0        \n\t"
        "beq        ii_eq0_%=             \n\t"
        "bne        ii_ne0_%=             \n\t"

        "ii_eq0_%=:                       \n\t"
        "vmov.f32   q10,    #0.0          \n\t"
        "b          gemm_nc2_%=           \n\t"

        "ii_ne0_%=:                       \n\t"
        "pld        [%[c0], #16]          \n\t"
        "vld1.32    {q10}, [%[c0]]        \n\t"

        "gemm_nc2_%=:                     \n\t"
        "pld        [%[b0], #16]          \n\t"
        "vld1.32    {q2}, [%[b0]]!        \n\t"
        "vmla.f32   q10, q2, d0[0]        \n\t"

        "pld        [%[b1], #16]          \n\t"
        "vld1.32    {q3}, [%[b1]]!        \n\t"
        "vmla.f32   q10, q3, d0[1]        \n\t"

        "pld        [%[b2], #16]          \n\t"
        "vld1.32    {q4}, [%[b2]]!        \n\t"
        "vmla.f32   q10, q4, d1[0]        \n\t"

        "pld        [%[b3], #16]          \n\t"
        "vld1.32    {q5}, [%[b3]]!        \n\t"
        "vmla.f32   q10, q5, d1[1]        \n\t"

        "vst1.32    {q10}, [%[c0]]!       \n\t"

        "subs       %[nc2], %[nc2], #1    \n\t"
        "bge        loop_nc2_%=           \n\t"
        "end_nc2_%=:                      \n\t"

        : [b0] "+r"(b0), [b1] "+r"(b1), [b2] "+r"(b2), [b3] "+r"(b3),
          [c0] "+r"(c0)
        : [a0] "r"(a0), [i] "r"(i), [nc1] "r"(nc1), [nc2] "r"(nc2)
        : "memory", "q0", "q2", "q3", "q4", "q5", "q10", "q11", "q12", "q13");

    for (int j = 0; j < nc3; j++) {
      if (i == 0) {
        *c0 = (*a0) * (*b0++);
      } else {
        *c0 += (*a0) * (*b0++);
      }
      *c0 += (*(a0 + 1)) * (*b1++);
      *c0 += (*(a0 + 2)) * (*b2++);
      *c0 += (*(a0 + 3)) * (*b3++);
      c0++;
    }
  }

  for (int i = 0; i < kc2; ++i) {
    a0 = A + 4 * kc1 + i;
    b0 = B + (4 * kc1 + i) * ldb;
    c0 = bufferC;
    asm volatile(
        "pld        [%[a0], #16]          \n\t"
        "vld1.32    {d0}, [%[a0]]         \n\t"

        "subs       %[nc1], %[nc1], #1    \n\t"
        "blt        end_nc1_%=            \n\t"
        "loop_nc1_%=:                     \n\t"

        "pld        [%[c0], #64]          \n\t"
        "vld1.32    {q10, q11}, [%[c0]]!  \n\t"
        "vld1.32    {q12, q13}, [%[c0]]   \n\t"
        "sub        %[c0], %[c0], #32     \n\t"

        "gemm_nc1_%=:                     \n\t"
        "pld        [%[b0], #64]          \n\t"
        "vld1.32    {q2, q3}, [%[b0]]!    \n\t"
        "vld1.32    {q4, q5}, [%[b0]]!    \n\t"
        "vmla.f32   q10, q2, d0[0]        \n\t"
        "vmla.f32   q11, q3, d0[0]        \n\t"
        "vmla.f32   q12, q4, d0[0]        \n\t"
        "vmla.f32   q13, q5, d0[0]        \n\t"

        "vst1.32    {q10, q11}, [%[c0]]!  \n\t"
        "vst1.32    {q12, q13}, [%[c0]]!  \n\t"

        "subs       %[nc1], %[nc1], #1    \n\t"
        "bge        loop_nc1_%=           \n\t"
        "end_nc1_%=:                      \n\t"

        "subs       %[nc2], %[nc2], #1    \n\t"
        "blt        end_nc2_%=            \n\t"
        "loop_nc2_%=:                     \n\t"

        "pld        [%[c0], #16]          \n\t"
        "vld1.32    {q10}, [%[c0]]        \n\t"

        "gemm_nc2_%=:                     \n\t"
        "vld1.32    {q2}, [%[b0]]!        \n\t"
        "vmla.f32   q10, q2, d0[0]        \n\t"

        "vst1.32    {q10}, [%[c0]]!       \n\t"

        "subs       %[nc2], %[nc2], #1    \n\t"
        "bge        loop_nc2_%=           \n\t"
        "end_nc2_%=:                      \n\t"

        : [b0] "+r"(b0), [b1] "+r"(b1), [b2] "+r"(b2), [b3] "+r"(b3),
          [c0] "+r"(c0)
        : [a0] "r"(a0), [nc1] "r"(nc1), [nc2] "r"(nc2)
        : "memory", "q0", "q2", "q3", "q4", "q5", "q10", "q11", "q12", "q13");

    for (int j = 0; j < nc3; j++) {
      *c0 += (*a0) * (*b0++);
      c0++;
    }
  }

  if (relu) {
    VecWriteWithBnRelu(n, bufferC, C, ldc, new_scale, new_bias);
  } else {
    VecWriteWithBn(n, bufferC, C, ldc, new_scale, new_bias);
  }
}

// C = A * B
void Gemm::WriteBasic(int mc, int nc, float *c, float *C, int ldc) {
  int nc1 = nc / 16;
  int _nc1 = nc % 16;
  int step = 4 * ldc;
  int step1 = 4 * (NC - 16 * nc1);
  int volatile m = mc;

  float *volatile c_ptr, *volatile C_ptr;
  float *C0, *c0;
  c_ptr = c;
  C_ptr = C;
  if (nc1 > 0) {
    asm volatile(
        "subs       %[mc], %[mc], #1        \n\t"
        "blt        end_mc_%=               \n\t"
        "loop_mc_%=:                        \n\t"

        "mov        r6,   %[C_ptr]          \n\t"
        "mov        r5,   %[nc1]            \n\t"
        "subs       r5,   r5,   #1          \n\t"
        "blt        end_nc1_%=              \n\t"
        "loop_nc1_%=:                       \n\t"

        "vld1.32    {q0, q1}, [%[c_ptr]]!   \n\t"
        "vst1.32    {q0, q1}, [r6]!         \n\t"

        "vld1.32    {q2, q3}, [%[c_ptr]]!   \n\t"
        "vst1.32    {q2, q3}, [r6]!         \n\t"

        "subs       r5,   r5,   #1          \n\t"
        "bge        loop_nc1_%=             \n\t"
        "end_nc1_%=:                        \n\t"

        "add        %[C_ptr], %[C_ptr], %[step]   \n\t"
        "add        %[c_ptr], %[c_ptr], %[step1]  \n\t"
        "subs       %[mc], %[mc], #1        \n\t"
        "bge        loop_mc_%=              \n\t"
        "end_mc_%=:                         \n\t"

        :
        : [C_ptr] "r"(C_ptr), [c_ptr] "r"(c_ptr), [mc] "r"(m), [nc1] "r"(nc1),
          [step] "r"(step), [step1] "r"(step1)
        : "memory", "r5", "r6", "q0", "q1", "q2", "q3");
  }

  if (_nc1 != 0) {
    for (int i = 0; i < mc; i++) {
      C0 = C_ptr + nc1 * 16 + i * ldc;
      c0 = c_ptr + nc1 * 16 + i * NC;
      for (int j = 0; j < _nc1; j++) {
        *C0++ = *c0++;
      }
    }
  }
}

// C = alpha * A * B + beta * C
void Gemm::WriteWithAlphaBeta(int mc, int nc, float *c, float *C, int ldc) {}

// C = A * B + C
void Gemm::WriteWithAdd(int mc, int nc, float *c, float *C, int ldc) {
  int nc1 = nc / 16;
  int _nc1 = nc % 16;
  int step = 4 * ldc;
  int step1 = 4 * (NC - 16 * nc1);
  int volatile m = mc;

  float *volatile c_ptr, *volatile C_ptr;
  float *C0, *c0;
  c_ptr = c;
  C_ptr = C;
  if (nc1 > 0) {
    asm volatile(
        "subs       %[mc], %[mc], #1        \n\t"
        "blt        end_mc_%=               \n\t"
        "loop_mc_%=:                        \n\t"

        "mov        r6,   %[C_ptr]          \n\t"
        "mov        r5,   %[nc1]            \n\t"
        "subs       r5,   r5,   #1          \n\t"
        "blt        end_nc1_%=              \n\t"
        "loop_nc1_%=:                       \n\t"

        "vld1.32    {q0, q1},   [r6]        \n\t"
        "vld1.32    {q2, q3},   [%[c_ptr]]! \n\t"
        "vadd.f32   q10,  q0,   q2          \n\t"
        "vadd.f32   q11,  q1,   q3          \n\t"
        "vst1.32    {q10, q11}, [r6]!       \n\t"

        "vld1.32    {q4, q5},   [r6]        \n\t"
        "vld1.32    {q6, q7},   [%[c_ptr]]! \n\t"
        "vadd.f32   q12,  q4,   q6          \n\t"
        "vadd.f32   q13,  q5,   q7          \n\t"
        "vst1.32    {q12, q13}, [r6]!       \n\t"

        "subs       r5,   r5,   #1          \n\t"
        "bge        loop_nc1_%=             \n\t"
        "end_nc1_%=:                        \n\t"

        "add        %[C_ptr], %[C_ptr], %[step]     \n\t"
        "add        %[c_ptr], %[c_ptr], %[step1]    \n\t"
        "subs       %[mc], %[mc], #1        \n\t"
        "bge        loop_mc_%=              \n\t"
        "end_mc_%=:                         \n\t"

        :
        : [C_ptr] "r"(C_ptr), [c_ptr] "r"(c_ptr), [mc] "r"(m), [nc1] "r"(nc1),
          [step] "r"(step), [step1] "r"(step1)
        : "memory", "r5", "r6", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
          "q10", "q11", "q12", "q13");
  }

  if (_nc1 != 0) {
    for (int i = 0; i < mc; i++) {
      C0 = C_ptr + nc1 * 16 + i * ldc;
      c0 = c_ptr + nc1 * 16 + i * NC;
      for (int j = 0; j < _nc1; j++) {
        *C0++ += *c0++;
      }
    }
  }
}

// C = A * B + bias
void Gemm::WriteWithAddV1(int mc, int nc, float *c, float *C, int ldc,
                          float *bias) {
  int nc1 = nc / 4;
  int _nc1 = nc % 4;

  float *c_ptr, *C_ptr;
  float32x4_t cv;
  float32x4_t biasv;
  for (int i = 0; i < mc; ++i) {
    c_ptr = c + i * NC;
    C_ptr = C + i * ldc;
    biasv = vld1q_dup_f32(bias + i);
    for (int j = 0; j < nc1; ++j) {
      cv = vld1q_f32(c_ptr);
      cv = vaddq_f32(cv, biasv);
      vst1q_f32(C_ptr, cv);
      c_ptr += 4;
      C_ptr += 4;
    }
    if (_nc1 != 0) {
      cv = vld1q_f32(c_ptr);
      cv = vaddq_f32(cv, biasv);
      if (_nc1 >= 1) {
        vst1q_lane_f32(C_ptr, cv, 0);
        C_ptr++;
      }
      if (_nc1 >= 2) {
        vst1q_lane_f32(C_ptr, cv, 1);
        C_ptr++;
      }
      if (_nc1 >= 3) {
        vst1q_lane_f32(C_ptr, cv, 2);
        C_ptr++;
      }
    }
  }
}

// C = A * B + C, relu(C)
void Gemm::WriteWithAddRelu(int mc, int nc, float *c, float *C, int ldc) {
  int nc1 = nc / 16;
  int _nc1 = nc % 16;
  int step = 4 * ldc;
  int step1 = 4 * (NC - 16 * nc1);
  int volatile m = mc;

  float *volatile c_ptr, *volatile C_ptr;
  float *C0, *c0;
  c_ptr = c;
  C_ptr = C;
  if (nc1 > 0) {
    asm volatile(
        "vmov.f32   q14,    #0.0            \n\t"
        "subs       %[mc], %[mc], #1        \n\t"
        "blt        end_mc_%=               \n\t"
        "loop_mc_%=:                        \n\t"

        "mov        r6,   %[C_ptr]          \n\t"
        "mov        r5,   %[nc1]            \n\t"
        "subs       r5,   r5,   #1          \n\t"
        "blt        end_nc1_%=              \n\t"
        "loop_nc1_%=:                       \n\t"

        "vld1.32    {q0, q1},   [r6]        \n\t"
        "vld1.32    {q2, q3},   [%[c_ptr]]! \n\t"
        "vadd.f32   q10,  q0,   q2          \n\t"
        "vadd.f32   q11,  q1,   q3          \n\t"
        "vmax.f32   q10,  q10,  q14         \n\t"
        "vmax.f32   q11,  q11,  q14         \n\t"
        "vst1.32    {q10, q11}, [r6]!       \n\t"

        "vld1.32    {q4, q5},   [r6]        \n\t"
        "vld1.32    {q6, q7},   [%[c_ptr]]! \n\t"
        "vadd.f32   q12,  q4,   q6          \n\t"
        "vadd.f32   q13,  q5,   q7          \n\t"
        "vmax.f32   q12,  q12,  q14         \n\t"
        "vmax.f32   q13,  q13,  q14         \n\t"
        "vst1.32    {q12, q13}, [r6]!       \n\t"

        "subs       r5,   r5,   #1          \n\t"
        "bge        loop_nc1_%=             \n\t"
        "end_nc1_%=:                        \n\t"

        "add        %[C_ptr], %[C_ptr], %[step]     \n\t"
        "add        %[c_ptr], %[c_ptr], %[step1]    \n\t"
        "subs       %[mc], %[mc], #1        \n\t"
        "bge        loop_mc_%=              \n\t"
        "end_mc_%=:                         \n\t"

        :
        : [C_ptr] "r"(C_ptr), [c_ptr] "r"(c_ptr), [mc] "r"(m), [nc1] "r"(nc1),
          [step] "r"(step), [step1] "r"(step1)
        : "memory", "r5", "r6", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
          "q10", "q11", "q12", "q13");
  }

  if (_nc1 != 0) {
    for (int i = 0; i < mc; i++) {
      C0 = C_ptr + nc1 * 16 + i * ldc;
      c0 = c_ptr + nc1 * 16 + i * NC;
      for (int j = 0; j < _nc1; j++) {
        *C0 += *c0;
        if (*C0 < 0) {
          *C0 = 0;
        }
        C0++;
        c0++;
      }
    }
  }
}

// C = A * B + bias, relu(C)
void Gemm::WriteWithAddReluV1(int mc, int nc, float *c, float *C, int ldc,
                              float *bias) {
  int nc1 = nc / 4;
  int _nc1 = nc % 4;

  float *c_ptr, *C_ptr;
  float32x4_t cv;
  float32x4_t biasv;
  float32x4_t zero = vdupq_n_f32(0.0);
  for (int i = 0; i < mc; ++i) {
    c_ptr = c + i * NC;
    C_ptr = C + i * ldc;
    biasv = vld1q_dup_f32(bias + i);
    for (int j = 0; j < nc1; ++j) {
      cv = vld1q_f32(c_ptr);
      cv = vaddq_f32(cv, biasv);
      cv = vmaxq_f32(cv, zero);
      vst1q_f32(C_ptr, cv);
      c_ptr += 4;
      C_ptr += 4;
    }
    if (_nc1 != 0) {
      cv = vld1q_f32(c_ptr);
      cv = vaddq_f32(cv, biasv);
      cv = vmaxq_f32(cv, zero);
      if (_nc1 >= 1) {
        vst1q_lane_f32(C_ptr, cv, 0);
        C_ptr++;
      }
      if (_nc1 >= 2) {
        vst1q_lane_f32(C_ptr, cv, 1);
        C_ptr++;
      }
      if (_nc1 >= 3) {
        vst1q_lane_f32(C_ptr, cv, 2);
        C_ptr++;
      }
    }
  }
}

void Gemm::WriteWithAddPRelu(int mc, int nc, float *c, float *C, int ldc,
                             float *p, std::string mode, float *bias,
                             float *bias1) {
  if (nc < 4) {
    if (bias1 == nullptr) {
      for (int i = 0; i < mc; ++i) {
        for (int j = 0; j < nc; ++j) {
          float r = c[i * NC + j] + bias[i];
          if (r < 0) {
            r *= p[i];
          }
          C[i * ldc + j] = r;
        }
      }
    } else {
      for (int i = 0; i < mc; ++i) {
        for (int j = 0; j < nc; ++j) {
          float r = c[i * NC + j] + bias[i];
          r += bias1[i * ldc + j];
          if (r < 0) {
            r *= p[i];
          }
          C[i * ldc + j] = r;
        }
      }
    }
    return;
  }

  int nc1 = nc / 16;
  int _nc1 = nc % 16;
  int nc2 = _nc1 / 4;
  int nc3 = 16 - 4 * (_nc1 % 4);
  int step = 4 * (ldc - nc);
  int step1 = 4 * (NC - nc);

  if (bias1 == nullptr) {
    asm volatile(
        "vmov.f32   q14,    #0.0            \n\t"
        "subs       %[mc], %[mc], #1        \n\t"
        "blt        end_mc_%=               \n\t"
        "loop_mc_%=:                        \n\t"

        "mov        r5,     %[nc1]          \n\t"
        "mov        r6,     %[nc2]          \n\t"
        "vld1.32    {d0},   [%[bias]]       \n\t"
        "vld1.32    {d1},   [%[p]]          \n\t"
        "vdup.32    q1,     d0[0]           \n\t"
        "vdup.32    q2,     d1[0]           \n\t"

        "subs       r5,   r5,   #1          \n\t"
        "blt        end_nc1_%=              \n\t"
        "loop_nc1_%=:                       \n\t"

        "pld        [%[c], #32]             \n\t"
        "vld1.32    {q3, q4},   [%[c]]!     \n\t"
        "vld1.32    {q9, q10},  [%[c]]!     \n\t"

        "vadd.f32   q3,   q3,   q1          \n\t"
        "vadd.f32   q4,   q4,   q1          \n\t"
        "vadd.f32   q9,   q9,   q1          \n\t"
        "vadd.f32   q10,  q10,  q1          \n\t"

        "vmax.f32   q5,   q3,   q14         \n\t"
        "vmin.f32   q7,   q3,   q14         \n\t"
        "vmax.f32   q6,   q4,   q14         \n\t"
        "vmin.f32   q8,   q4,   q14         \n\t"

        "vmax.f32   q11,  q9,   q14         \n\t"
        "vmin.f32   q13,  q9,   q14         \n\t"
        "vmax.f32   q12,  q10,  q14         \n\t"
        "vmin.f32   q15,  q10,  q14         \n\t"

        "vmla.f32   q5,   q7,   q2          \n\t"
        "vmla.f32   q6,   q8,   q2          \n\t"
        "vmla.f32   q11,  q13,  q2          \n\t"
        "vmla.f32   q12,  q15,  q2          \n\t"

        "vst1.32    {q5, q6},   [%[C]]!     \n\t"
        "vst1.32    {q11, q12}, [%[C]]!     \n\t"

        "subs       r5,   r5,   #1          \n\t"
        "bge        loop_nc1_%=             \n\t"
        "end_nc1_%=:                        \n\t"

        "subs       r6,  r6,   #1           \n\t"
        "blt        end_nc2_%=              \n\t"
        "loop_nc2_%=:                       \n\t"

        "vld1.32    {q3},       [%[c]]!     \n\t"
        "vadd.f32   q3,   q3,   q1          \n\t"
        "vmax.f32   q5,   q3,   q14         \n\t"
        "vmin.f32   q7,   q3,   q14         \n\t"
        "vmla.f32   q5,   q7,   q2          \n\t"
        "vst1.32    {q5},       [%[C]]!     \n\t"

        "subs       r6,   r6,   #1          \n\t"
        "bge        loop_nc2_%=             \n\t"
        "end_nc2_%=:                        \n\t"

        "cmp        %[nc3],    #16          \n\t"
        "beq        end_nc3_%=              \n\t"

        "sub        %[c],     %[c],   %[nc3]      \n\t"
        "sub        %[C],     %[C],   %[nc3]      \n\t"

        "vld1.32    {q4},       [%[c]]!     \n\t"
        "vadd.f32   q4,   q4,   q1          \n\t"
        "vmax.f32   q6,   q4,   q14         \n\t"
        "vmin.f32   q8,   q4,   q14         \n\t"
        "vmla.f32   q6,   q8,   q2          \n\t"
        "vst1.32    {q6},       [%[C]]!     \n\t"
        "end_nc3_%=:                        \n\t"

        "add        %[p],     %[p],     #4        \n\t"
        "add        %[bias],  %[bias],  #4        \n\t"
        "add        %[c],     %[c],     %[step1]  \n\t"
        "add        %[C],     %[C],     %[step]   \n\t"

        "subs       %[mc], %[mc], #1        \n\t"
        "bge        loop_mc_%=              \n\t"
        "end_mc_%=:                         \n\t"

        :
        : [C] "r"(C), [c] "r"(c), [mc] "r"(mc), [nc1] "r"(nc1), [nc2] "r"(nc2),
          [nc3] "r"(nc3), [step] "r"(step), [step1] "r"(step1), [p] "r"(p),
          [bias] "r"(bias), [bias1] "r"(bias1)
        : "memory", "r5", "r6", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
          "q8");
  } else {
    asm volatile(
        "vmov.f32   q14,    #0.0            \n\t"
        "subs       %[mc], %[mc], #1        \n\t"
        "blt        end_mc_%=               \n\t"
        "loop_mc_%=:                        \n\t"

        "mov        r5,     %[nc1]          \n\t"
        "mov        r6,     %[nc2]          \n\t"
        "vld1.32    {d0},   [%[bias]]       \n\t"
        "vld1.32    {d1},   [%[p]]          \n\t"
        "vdup.32    q1,     d0[0]           \n\t"
        "vdup.32    q2,     d1[0]           \n\t"

        "subs       r5,   r5,   #1          \n\t"
        "blt        end_nc1_%=              \n\t"
        "loop_nc1_%=:                       \n\t"

        "pld        [%[c], #32]             \n\t"
        "pld        [%[bias1], #32]         \n\t"
        "vld1.32    {q3, q4},   [%[c]]!     \n\t"
        "vld1.32    {q9, q10},  [%[bias1]]! \n\t"
        "vadd.f32   q3,   q3,   q1          \n\t"
        "vadd.f32   q4,   q4,   q1          \n\t"
        "vadd.f32   q3,   q3,   q9          \n\t"
        "vadd.f32   q4,   q4,   q10         \n\t"
        "vmax.f32   q5,   q3,   q14         \n\t"
        "vmin.f32   q7,   q3,   q14         \n\t"
        "vmax.f32   q6,   q4,   q14         \n\t"
        "vmin.f32   q8,   q4,   q14         \n\t"
        "vmla.f32   q5,   q7,   q2          \n\t"
        "vmla.f32   q6,   q8,   q2          \n\t"
        "vst1.32    {q5, q6},   [%[C]]!     \n\t"

        "vld1.32    {q3, q4},   [%[c]]!     \n\t"
        "vld1.32    {q9, q10},  [%[bias1]]! \n\t"
        "vadd.f32   q3,   q3,   q1          \n\t"
        "vadd.f32   q4,   q4,   q1          \n\t"
        "vadd.f32   q3,   q3,   q9          \n\t"
        "vadd.f32   q4,   q4,   q10         \n\t"
        "vmax.f32   q5,   q3,   q14         \n\t"
        "vmin.f32   q7,   q3,   q14         \n\t"
        "vmax.f32   q6,   q4,   q14         \n\t"
        "vmin.f32   q8,   q4,   q14         \n\t"
        "vmla.f32   q5,   q7,   q2          \n\t"
        "vmla.f32   q6,   q8,   q2          \n\t"
        "vst1.32    {q5, q6},   [%[C]]!     \n\t"

        "subs       r5,   r5,   #1          \n\t"
        "bge        loop_nc1_%=             \n\t"
        "end_nc1_%=:                        \n\t"

        "subs       r6,  r6,   #1           \n\t"
        "blt        end_nc2_%=              \n\t"
        "loop_nc2_%=:                       \n\t"

        "vld1.32    {q3},       [%[c]]!     \n\t"
        "vld1.32    {q9},       [%[bias1]]! \n\t"
        "vadd.f32   q3,   q3,   q1          \n\t"
        "vadd.f32   q3,   q3,   q9          \n\t"
        "vmax.f32   q5,   q3,   q14         \n\t"
        "vmin.f32   q7,   q3,   q14         \n\t"
        "vmla.f32   q5,   q7,   q2          \n\t"
        "vst1.32    {q5},      [%[C]]!      \n\t"

        "subs       r6,   r6,   #1          \n\t"
        "bge        loop_nc2_%=             \n\t"
        "end_nc2_%=:                        \n\t"

        "cmp        %[nc3],    #16          \n\t"
        "beq        end_nc3_%=              \n\t"

        "sub        %[c],     %[c],     %[nc3]    \n\t"
        "sub        %[C],     %[C],     %[nc3]    \n\t"
        "sub        %[bias1], %[bias1], %[nc3]    \n\t"

        "vld1.32    {q4},       [%[c]]!     \n\t"
        "vld1.32    {q10},      [%[bias1]]! \n\t"
        "vadd.f32   q4,   q4,   q1          \n\t"
        "vadd.f32   q4,   q4,   q10         \n\t"
        "vmax.f32   q6,   q4,   q14         \n\t"
        "vmin.f32   q8,   q4,   q14         \n\t"
        "vmla.f32   q6,   q8,   q2          \n\t"
        "vst1.32    {q6},       [%[C]]!     \n\t"
        "end_nc3_%=:                        \n\t"

        "add        %[p],     %[p],     #4        \n\t"
        "add        %[bias],  %[bias],  #4        \n\t"
        "add        %[c],     %[c],     %[step1]  \n\t"
        "add        %[C],     %[C],     %[step]   \n\t"
        "add        %[bias1], %[bias1], %[step]   \n\t"

        "subs       %[mc], %[mc], #1        \n\t"
        "bge        loop_mc_%=              \n\t"
        "end_mc_%=:                         \n\t"

        :
        : [C] "r"(C), [c] "r"(c), [mc] "r"(mc), [nc1] "r"(nc1), [nc2] "r"(nc2),
          [nc3] "r"(nc3), [step] "r"(step), [step1] "r"(step1), [p] "r"(p),
          [bias] "r"(bias), [bias1] "r"(bias1)
        : "memory", "r5", "r6", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
          "q8", "q9", "q10");
  }
}

// C = A * B, batchnorm(C)
void Gemm::WriteWithBn(int mc, int nc, float *c, float *C, int ldc,
                       float *scale, float *bias) {
  if (nc < 4) {
    for (int i = 0; i < mc; ++i) {
      for (int j = 0; j < nc; ++j) {
        *C = (*c) * (*scale) + (*bias);
        C++;
        c++;
      }
      C += (ldc - nc);
      c += (NC - nc);
      scale++;
      bias++;
    }
    return;
  }

  int volatile nc1 = nc / 16;
  int _nc1 = nc % 16;
  int volatile nc2 = _nc1 / 4;
  int volatile nc3 = 16 - 4 * (_nc1 % 4);
  int volatile step = 4 * (ldc - nc);
  int volatile step1 = 4 * (NC - nc);

  asm volatile(
      "subs       %[mc], %[mc], #1        \n\t"
      "blt        end_mc_%=               \n\t"
      "loop_mc_%=:                        \n\t"

      "mov        r5,   %[nc1]            \n\t"
      "mov        r6,   %[nc2]            \n\t"
      "vld1.32    {d0},   [%[scale]]      \n\t"
      "vld1.32    {d1},   [%[bias]]       \n\t"
      "vdup.32    q1,   d0[0]             \n\t"
      "vdup.32    q2,   d1[0]             \n\t"

      "subs       r5,   r5,   #1          \n\t"
      "blt        end_nc1_%=              \n\t"
      "loop_nc1_%=:                       \n\t"

      "vld1.32    {q3, q4},   [%[c]]!     \n\t"
      "vmul.f32   q10,  q3,   q1          \n\t"
      "vmul.f32   q11,  q4,   q1          \n\t"
      "vadd.f32   q10,  q10,  q2          \n\t"
      "vadd.f32   q11,  q11,  q2          \n\t"
      "vst1.32    {q10, q11}, [%[C]]!     \n\t"

      "vld1.32    {q5, q6},   [%[c]]!     \n\t"
      "vmul.f32   q12,  q5,   q1          \n\t"
      "vmul.f32   q13,  q6,   q1          \n\t"
      "vadd.f32   q12,  q12,  q2          \n\t"
      "vadd.f32   q13,  q13,  q2          \n\t"
      "vst1.32    {q12, q13}, [%[C]]!     \n\t"

      "subs       r5,   r5,   #1          \n\t"
      "bge        loop_nc1_%=             \n\t"
      "end_nc1_%=:                        \n\t"

      "subs       r6,  r6,   #1           \n\t"
      "blt        end_nc2_%=              \n\t"
      "loop_nc2_%=:                       \n\t"

      "vld1.32    {q7},       [%[c]]!     \n\t"
      "vmul.f32   q10,  q7,   q1          \n\t"
      "vadd.f32   q10,  q10,  q2          \n\t"
      "vst1.32    {q10},      [%[C]]!     \n\t"

      "subs       r6,   r6,   #1          \n\t"
      "bge        loop_nc2_%=             \n\t"
      "end_nc2_%=:                        \n\t"

      "cmp        %[nc3],    #16          \n\t"
      "beq        end_nc3_%=              \n\t"

      "sub        %[c],     %[c],   %[nc3]      \n\t"
      "sub        %[C],     %[C],   %[nc3]      \n\t"

      "vld1.32    {q8},       [%[c]]!     \n\t"
      "vmul.f32   q11,  q8,   q1          \n\t"
      "vadd.f32   q11,  q11,  q2          \n\t"
      "vst1.32    {q11},      [%[C]]!     \n\t"
      "end_nc3_%=:                        \n\t"

      "add        %[scale], %[scale], #4        \n\t"
      "add        %[bias],  %[bias],  #4        \n\t"
      "add        %[c],     %[c],     %[step1]  \n\t"
      "add        %[C],     %[C],     %[step]   \n\t"

      "subs       %[mc], %[mc], #1        \n\t"
      "bge        loop_mc_%=              \n\t"
      "end_mc_%=:                         \n\t"

      :
      : [C] "r"(C), [c] "r"(c), [mc] "r"(mc), [nc1] "r"(nc1), [nc2] "r"(nc2),
        [nc3] "r"(nc3), [step] "r"(step), [step1] "r"(step1),
        [scale] "r"(scale), [bias] "r"(bias)
      : "memory", "r5", "r6", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
        "q8", "q10", "q11", "q12", "q13");
}

// C = A * B, batchnorm(C), relu(C)
void Gemm::WriteWithBnRelu(int mc, int nc, float *c, float *C, int ldc,
                           float *scale, float *bias) {
  if (nc < 4) {
    for (int i = 0; i < mc; ++i) {
      for (int j = 0; j < nc; ++j) {
        *C = (*c) * (*scale) + (*bias);
        if (*C < 0) {
          *C = 0;
        }
        C++;
        c++;
      }
      C += (ldc - nc);
      c += (NC - nc);
      scale++;
      bias++;
    }
    return;
  }

  int nc1 = nc / 16;
  int _nc1 = nc % 16;
  int nc2 = _nc1 / 4;
  int nc3 = 16 - 4 * (_nc1 % 4);
  int step = 4 * (ldc - nc);
  int step1 = 4 * (NC - nc);

  asm volatile(
      "vmov.f32   q14,    #0.0            \n\t"
      "subs       %[mc], %[mc], #1        \n\t"
      "blt        end_mc_%=               \n\t"
      "loop_mc_%=:                        \n\t"

      "mov        r5,   %[nc1]            \n\t"
      "mov        r6,   %[nc2]            \n\t"
      "vld1.32    {d0},   [%[scale]]      \n\t"
      "vld1.32    {d1},   [%[bias]]       \n\t"
      "vdup.32    q1,   d0[0]             \n\t"
      "vdup.32    q2,   d1[0]             \n\t"

      "subs       r5,   r5,   #1          \n\t"
      "blt        end_nc1_%=              \n\t"
      "loop_nc1_%=:                       \n\t"

      "vld1.32    {q3, q4},   [%[c]]!     \n\t"
      "vmul.f32   q10,  q3,   q1          \n\t"
      "vmul.f32   q11,  q4,   q1          \n\t"
      "vadd.f32   q10,  q10,  q2          \n\t"
      "vadd.f32   q11,  q11,  q2          \n\t"
      "vmax.f32   q10,  q10,  q14         \n\t"
      "vmax.f32   q11,  q11,  q14         \n\t"
      "vst1.32    {q10, q11}, [%[C]]!     \n\t"

      "vld1.32    {q5, q6},   [%[c]]!     \n\t"
      "vmul.f32   q12,  q5,   q1          \n\t"
      "vmul.f32   q13,  q6,   q1          \n\t"
      "vadd.f32   q12,  q12,  q2          \n\t"
      "vadd.f32   q13,  q13,  q2          \n\t"
      "vmax.f32   q12,  q12,  q14         \n\t"
      "vmax.f32   q13,  q13,  q14         \n\t"
      "vst1.32    {q12, q13}, [%[C]]!     \n\t"

      "subs       r5,   r5,   #1          \n\t"
      "bge        loop_nc1_%=             \n\t"
      "end_nc1_%=:                        \n\t"

      "subs       r6,  r6,   #1           \n\t"
      "blt        end_nc2_%=              \n\t"
      "loop_nc2_%=:                       \n\t"

      "vld1.32    {q7},       [%[c]]!     \n\t"
      "vmul.f32   q10,  q7,   q1          \n\t"
      "vadd.f32   q10,  q10,  q2          \n\t"
      "vmax.f32   q10,  q10,  q14         \n\t"
      "vst1.32    {q10},      [%[C]]!     \n\t"

      "subs       r6,   r6,   #1          \n\t"
      "bge        loop_nc2_%=             \n\t"
      "end_nc2_%=:                        \n\t"

      "cmp        %[nc3],    #16          \n\t"
      "beq        end_nc3_%=              \n\t"

      "sub        %[c],     %[c],   %[nc3]      \n\t"
      "sub        %[C],     %[C],   %[nc3]      \n\t"

      "vld1.32    {q8},       [%[c]]!     \n\t"
      "vmul.f32   q11,  q8,   q1          \n\t"
      "vadd.f32   q11,  q11,  q2          \n\t"
      "vmax.f32   q11,  q11,  q14         \n\t"
      "vst1.32    {q11},      [%[C]]!     \n\t"
      "end_nc3_%=:                        \n\t"

      "add        %[scale], %[scale], #4        \n\t"
      "add        %[bias],  %[bias],  #4        \n\t"
      "add        %[c],     %[c],     %[step1]  \n\t"
      "add        %[C],     %[C],     %[step]   \n\t"

      "subs       %[mc], %[mc], #1        \n\t"
      "bge        loop_mc_%=              \n\t"
      "end_mc_%=:                         \n\t"

      :
      : [C] "r"(C), [c] "r"(c), [mc] "r"(mc), [nc1] "r"(nc1), [nc2] "r"(nc2),
        [nc3] "r"(nc3), [step] "r"(step), [step1] "r"(step1),
        [scale] "r"(scale), [bias] "r"(bias)
      : "memory", "r5", "r6", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
        "q8", "q10", "q11", "q12", "q13", "q14");
}

// C = A * B, batchnorm(C),C = C + bias; relu(C)
void Gemm::WriteWithBnAddRelu(int mc, int nc, float *c, float *C, int ldc,
                              float *new_scale, float *new_bias, float *bias) {
  int nc1 = nc / 4;
  int _nc1 = nc % 4;

  float *c_ptr, *C_ptr, *bias_ptr;
  float32x4_t cv;
  float32x4_t nbias;
  float32x2_t scale;
  float32x4_t biasv;
  float32x4_t zero = vdupq_n_f32(0.0);
  for (int i = 0; i < mc; ++i) {
    c_ptr = c + i * NC;
    C_ptr = C + i * ldc;
    bias_ptr = bias + i * ldc;
    nbias = vld1q_dup_f32(new_bias);
    scale = vld1_dup_f32(new_scale);
    new_bias++;
    new_scale++;
    float scale0 = vget_lane_f32(scale, 0);
    for (int j = 0; j < nc1; ++j) {
      cv = vld1q_f32(c_ptr);
      biasv = vld1q_f32(bias_ptr);
      cv = vmlaq_n_f32(nbias, cv, scale0);
      cv = vaddq_f32(cv, biasv);
      cv = vmaxq_f32(cv, zero);
      vst1q_f32(C_ptr, cv);
      c_ptr += 4;
      C_ptr += 4;
      bias_ptr += 4;
    }
    if (_nc1 != 0) {
      cv = vld1q_f32(c_ptr);
      biasv = vld1q_f32(bias_ptr);
      cv = vmlaq_n_f32(nbias, cv, scale0);
      cv = vaddq_f32(cv, biasv);
      cv = vmaxq_f32(cv, zero);
      if (_nc1 >= 1) {
        vst1q_lane_f32(C_ptr, cv, 0);
        C_ptr++;
      }
      if (_nc1 >= 2) {
        vst1q_lane_f32(C_ptr, cv, 1);
        C_ptr++;
      }
      if (_nc1 >= 3) {
        vst1q_lane_f32(C_ptr, cv, 2);
      }
    }
  }
}

// C = A * B
void Gemm::VecWriteBasic(int n, float *c, float *C, int ldc) {
  int nc1 = n / 16;
  int _nc1 = n % 16;
  int nc2 = _nc1 / 4;
  int nc3 = 16 - 4 * (_nc1 % 4);

  asm volatile(
      "subs       %[nc1],   %[nc1],   #1  \n\t"
      "blt        end_nc1_%=              \n\t"
      "loop_nc1_%=:                       \n\t"

      "vld1.32    {q0, q1}, [%[c]]!       \n\t"
      "vst1.32    {q0, q1}, [%[C]]!       \n\t"

      "vld1.32    {q2, q3}, [%[c]]!       \n\t"
      "vst1.32    {q2, q3}, [%[C]]!       \n\t"

      "subs       %[nc1],   %[nc1],   #1  \n\t"
      "bge        loop_nc1_%=             \n\t"
      "end_nc1_%=:                        \n\t"

      "subs       %[nc2],   %[nc2],   #1  \n\t"
      "blt        end_nc2_%=              \n\t"
      "loop_nc2_%=:                       \n\t"

      "vld1.32    {q4},     [%[c]]!       \n\t"
      "vst1.32    {q4},     [%[C]]!       \n\t"

      "subs       %[nc2],   %[nc2],   #1  \n\t"
      "bge        loop_nc2_%=             \n\t"
      "end_nc2_%=:                        \n\t"

      "cmp        %[nc3],    #16          \n\t"
      "beq        end_nc3_%=              \n\t"
      "sub        %[c],     %[c],   %[nc3]    \n\t"
      "sub        %[C],     %[C],   %[nc3]    \n\t"
      "vld1.32    {q5},     [%[c]]!       \n\t"
      "vst1.32    {q5},     [%[C]]!       \n\t"
      "end_nc3_%=:                        \n\t"

      :
      : [C] "r"(C), [c] "r"(c), [nc1] "r"(nc1), [nc2] "r"(nc2), [nc3] "r"(nc3)
      : "memory", "q0", "q1", "q2", "q3", "q4", "q5");
}

// C = alpha * A * B + beta * C
void Gemm::VecWriteWithAlphaBeta(int n, float *c, float *C, int ldc) {}

// C = A * B + C
void Gemm::VecWriteWithAdd(int n, float *c, float *C, int ldc) {
  int nc1 = n / 16;
  int _nc1 = n % 16;

  asm volatile(
      "subs       %[nc1],   %[nc1],   #1  \n\t"
      "blt        end_nc1_%=              \n\t"
      "loop_nc1_%=:                       \n\t"

      "vld1.32    {q0, q1},   [%[c]]!     \n\t"
      "vld1.32    {q2, q3},   [%[C]]      \n\t"
      "vadd.f32   q10,  q0,   q2          \n\t"
      "vadd.f32   q11,  q1,   q3          \n\t"
      "vst1.32    {q10, q11}, [%[C]]!     \n\t"

      "vld1.32    {q4, q5},   [%[c]]!     \n\t"
      "vld1.32    {q6, q7},   [%[C]]      \n\t"
      "vadd.f32   q12,  q4,   q6          \n\t"
      "vadd.f32   q13,  q5,   q7          \n\t"
      "vst1.32    {q12, q13}, [%[C]]!     \n\t"

      "subs       %[nc1],   %[nc1],   #1  \n\t"
      "bge        loop_nc1_%=             \n\t"
      "end_nc1_%=:                        \n\t"

      : [C] "+r"(C), [c] "+r"(c)
      : [nc1] "r"(nc1)
      : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q10", "q11",
        "q12", "q13");

  if (_nc1 != 0) {
    for (int j = 0; j < _nc1; j++) {
      *C++ += *c++;
    }
  }
}

// C = A * B + C, relu(C)
void Gemm::VecWriteWithAddRelu(int n, float *c, float *C, int ldc) {
  int nc1 = n / 16;
  int _nc1 = n % 16;

  asm volatile(
      "vmov.f32   q14,      #0.0          \n\t"
      "subs       %[nc1],   %[nc1],   #1  \n\t"
      "blt        end_nc1_%=              \n\t"
      "loop_nc1_%=:                       \n\t"

      "vld1.32    {q0, q1},   [%[c]]!     \n\t"
      "vld1.32    {q2, q3},   [%[C]]      \n\t"
      "vadd.f32   q10,  q0,   q2          \n\t"
      "vadd.f32   q11,  q1,   q3          \n\t"
      "vmax.f32   q10,  q10,  q14         \n\t"
      "vmax.f32   q11,  q11,  q14         \n\t"
      "vst1.32    {q10, q11}, [%[C]]!     \n\t"

      "vld1.32    {q4, q5},   [%[c]]!     \n\t"
      "vld1.32    {q6, q7},   [%[C]]      \n\t"
      "vadd.f32   q12,  q4,   q6          \n\t"
      "vadd.f32   q13,  q5,   q7          \n\t"
      "vmax.f32   q12,  q12,  q14         \n\t"
      "vmax.f32   q13,  q13,  q14         \n\t"
      "vst1.32    {q12, q13}, [%[C]]!     \n\t"

      "subs       %[nc1],   %[nc1],   #1  \n\t"
      "bge        loop_nc1_%=             \n\t"
      "end_nc1_%=:                        \n\t"

      : [C] "+r"(C), [c] "+r"(c)
      : [nc1] "r"(nc1)
      : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q10", "q11",
        "q12", "q13");

  if (_nc1 != 0) {
    for (int j = 0; j < _nc1; j++) {
      *C += *c;
      if (*C < 0) {
        *C = 0;
      }
      C++;
      c++;
    }
  }
}

// C = A * B, batchnorm(C)
void Gemm::VecWriteWithBn(int n, float *c, float *C, int ldc, float *scale,
                          float *bias) {
  int nc1 = n / 16;
  int _nc1 = n % 16;
  int nc2 = _nc1 / 4;
  int nc3 = 16 - 4 * (_nc1 % 4);

  asm volatile(
      "subs       %[nc1],   %[nc1],   #1  \n\t"
      "blt        end_nc1_%=              \n\t"
      "loop_nc1_%=:                       \n\t"

      "vld1.32    {q0, q1},   [%[c]]!     \n\t"
      "vld1.32    {q2, q3},   [%[scale]]! \n\t"
      "vld1.32    {q10, q11}, [%[bias]]!  \n\t"
      "vmla.f32   q10,  q0,   q2          \n\t"
      "vmla.f32   q11,  q1,   q3          \n\t"
      "vst1.32    {q10, q11}, [%[C]]!     \n\t"

      "vld1.32    {q4, q5},   [%[c]]!     \n\t"
      "vld1.32    {q6, q7},   [%[scale]]! \n\t"
      "vld1.32    {q12, q13}, [%[bias]]!  \n\t"
      "vmla.f32   q12,  q4,   q6          \n\t"
      "vmla.f32   q13,  q5,   q7          \n\t"
      "vst1.32    {q12, q13}, [%[C]]!     \n\t"

      "subs       %[nc1],   %[nc1],   #1  \n\t"
      "bge        loop_nc1_%=             \n\t"
      "end_nc1_%=:                        \n\t"

      "subs       %[nc2],   %[nc2],   #1  \n\t"
      "blt        end_nc2_%=              \n\t"
      "loop_nc2_%=:                       \n\t"

      "vld1.32    {q0},   [%[c]]!         \n\t"
      "vld1.32    {q1},   [%[scale]]!     \n\t"
      "vld1.32    {q10},  [%[bias]]!      \n\t"
      "vmla.f32   q10,    q0,   q1        \n\t"
      "vst1.32    {q10},  [%[C]]!         \n\t"

      "subs       %[nc2],   %[nc2],   #1  \n\t"
      "bge        loop_nc2_%=             \n\t"
      "end_nc2_%=:                        \n\t"

      "cmp        %[nc3],    #16          \n\t"
      "beq        end_nc3_%=              \n\t"

      "sub        %[c],     %[c],   %[nc3]      \n\t"
      "sub        %[scale], %[scale],  %[nc3]   \n\t"
      "sub        %[bias],  %[bias],   %[nc3]   \n\t"
      "sub        %[C],     %[C],   %[nc3]      \n\t"

      "vld1.32    {q0},   [%[c]]!         \n\t"
      "vld1.32    {q1},   [%[scale]]!     \n\t"
      "vld1.32    {q10},  [%[bias]]!      \n\t"
      "vmla.f32   q10,    q0,   q1        \n\t"
      "vst1.32    {q10},  [%[C]]!         \n\t"
      "end_nc3_%=:                        \n\t"

      :
      : [C] "r"(C), [c] "r"(c), [nc1] "r"(nc1), [nc2] "r"(nc2), [nc3] "r"(nc3),
        [scale] "r"(scale), [bias] "r"(bias)
      : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q10", "q11",
        "q12", "q13");
}

// C = A * B, batchnorm(C), relu(C)
void Gemm::VecWriteWithBnRelu(int n, float *c, float *C, int ldc, float *scale,
                              float *bias) {
  int nc1 = n / 16;
  int _nc1 = n % 16;
  int nc2 = _nc1 / 4;
  int nc3 = 16 - 4 * (_nc1 % 4);

  asm volatile(
      "vmov.f32   q14,      #0.0          \n\t"
      "subs       %[nc1],   %[nc1],   #1  \n\t"
      "blt        end_nc1_%=              \n\t"
      "loop_nc1_%=:                       \n\t"

      "vld1.32    {q0, q1},   [%[c]]!     \n\t"
      "vld1.32    {q2, q3},   [%[scale]]! \n\t"
      "vld1.32    {q10, q11}, [%[bias]]!  \n\t"
      "vmla.f32   q10,  q0,   q2          \n\t"
      "vmla.f32   q11,  q1,   q3          \n\t"
      "vmax.f32   q10,  q10,  q14         \n\t"
      "vmax.f32   q11,  q11,  q14         \n\t"
      "vst1.32    {q10, q11}, [%[C]]!     \n\t"

      "vld1.32    {q4, q5},   [%[c]]!     \n\t"
      "vld1.32    {q6, q7},   [%[scale]]! \n\t"
      "vld1.32    {q12, q13}, [%[bias]]!  \n\t"
      "vmla.f32   q12,  q4,   q6          \n\t"
      "vmla.f32   q13,  q5,   q7          \n\t"
      "vmax.f32   q12,  q12,  q14         \n\t"
      "vmax.f32   q13,  q13,  q14         \n\t"
      "vst1.32    {q12, q13}, [%[C]]!     \n\t"

      "subs       %[nc1],   %[nc1],   #1  \n\t"
      "bge        loop_nc1_%=             \n\t"
      "end_nc1_%=:                        \n\t"

      "subs       %[nc2],   %[nc2],   #1  \n\t"
      "blt        end_nc2_%=              \n\t"
      "loop_nc2_%=:                       \n\t"

      "vld1.32    {q0},   [%[c]]!         \n\t"
      "vld1.32    {q1},   [%[scale]]!     \n\t"
      "vld1.32    {q10},  [%[bias]]!      \n\t"
      "vmla.f32   q10,    q0,   q1        \n\t"
      "vmax.f32   q10,    q10,  q14       \n\t"
      "vst1.32    {q10},  [%[C]]!         \n\t"

      "subs       %[nc2],   %[nc2],   #1  \n\t"
      "bge        loop_nc2_%=             \n\t"
      "end_nc2_%=:                        \n\t"

      "cmp        %[nc3],    #16          \n\t"
      "beq        end_nc3_%=              \n\t"

      "sub        %[c],     %[c],   %[nc3]      \n\t"
      "sub        %[scale], %[scale],  %[nc3]   \n\t"
      "sub        %[bias],  %[bias],   %[nc3]   \n\t"
      "sub        %[C],     %[C],   %[nc3]      \n\t"

      "vld1.32    {q0},   [%[c]]!         \n\t"
      "vld1.32    {q1},   [%[scale]]!     \n\t"
      "vld1.32    {q10},  [%[bias]]!      \n\t"
      "vmla.f32   q10,    q0,   q1        \n\t"
      "vmax.f32   q10,    q10,  q14       \n\t"
      "vst1.32    {q10},  [%[C]]!         \n\t"
      "end_nc3_%=:                        \n\t"

      :
      : [C] "r"(C), [c] "r"(c), [nc1] "r"(nc1), [nc2] "r"(nc2), [nc3] "r"(nc3),
        [scale] "r"(scale), [bias] "r"(bias)
      : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q10", "q11",
        "q12", "q13", "q14");
}

#endif  // __aarch64__
#endif  // __ARM_NEON

// 32位 float 矩阵乘法
void Gemm::Sgemm(int m, int n, int k, float alpha, const float *A, int lda,
                 const float *B, int ldb, float beta, float *C, int ldc,
                 bool relu, float *bias) {
  // L1 data cache is 32 kib (Per Contex-A57, Contex-A72, Contex-A73)
  // L2 cache is 0.5~4 Mib (Contex-A72 cluster)
  int L1 = 32 * 1024;
  int L2 = 512 * 1024;

  KC = k;
  MC = L1 / (KC * sizeof(float));
  NC = L2 / (KC * sizeof(float));

  // make sure MC is multiple of MR, and NC is multiple of NR
  if (MC == 0) {
    MC = MR;
  } else {
    int mblock_num = (m + MC - 1) / MC;
    MC = (m + mblock_num - 1) / mblock_num;
    MC = (MC + MR - 1) / MR * MR;
  }
  //  DLOG << "mblock_num = " << mblock_num << ", MC = " << MC << "\n";
  if (NC == 0) {
    NC = NR;
  } else {
    int nblock_num = (n + NC - 1) / NC;
    NC = (n + nblock_num - 1) / nblock_num;
    NC = (NC + NR - 1) / NR * NR;
  }
  //  DLOG << "nblock_num = " << nblock_num << ", NC = " << NC << "\n";

  packedA = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * MC * KC));
  packedB = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * KC * NC));
  packedC = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * MC * NC));

  int mc, nc;
  for (int j = 0; j < n; j += NC) {
    nc = s_min(n - j, NC);
#if __aarch64__
    // PackMatrixB_12c(KC, nc, nc % NR, &B(0, j), ldb, packedB);
    PackMatrixB_16c(KC, nc, nc % NR, &B(0, j), ldb, packedB, false);
#else
    PackMatrixB_8c(KC, nc, nc % NR, &B(0, j), ldb, packedB, false);
#endif
    for (int i = 0; i < m; i += MC) {
      mc = s_min(m - i, MC);
#if __aarch64__
      PackMatrixA_6r(mc, KC, mc % MR, &A(i, 0), lda, packedA, false);
      // PackMatrixA_8r(mc, KC, mc % MR, &A(i, 0), lda, packedA);
#else
      PackMatrixA_6r(mc, KC, mc % MR, &A(i, 0), lda, packedA, false);
#endif
      if (bias == nullptr) {
        InnerKernelWithBias(mc, nc, alpha, packedA, packedB, beta, packedC,
                            &C(i, j), ldc, relu, nullptr);
      } else {
        InnerKernelWithBias(mc, nc, alpha, packedA, packedB, beta, packedC,
                            &C(i, j), ldc, relu, bias + i);
      }
    }
  }

  paddle_mobile::memory::Free(packedA);
  paddle_mobile::memory::Free(packedB);
  paddle_mobile::memory::Free(packedC);
}

void Gemm::SgemmWithBn(int m, int n, int k, float alpha, const float *A,
                       int lda, const float *B, int ldb, float beta, float *C,
                       int ldc, bool relu, float *new_scale, float *new_bias,
                       float *bias) {
  // L1 data cache is 32 kib (Per Contex-A57, Contex-A72, Contex-A73)
  // L2 cache is 0.5~4 Mib (Contex-A72 cluster)
  int L1 = 32 * 1024;
  int L2 = 512 * 1024;

  KC = k;
  MC = L1 / (KC * sizeof(float));
  NC = L2 / (KC * sizeof(float));

  // make sure MC is multiple of MR, and NC is multiple of NR
  if (MC == 0) {
    MC = MR;
  } else {
    int mblock_num = (m + MC - 1) / MC;
    MC = (m + mblock_num - 1) / mblock_num;
    MC = (MC + MR - 1) / MR * MR;
  }
  //  DLOG << "mblock_num = " << mblock_num << ", MC = " << MC << "\n";
  if (NC == 0) {
    NC = NR;
  } else {
    int nblock_num = (n + NC - 1) / NC;
    NC = (n + nblock_num - 1) / nblock_num;
    NC = (NC + NR - 1) / NR * NR;
  }
  //  DLOG << "nblock_num = " << nblock_num << ", NC = " << NC << "\n";

  packedA = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * MC * KC));
  packedB = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * KC * NC));
  packedC = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * MC * NC));

  int mc, nc;
  for (int j = 0; j < n; j += NC) {
    nc = s_min(n - j, NC);
#if __aarch64__
    // PackMatrixB_12c(KC, nc, nc % NR, &B(0, j), ldb, packedB);
    PackMatrixB_16c(KC, nc, nc % NR, &B(0, j), ldb, packedB, false);
#else
    PackMatrixB_8c(KC, nc, nc % NR, &B(0, j), ldb, packedB, false);
#endif
    for (int i = 0; i < m; i += MC) {
      mc = s_min(m - i, MC);
#if __aarch64__
      PackMatrixA_6r(mc, KC, mc % MR, &A(i, 0), lda, packedA, false);
      // PackMatrixA_8r(mc, KC, mc % MR, &A(i, 0), lda, packedA);
#else
      PackMatrixA_6r(mc, KC, mc % MR, &A(i, 0), lda, packedA, false);
#endif
      if (bias == nullptr) {
        InnerKernelWithBn(mc, nc, alpha, packedA, packedB, beta, packedC,
                          &C(i, j), ldc, relu, new_scale + i, new_bias + i);
      } else {
        InnerKernelWithBnAdd(mc, nc, alpha, packedA, packedB, beta, packedC,
                             &C(i, j), ldc, relu, new_scale + i, new_bias + i,
                             bias + i * ldc + j);
      }
    }
  }

  paddle_mobile::memory::Free(packedA);
  paddle_mobile::memory::Free(packedB);
  paddle_mobile::memory::Free(packedC);
}

void Gemm::SgemmWithPRelu(int m, int n, int k, const float *A, int lda,
                          const float *B, int ldb, float *C, int ldc, float *p,
                          std::string mode, float *bias, float *bias1) {
  // L1 data cache is 32 kib (Per Contex-A57, Contex-A72, Contex-A73)
  // L2 cache is 0.5~4 Mib (Contex-A72 cluster)
  int L1 = 32 * 1024;
  int L2 = 0.5 * 1024 * 1024;

  KC = k;
  MC = L1 / (KC * sizeof(float));
  NC = L2 / (KC * sizeof(float));

  // make sure MC is multiple of MR, and NC is multiple of NR
  if (MC == 0) {
    MC = MR;
  } else {
    int mblock_num = (m + MC - 1) / MC;
    MC = (m + mblock_num - 1) / mblock_num;
    MC = (MC + MR - 1) / MR * MR;
  }
  //  DLOG << "mblock_num = " << mblock_num << ", MC = " << MC << "\n";
  if (NC == 0) {
    NC = NR;
  } else {
    int nblock_num = (n + NC - 1) / NC;
    NC = (n + nblock_num - 1) / nblock_num;
    NC = (NC + NR - 1) / NR * NR;
  }
  //  DLOG << "nblock_num = " << nblock_num << ", NC = " << NC << "\n";

  packedA = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * MC * KC));
  packedB = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * KC * NC));
  packedC = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * MC * NC));

  int mc, nc;
  for (int j = 0; j < n; j += NC) {
    nc = s_min(n - j, NC);
#if __aarch64__
    // PackMatrixB_12c(KC, nc, nc % NR, &B(0, j), ldb, packedB);
    PackMatrixB_16c(KC, nc, nc % NR, &B(0, j), ldb, packedB, false);
#else
    PackMatrixB_8c(KC, nc, nc % NR, &B(0, j), ldb, packedB, false);
#endif
    for (int i = 0; i < m; i += MC) {
      mc = s_min(m - i, MC);
#if __aarch64__
      PackMatrixA_6r(mc, KC, mc % MR, &A(i, 0), lda, packedA, false);
      // PackMatrixA_8r(mc, KC, mc % MR, &A(i, 0), lda, packedA);
#else
      PackMatrixA_6r(mc, KC, mc % MR, &A(i, 0), lda, packedA, false);
#endif
      if (bias1 == nullptr) {
        InnerKernelWithPRelu(mc, nc, packedA, packedB, packedC, &C(i, j), ldc,
                             p + i, mode, bias + i, nullptr);
      } else {
        InnerKernelWithPRelu(mc, nc, packedA, packedB, packedC, &C(i, j), ldc,
                             p + i, mode, bias + i, bias1 + i * ldc + j);
      }
    }
  }

  paddle_mobile::memory::Free(packedA);
  paddle_mobile::memory::Free(packedB);
  paddle_mobile::memory::Free(packedC);
}

// 32位 float 矩阵乘法
void Gemm::Sgemm_omp(int m, int n, int k, float alpha, const float *A, int lda,
                     const float *B, int ldb, float beta, float *C, int ldc,
                     bool relu, float *bias) {
#ifndef __aarch64__
  if (m == 1 && bias == nullptr) {
    return VectorKernel(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, relu);
  }
#endif  // __aarch64__
#ifdef _OPENMP
  int max_threads = omp_get_max_threads();
#else
  int max_threads = 1;
#endif

  //  int L1 = 64 / max_threads * 1024;
  int L = (max_threads > 2) ? 64 : 32;
  int L1 = L / max_threads * 1024;
  KC = k;
  if (m > n) {
    // 对 A 分块
    MC = L1 / (KC * sizeof(float));
    if (MC == 0) {
      MC = MR;
    } else {
      int mblock_num = (m + MC - 1) / MC;
      MC = (m + mblock_num - 1) / mblock_num;
      MC = (MC + MR - 1) / MR * MR;
    }
    // 补齐 B
    NC = (n + NR - 1) / NR * NR;

#if __aarch64__
    procPackA = &Gemm::PackMatrixA_6r;
    procPackB = &Gemm::PackMatrixB_16c;
    procAddDot = &Gemm::AddDot6x16;
#else
    procPackA = &Gemm::PackMatrixA_6r;
    procPackB = &Gemm::PackMatrixB_8c;
    procAddDot = &Gemm::AddDot6x8;
#endif

    packedB = static_cast<float *>(
        paddle_mobile::memory::Alloc(sizeof(float) * KC * NC));
    (*this.*procPackB)(KC, n, n % NR, B, ldb, packedB, true);
    packedA = static_cast<float *>(
        paddle_mobile::memory::Alloc(sizeof(float) * MC * KC * max_threads));
  } else {
    // 对 B 分块
    NC = L1 / (KC * sizeof(float));
    if (NC == 0) {
      NC = NR;
    } else {
      int nblock_num = (n + NC - 1) / NC;
      NC = (n + nblock_num - 1) / nblock_num;
      NC = (NC + NR - 1) / NR * NR;
    }
    // 补齐 A
    MC = (m + MR - 1) / MR * MR;

#if __aarch64__
    procPackA = &Gemm::PackMatrixA_6r;
    procPackB = &Gemm::PackMatrixB_16c;
    procAddDot = &Gemm::AddDot6x16;
#else

    procPackA = &Gemm::PackMatrixA_6r;
    procPackB = &Gemm::PackMatrixB_8c;
    procAddDot = &Gemm::AddDot6x8;
#endif

    packedA = static_cast<float *>(
        paddle_mobile::memory::Alloc(sizeof(float) * MC * KC));
    (*this.*procPackA)(m, KC, m % MR, A, lda, packedA, true);
    packedB = static_cast<float *>(
        paddle_mobile::memory::Alloc(sizeof(float) * KC * NC * max_threads));
  }
  packedC = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * MC * NC * max_threads));

  if (m > n) {
#pragma omp parallel for num_threads(framework::threads())
    for (int i = 0; i < m; i += MC) {
#ifdef _OPENMP
      int local_threads = omp_get_thread_num();
#else
      int local_threads = 0;
#endif

      int mc;
      mc = s_min(m - i, MC);
      float *local_A = packedA + MC * KC * local_threads;
      float *local_C = packedC + MC * NC * local_threads;
      (*this.*procPackA)(mc, KC, mc % MR, &A(i, 0), lda, local_A, false);
      if (bias == nullptr) {
        InnerKernelWithBias(mc, n, alpha, local_A, packedB, beta, local_C,
                            &C(i, 0), ldc, relu, nullptr);
      } else {
        InnerKernelWithBias(mc, n, alpha, local_A, packedB, beta, local_C,
                            &C(i, 0), ldc, relu, bias + i);
      }
    }
  } else {
#pragma omp parallel for num_threads(framework::threads())
    for (int j = 0; j < n; j += NC) {
#ifdef _OPENMP
      int local_threads = omp_get_thread_num();
#else
      int local_threads = 0;
#endif

      int nc;
      nc = s_min(n - j, NC);
      float *local_B = packedB + KC * NC * local_threads;
      float *local_C = packedC + MC * NC * local_threads;
      (*this.*procPackB)(KC, nc, nc % NR, &B(0, j), ldb, local_B, false);
      InnerKernelWithBias(m, nc, alpha, packedA, local_B, beta, local_C,
                          &C(0, j), ldc, relu, bias);
    }
  }

  paddle_mobile::memory::Free(packedA);
  paddle_mobile::memory::Free(packedB);
  paddle_mobile::memory::Free(packedC);
}

void Gemm::SgemmWithBn_omp(int m, int n, int k, float alpha, const float *A,
                           int lda, const float *B, int ldb, float beta,
                           float *C, int ldc, bool relu, float *new_scale,
                           float *new_bias, float *bias) {
#ifdef _OPENMP
  int max_threads = omp_get_max_threads();
#else
  int max_threads = 1;
#endif

  int L1 = 64 / max_threads * 1024;
  KC = k;
  if (m > n) {
    // 对 A 分块
    MC = L1 / (KC * sizeof(float));
    if (MC == 0) {
      MC = MR;
    } else {
      int mblock_num = (m + MC - 1) / MC;
      MC = (m + mblock_num - 1) / mblock_num;
      MC = (MC + MR - 1) / MR * MR;
    }
    // 补齐 B
    NC = (n + NR - 1) / NR * NR;

#if __aarch64__
    procPackA = &Gemm::PackMatrixA_6r;
    procPackB = &Gemm::PackMatrixB_16c;
    procAddDot = &Gemm::AddDot6x16;
#else
    procPackA = &Gemm::PackMatrixA_6r;
    procPackB = &Gemm::PackMatrixB_8c;
    procAddDot = &Gemm::AddDot6x8;
#endif

    packedB = static_cast<float *>(
        paddle_mobile::memory::Alloc(sizeof(float) * KC * NC));
    (*this.*procPackB)(KC, n, n % NR, B, ldb, packedB, true);
    packedA = static_cast<float *>(
        paddle_mobile::memory::Alloc(sizeof(float) * MC * KC * max_threads));
  } else {
    // 对 B 分块
    NC = L1 / (KC * sizeof(float));
    if (NC == 0) {
      NC = NR;
    } else {
      int nblock_num = (n + NC - 1) / NC;
      NC = (n + nblock_num - 1) / nblock_num;
      NC = (NC + NR - 1) / NR * NR;
    }
    // 补齐 A
    MC = (m + MR - 1) / MR * MR;

#if __aarch64__
    procPackA = &Gemm::PackMatrixA_6r;
    procPackB = &Gemm::PackMatrixB_16c;
    procAddDot = &Gemm::AddDot6x16;
#else
    procPackA = &Gemm::PackMatrixA_6r;
    procPackB = &Gemm::PackMatrixB_8c;
    procAddDot = &Gemm::AddDot6x8;
#endif

    packedA = static_cast<float *>(
        paddle_mobile::memory::Alloc(sizeof(float) * MC * KC));
    (*this.*procPackA)(m, KC, m % MR, A, lda, packedA, true);
    packedB = static_cast<float *>(
        paddle_mobile::memory::Alloc(sizeof(float) * KC * NC * max_threads));
  }
  packedC = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * MC * NC * max_threads));

  if (m > n) {
#pragma omp parallel for num_threads(framework::threads())
    for (int i = 0; i < m; i += MC) {
#ifdef _OPENMP
      int local_threads = omp_get_thread_num();
#else
      int local_threads = 0;
#endif

      int mc;
      mc = s_min(m - i, MC);
      float *local_A = packedA + MC * KC * local_threads;
      float *local_C = packedC + MC * NC * local_threads;
      (*this.*procPackA)(mc, KC, mc % MR, &A(i, 0), lda, local_A, false);
      if (bias == nullptr) {
        InnerKernelWithBn(mc, n, alpha, local_A, packedB, beta, local_C,
                          &C(i, 0), ldc, relu, new_scale + i, new_bias + i);
      } else {
        InnerKernelWithBnAdd(mc, n, alpha, local_A, packedB, beta, local_C,
                             &C(i, 0), ldc, relu, new_scale + i, new_bias + i,
                             bias + i * ldc);
      }
    }
  } else {
#pragma omp parallel for num_threads(framework::threads())
    for (int j = 0; j < n; j += NC) {
#ifdef _OPENMP
      int local_threads = omp_get_thread_num();
#else
      int local_threads = 0;
#endif

      int nc;
      nc = s_min(n - j, NC);
      float *local_B = packedB + KC * NC * local_threads;
      float *local_C = packedC + MC * NC * local_threads;
      (*this.*procPackB)(KC, nc, nc % NR, &B(0, j), ldb, local_B, false);
      if (bias == nullptr) {
        InnerKernelWithBn(m, nc, alpha, packedA, local_B, beta, local_C,
                          &C(0, j), ldc, relu, new_scale, new_bias);
      } else {
        InnerKernelWithBnAdd(m, nc, alpha, packedA, local_B, beta, local_C,
                             &C(0, j), ldc, relu, new_scale, new_bias,
                             bias + j);
      }
    }
  }

  paddle_mobile::memory::Free(packedA);
  paddle_mobile::memory::Free(packedB);
  paddle_mobile::memory::Free(packedC);
}

void Gemm::SgemmWithPRelu_omp(int m, int n, int k, const float *A, int lda,
                              const float *B, int ldb, float *C, int ldc,
                              float *p, std::string mode, float *bias,
                              float *bias1) {
#ifdef _OPENMP
  int max_threads = omp_get_max_threads();
#else
  int max_threads = 1;
#endif

  int L1 = 8 * 1024;
  KC = k;
  if (m > n) {
    // 对 A 分块
    MC = L1 / (KC * sizeof(float));
    if (MC == 0) {
      MC = MR;
    } else {
      int mblock_num = (m + MC - 1) / MC;
      MC = (m + mblock_num - 1) / mblock_num;
      MC = (MC + MR - 1) / MR * MR;
    }
    // 补齐 B
    NC = (n + NR - 1) / NR * NR;

#if __aarch64__
    procPackA = &Gemm::PackMatrixA_6r;
    procPackB = &Gemm::PackMatrixB_16c;
    procAddDot = &Gemm::AddDot6x16;
#else
    procPackA = &Gemm::PackMatrixA_6r;
    procPackB = &Gemm::PackMatrixB_8c;
    procAddDot = &Gemm::AddDot6x8;
#endif

    packedB = static_cast<float *>(
        paddle_mobile::memory::Alloc(sizeof(float) * KC * NC));
    (*this.*procPackB)(KC, n, n % NR, B, ldb, packedB, true);
    packedA = static_cast<float *>(
        paddle_mobile::memory::Alloc(sizeof(float) * MC * KC * max_threads));
  } else {
    // 对 B 分块
    NC = L1 / (KC * sizeof(float));
    if (NC == 0) {
      NC = NR;
    } else {
      int nblock_num = (n + NC - 1) / NC;
      NC = (n + nblock_num - 1) / nblock_num;
      NC = (NC + NR - 1) / NR * NR;
    }
    // 补齐 A
    MC = (m + MR - 1) / MR * MR;

#if __aarch64__
    procPackA = &Gemm::PackMatrixA_6r;
    procPackB = &Gemm::PackMatrixB_16c;
    procAddDot = &Gemm::AddDot6x16;
#else
    procPackA = &Gemm::PackMatrixA_6r;
    procPackB = &Gemm::PackMatrixB_8c;
    procAddDot = &Gemm::AddDot6x8;
#endif

    packedA = static_cast<float *>(
        paddle_mobile::memory::Alloc(sizeof(float) * MC * KC));
    (*this.*procPackA)(m, KC, m % MR, A, lda, packedA, true);
    packedB = static_cast<float *>(
        paddle_mobile::memory::Alloc(sizeof(float) * KC * NC * max_threads));
  }
  packedC = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * MC * NC * max_threads));

  if (m > n) {
#pragma omp parallel for num_threads(framework::threads())
    for (int i = 0; i < m; i += MC) {
#ifdef _OPENMP
      int local_threads = omp_get_thread_num();
#else
      int local_threads = 0;
#endif

      int mc;
      mc = s_min(m - i, MC);
      float *local_A = packedA + MC * KC * local_threads;
      float *local_C = packedC + MC * NC * local_threads;
      (*this.*procPackA)(mc, KC, mc % MR, &A(i, 0), lda, local_A, false);
      if (bias1 == nullptr) {
        InnerKernelWithPRelu(mc, n, local_A, packedB, local_C, &C(i, 0), ldc,
                             p + i, mode, bias + i, nullptr);
      } else {
        InnerKernelWithPRelu(mc, n, local_A, packedB, local_C, &C(i, 0), ldc,
                             p + i, mode, bias + i, bias1 + i * ldc);
      }
    }
  } else {
#pragma omp parallel for num_threads(framework::threads())
    for (int j = 0; j < n; j += NC) {
#ifdef _OPENMP
      int local_threads = omp_get_thread_num();
#else
      int local_threads = 0;
#endif

      int nc;
      nc = s_min(n - j, NC);
      float *local_B = packedB + KC * NC * local_threads;
      float *local_C = packedC + MC * NC * local_threads;
      (*this.*procPackB)(KC, nc, nc % NR, &B(0, j), ldb, local_B, false);
      if (bias1 == nullptr) {
        InnerKernelWithPRelu(m, nc, packedA, local_B, local_C, &C(0, j), ldc, p,
                             mode, bias, nullptr);
      } else {
        InnerKernelWithPRelu(m, nc, packedA, local_B, local_C, &C(0, j), ldc, p,
                             mode, bias, bias1 + j);
      }
    }
  }

  paddle_mobile::memory::Free(packedA);
  paddle_mobile::memory::Free(packedB);
  paddle_mobile::memory::Free(packedC);
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
