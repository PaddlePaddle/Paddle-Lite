/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#if defined(__ARM_NEON__) || defined(__ARM_NEON)

#include <arm_neon.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "operators/math/math.h"

namespace paddle_mobile {
namespace operators {
namespace math {

void pack_lhs_6r(const int m, const int k, const float *A, const int lda,
                 float *output, const bool unroll) {
  uint32_t mask[8] = {0, 1, 2, 3, 4, 5, 4, 5};
  int remain_k = k & 0x3;
  uint32x4_t vzero = vdupq_n_u32(0);
  uint32x4_t vmask1 = vcltq_u32(vld1q_u32(mask), vdupq_n_u32(remain_k));

  #pragma omp parallel for if (unroll)
  for (int i = 0; i < m - 5; i += 6) {
    const float *a0 = A + i * lda;
    const float *a1 = A + (i + 1) * lda;
    const float *a2 = A + (i + 2) * lda;
    const float *a3 = A + (i + 3) * lda;
    const float *a4 = A + (i + 4) * lda;
    const float *a5 = A + (i + 5) * lda;
    float *out_ptr = output + i * k;

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

      _d0 = vandq_f32_u32(_d0, vmask1);
      _d1 = vandq_f32_u32(_d1, vmask1);
      _d2 = vandq_f32_u32(_d2, vmask1);
      _d3 = vandq_f32_u32(_d3, vmask1);
      _d4 = vandq_f32_u32(_d4, vmask1);
      _d5 = vandq_f32_u32(_d5, vmask1);

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
    float *out_ptr = output + remain_m_start * k;

    uint32x4_t vmask2 = vcltq_u32(vld1q_u32(mask), vdupq_n_u32(remain_m));
    uint32x4_t vmask3 = vcltq_u32(vld1q_u32(mask + 4), vdupq_n_u32(remain_m));
    const float zerobuff[4] = {0.f, 0.f, 0.f, 0.f};

    int lk = 0;
    for (; lk < k - 3; lk += 4) {
      switch (remain_m) {
        case 1:
          a1 = zerobuff;
        case 2:
          a2 = zerobuff;
        case 3:
          a3 = zerobuff;
        case 4:
          a4 = zerobuff;
        case 5:
          a5 = zerobuff;
        default:
          break;
      }
#if __aarch64__
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
      _d2 = vcombine_f32(vget_high_f32(_q0.val[0]), vget_high_f32(_q1.val[0]));
      _d3 = vcombine_f32(vget_high_f32(_q0.val[1]), vget_high_f32(_q1.val[1]));

      _d0 = vandq_f32_u32(_d0, vmask2);
      _d1 = vandq_f32_u32(_d1, vmask2);
      _d2 = vandq_f32_u32(_d2, vmask2);
      _d3 = vandq_f32_u32(_d3, vmask2);
      _d4 = vandq_f32_u32(_q3.val[0], vmask3);
      _d5 = vandq_f32_u32(_q3.val[1], vmask3);

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
#else
      asm volatile(
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
          : [out] "+r"(out_ptr), [a0] "+r"(a0), [a1] "+r"(a1), [a2] "+r"(a2),
            [a3] "+r"(a3), [a4] "+r"(a4), [a5] "+r"(a5)
          : [vmask2] "w"(vmask2), [vmask3] "w"(vmask3), [vzero] "w"(vzero)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5");
#endif
    }
    // remain k
    switch (remain_m) {
      case 1:
        a1 = zerobuff;
      case 2:
        a2 = zerobuff;
      case 3:
        a3 = zerobuff;
      case 4:
        a4 = zerobuff;
      case 5:
        a5 = zerobuff;
      default:
        break;
    }
    for (; lk < k; ++lk) {
      *out_ptr++ = *a0++;
      *out_ptr++ = *a1++;
      *out_ptr++ = *a2++;
      *out_ptr++ = *a3++;
      *out_ptr++ = *a4++;
      *out_ptr++ = *a5++;
    }
  }
}

#if __aarch64__
void pack_rhs_16c(int k, int n, const float *B, int ldb, float *output,
                  const bool unroll) {
  uint32_t mask[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint32_t remain_n = n & 0x7;
  float32x4_t vzero = vdupq_n_f32(0.f);
  uint32x4_t vmask1 = vcltq_u32(vld1q_u32(mask), vdupq_n_u32(remain_n));
  uint32x4_t vmask2 = vcltq_u32(vld1q_u32(mask + 4), vdupq_n_u32(remain_n));

  #pragma omp parallel for if (unroll)
  for (int i = 0; i < k - 3; i += 4) {
    const float *b0 = B + i * ldb;
    const float *b1 = b0 + ldb;
    const float *b2 = b1 + ldb;
    const float *b3 = b2 + ldb;
    int j = 0;
    asm volatile(
        "prfm   pldl1keep,       [%[b0]]            \n"
        "prfm   pldl1keep,       [%[b1]]            \n"
        "prfm   pldl1keep,       [%[b2]]            \n"
        "prfm   pldl1keep,       [%[b3]]            \n"
        :
        : [b0] "r"(b0), [b1] "r"(b1), [b2] "r"(b2), [b3] "r"(b3));

    for (; j < n - 15; j += 16) {
      float *out_ptr0 = output + j * k + 16 * i;
      asm volatile(
          "ld1    {v0.4s, v1.4s, v2.4s, v3.4s},  [%[b0]], #64  \n"
          "ld1    {v4.4s, v5.4s, v6.4s, v7.4s},  [%[b1]], #64  \n"
          "st1    {v0.4s, v1.4s, v2.4s, v3.4s},  [%[out_ptr0]], #64 \n"
          "st1    {v4.4s, v5.4s, v6.4s, v7.4s},  [%[out_ptr0]], #64 \n"

          "ld1    {v0.4s, v1.4s, v2.4s, v3.4s},  [%[b2]], #64  \n"
          "ld1    {v4.4s, v5.4s, v6.4s, v7.4s},  [%[b3]], #64  \n"
          "st1    {v0.4s, v1.4s, v2.4s, v3.4s},  [%[out_ptr0]], #64 \n"
          "st1    {v4.4s, v5.4s, v6.4s, v7.4s},  [%[out_ptr0]], #64 \n"
          : [out_ptr0] "+r"(out_ptr0), [b0] "+r"(b0), [b1] "+r"(b1),
            [b2] "+r"(b2), [b3] "+r"(b3)
          :
          : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
    }
    for (; j < n - 7; j += 8) {
      float *out_ptr0 = output + (j & 0xFFF0) * k + 16 * i + (j & 0xF);
      int step = 64;
      asm volatile(
          "ld1    {v0.4s, v1.4s},  [%[b0]], #32  \n"
          "ld1    {v2.4s, v3.4s},  [%[b1]], #32  \n"
          "ld1    {v4.4s, v5.4s},  [%[b2]], #32  \n"
          "ld1    {v6.4s, v7.4s},  [%[b3]], #32  \n"

          "st1    {v0.4s, v1.4s},  [%[out_ptr0]], %[step] \n"
          "st1    {v2.4s, v3.4s},  [%[out_ptr0]], %[step] \n"
          "st1    {v4.4s, v5.4s},  [%[out_ptr0]], %[step] \n"
          "st1    {v6.4s, v7.4s},  [%[out_ptr0]], %[step] \n"
          : [out_ptr0] "+r"(out_ptr0), [b0] "+r"(b0), [b1] "+r"(b1),
            [b2] "+r"(b2), [b3] "+r"(b3)
          : [step] "r"(step)
          : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
    }
    if (j < n) {
      float *out_ptr0 = output + (j & 0xFFF0) * k + 16 * i + (j & 0xF);
      int step = 64;
      asm volatile(
          "ld1    {v0.4s, v1.4s},  [%[b0]]         \n"
          "ld1    {v2.4s, v3.4s},  [%[b1]]         \n"
          "ld1    {v4.4s, v5.4s},  [%[b2]]         \n"
          "ld1    {v6.4s, v7.4s},  [%[b3]]         \n"

          "and    v0.16b, v0.16b, %[vmask1].16b    \n"
          "and    v1.16b, v1.16b, %[vmask2].16b    \n"
          "and    v2.16b, v2.16b, %[vmask1].16b    \n"
          "and    v3.16b, v3.16b, %[vmask2].16b    \n"
          "and    v4.16b, v4.16b, %[vmask1].16b    \n"
          "and    v5.16b, v5.16b, %[vmask2].16b    \n"
          "and    v6.16b, v6.16b, %[vmask1].16b    \n"
          "and    v7.16b, v7.16b, %[vmask2].16b    \n"

          "st1    {v0.4s, v1.4s},  [%[out_ptr0]], %[step]  \n"
          "st1    {v2.4s, v3.4s},  [%[out_ptr0]], %[step]  \n"
          "st1    {v4.4s, v5.4s},  [%[out_ptr0]], %[step]  \n"
          "st1    {v6.4s, v7.4s},  [%[out_ptr0]], %[step]  \n"
          : [out_ptr0] "+r"(out_ptr0)
          : [vmask1] "w"(vmask1), [vmask2] "w"(vmask2), [b0] "r"(b0),
            [b1] "r"(b1), [b2] "r"(b2), [b3] "r"(b3), [step] "r"(step)
          : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
      j += 8;
    }

    if (j & 0xf) {
      float *out_ptr0 = output + (j & 0xFFF0) * k + 16 * i + (j & 0xF);
      vst1q_f32(out_ptr0, vzero);
      vst1q_f32(out_ptr0 + 4, vzero);
      out_ptr0 += 16;
      vst1q_f32(out_ptr0, vzero);
      vst1q_f32(out_ptr0 + 4, vzero);
      out_ptr0 += 16;
      vst1q_f32(out_ptr0, vzero);
      vst1q_f32(out_ptr0 + 4, vzero);
      out_ptr0 += 16;
      vst1q_f32(out_ptr0, vzero);
      vst1q_f32(out_ptr0 + 4, vzero);
    }
  }
  // remain k
  for (int i = (k & 0xFFFC); i < k; ++i) {
    const float *b0 = B + i * ldb;
    int j = 0;
    asm volatile("prfm   pldl1keep,       [%[b0]]            \n"
                 :
                 : [b0] "r"(b0));

    for (; j < n - 15; j += 16) {
      float *out_ptr0 = output + j * k + 16 * i;
      asm volatile(
          "ld1    {v0.4s, v1.4s, v2.4s, v3.4s},     [%[b0]], #64  \n"
          "st1    {v0.4s, v1.4s, v2.4s, v3.4s},     [%[out_ptr0]], #64 \n"
          : [out_ptr0] "+r"(out_ptr0), [b0] "+r"(b0)
          :
          : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
    }
    for (; j < n - 7; j += 8) {
      float *out_ptr0 = output + (j & 0xFFF0) * k + 16 * i + (j & 0xF);
      int step = 64;
      asm volatile(
          "ld1    {v0.4s, v1.4s},  [%[b0]], #32  \n"
          "st1    {v0.4s, v1.4s},  [%[out_ptr0]], %[step] \n"
          : [out_ptr0] "+r"(out_ptr0), [b0] "+r"(b0)
          : [step] "r"(step)
          : "memory", "v0", "v1");
    }
    if (j < n) {
      float *out_ptr0 = output + (j & 0xFFF0) * k + 16 * i + (j & 0xF);
      asm volatile(
          "ld1    {v0.4s, v1.4s},  [%[b0]]          \n"
          "and    v0.16b, v0.16b,  %[vmask1].16b    \n"
          "and    v1.16b, v1.16b,  %[vmask2].16b    \n"
          "st1    {v0.4s, v1.4s},  [%[out_ptr0]]    \n"
          : [out_ptr0] "+r"(out_ptr0)
          : [vmask1] "w"(vmask1), [vmask2] "w"(vmask2), [b0] "r"(b0)
          : "memory", "v0", "v1");
      j += 8;
    }
    if (j & 0xf) {
      float *out_ptr0 = output + (j & 0xFFF0) * k + 16 * i + (j & 0xF);
      vst1q_f32(out_ptr0, vzero);
      vst1q_f32(out_ptr0 + 4, vzero);
    }
  }
}
#else

void pack_rhs_8c(int k, int n, const float *B, int ldb, float *output,
                 const bool unroll) {
  uint32_t mask[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint32_t remain_n = n & 0x7;
  uint32x4_t vmask1 = vcltq_u32(vld1q_u32(mask), vdupq_n_u32(remain_n));
  uint32x4_t vmask2 = vcltq_u32(vld1q_u32(mask + 4), vdupq_n_u32(remain_n));

  #pragma omp parallel for if (unroll)
  for (int i = 0; i < k - 3; i += 4) {
    const float *b0 = B + i * ldb;
    const float *b1 = b0 + ldb;
    const float *b2 = b1 + ldb;
    const float *b3 = b2 + ldb;
    int j = 0;
    for (; j < n - 15; j += 16) {
      float *out_ptr0 = output + j * k + 8 * i;
      float *out_ptr1 = out_ptr0 + 8 * k;
      asm volatile(
          "vld1.32  {q0, q1},   [%[b0]]!          \n"
          "vld1.32  {q2, q3},   [%[b1]]!          \n"
          "vld1.32  {q4, q5},   [%[b0]]!          \n"
          "vld1.32  {q6, q7},   [%[b1]]!          \n"
          "vst1.32  {q0, q1},   [%[out_ptr0]]!    \n"
          "vst1.32  {q2, q3},   [%[out_ptr0]]!    \n"
          "vst1.32  {q4, q5},   [%[out_ptr1]]!    \n"
          "vst1.32  {q6, q7},   [%[out_ptr1]]!    \n"

          "vld1.32  {q0, q1},   [%[b2]]!          \n"
          "vld1.32  {q2, q3},   [%[b3]]!          \n"
          "vld1.32  {q4, q5},   [%[b2]]!          \n"
          "vld1.32  {q6, q7},   [%[b3]]!          \n"
          "vst1.32  {q0, q1},   [%[out_ptr0]]!    \n"
          "vst1.32  {q2, q3},   [%[out_ptr0]]!    \n"
          "vst1.32  {q4, q5},   [%[out_ptr1]]!    \n"
          "vst1.32  {q6, q7},   [%[out_ptr1]]!    \n"
          : [out_ptr0] "+r"(out_ptr0), [out_ptr1] "+r"(out_ptr1), [b0] "+r"(b0),
            [b1] "+r"(b1), [b2] "+r"(b2), [b3] "+r"(b3)
          :
          : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
    }
    for (; j < n - 7; j += 8) {
      float *out_ptr0 = output + j * k + 8 * i;
      asm volatile(
          "vld1.32  {q0, q1},   [%[b0]]!          \n"
          "vld1.32  {q2, q3},   [%[b1]]!          \n"
          "vld1.32  {q4, q5},   [%[b2]]!          \n"
          "vld1.32  {q6, q7},   [%[b3]]!          \n"
          "vst1.32  {q0, q1},   [%[out_ptr0]]!    \n"
          "vst1.32  {q2, q3},   [%[out_ptr0]]!    \n"
          "vst1.32  {q4, q5},   [%[out_ptr0]]!    \n"
          "vst1.32  {q6, q7},   [%[out_ptr0]]!    \n"
          : [out_ptr0] "+r"(out_ptr0), [b0] "+r"(b0), [b1] "+r"(b1),
            [b2] "+r"(b2), [b3] "+r"(b3)
          :
          : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
    }
    if (j < n) {
      float *out_ptr0 = output + j * k + 8 * i;
      asm volatile(
          "vld1.32  {q0, q1},   [%[b0]]               \n"
          "vld1.32  {q2, q3},   [%[b1]]               \n"
          "vld1.32  {q4, q5},   [%[b2]]               \n"
          "vld1.32  {q6, q7},   [%[b3]]               \n"
          "vand     q0, q0, %q[vmask1]         \n"
          "vand     q1, q1, %q[vmask2]         \n"
          "vand     q2, q2, %q[vmask1]         \n"
          "vand     q3, q3, %q[vmask2]         \n"
          "vand     q4, q4, %q[vmask1]         \n"
          "vand     q5, q5, %q[vmask2]         \n"
          "vand     q6, q6, %q[vmask1]         \n"
          "vand     q7, q7, %q[vmask2]         \n"

          "vst1.32  {q0, q1},   [%[out_ptr0]]!        \n"
          "vst1.32  {q2, q3},   [%[out_ptr0]]!        \n"
          "vst1.32  {q4, q5},   [%[out_ptr0]]!        \n"
          "vst1.32  {q6, q7},   [%[out_ptr0]]!        \n"
          : [out_ptr0] "+r"(out_ptr0)
          : [vmask1] "w"(vmask1), [vmask2] "w"(vmask2), [b0] "r"(b0),
            [b1] "r"(b1), [b2] "r"(b2), [b3] "r"(b3)
          : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
    }
  }
  // remain k
  for (int i = (k & 0xFFFC); i < k; ++i) {
    const float *b0 = B + i * ldb;
    int j = 0;
    for (; j < n - 15; j += 16) {
      float *out_ptr0 = output + j * k + 8 * i;
      float *out_ptr1 = out_ptr0 + 8 * k;
      asm volatile(
          "vld1.32  {q0, q1},   [%[b0]]!          \n"
          "vld1.32  {q2, q3},   [%[b0]]!          \n"
          "vst1.32  {q0, q1},   [%[out_ptr0]]!    \n"
          "vst1.32  {q2, q3},   [%[out_ptr1]]!    \n"
          : [out_ptr0] "+r"(out_ptr0), [out_ptr1] "+r"(out_ptr1), [b0] "+r"(b0)
          :
          : "memory", "q0", "q1", "q2", "q3");
    }
    for (; j < n - 7; j += 8) {
      float *out_ptr0 = output + j * k + 8 * i;
      asm volatile(
          "vld1.32  {q0, q1},   [%[b0]]!          \n"
          "vst1.32  {q0, q1},   [%[out_ptr0]]!    \n"
          : [out_ptr0] "+r"(out_ptr0), [b0] "+r"(b0)
          :
          : "memory", "q0", "q1");
    }
    if (j < n) {
      float *out_ptr0 = output + j * k + 8 * i;
      asm volatile(
          "vld1.32  {q0, q1},   [%[b0]]           \n"
          "vand     q0, q0, %q[vmask1]            \n"
          "vand     q1, q1, %q[vmask2]            \n"
          "vst1.32  {q0, q1},   [%[out_ptr0]]     \n"
          : [out_ptr0] "+r"(out_ptr0)
          : [vmask1] "w"(vmask1), [vmask2] "w"(vmask2), [b0] "r"(b0)
          : "memory", "q0", "q1");
    }
  }
}
#endif  // __aarch64__

void write_back_alpha_beta(const int mc, const int nc, const float alpha,
                           const float *c, const int ldc1, const float beta,
                           float *C, const int ldc2) {
  int nc1 = nc / 4;
  int _nc1 = nc % 4;

  float32x4_t _alpha = vdupq_n_f32(alpha);
  float32x4_t _beta = vdupq_n_f32(beta);
  float32x4_t cv, cv2;
  for (int i = 0; i < mc; ++i) {
    const float *c_ptr = c + i * ldc1;
    float *C_ptr = C + i * ldc2;
    for (int j = 0; j < nc1; ++j) {
      cv = vld1q_f32(c_ptr);
      cv = vmulq_f32(_alpha, cv);
      cv2 = vld1q_f32(C_ptr);
      cv = vmlaq_f32(cv, _beta, cv2);
      vst1q_f32(C_ptr, cv);
      c_ptr += 4;
      C_ptr += 4;
    }
    if (_nc1 != 0) {
      cv = vld1q_f32(c_ptr);
      cv = vmulq_f32(_alpha, cv);
      cv2 = vld1q_f32(C_ptr);
      cv = vmlaq_f32(cv, _beta, cv2);
      switch (_nc1) {
        case 3:
          vst1q_lane_f32(C_ptr + 2, cv, 2);
        case 2:
          vst1_f32(C_ptr, vget_low_f32(cv));
          break;
        case 1:
          vst1q_lane_f32(C_ptr, cv, 0);
          break;
      }
    }
  }
}

#if __aarch64__
void write_back_alpha1_beta0(const int mc, const int nc, const float *c,
                             const int ldc1, float *C, const int ldc2) {
  int nc1 = nc / 4;
  int _nc1 = nc % 4;

  const float *c_ptr;
  float *C_ptr;
  float32x4_t cv;
  for (int i = 0; i < mc; ++i) {
    c_ptr = c + i * ldc1;
    C_ptr = C + i * ldc2;
    for (int j = 0; j < nc1; ++j) {
      cv = vld1q_f32(c_ptr);
      vst1q_f32(C_ptr, cv);
      c_ptr += 4;
      C_ptr += 4;
    }
    if (_nc1 != 0) {
      cv = vld1q_f32(c_ptr);
      switch (_nc1) {
        case 3:
          vst1q_lane_f32(C_ptr + 2, cv, 2);
        case 2:
          vst1_f32(C_ptr, vget_low_f32(cv));
          break;
        case 1:
          vst1q_lane_f32(C_ptr, cv, 0);
          break;
      }
    }
  }
}

void write_back_alpha1_beta1(const int mc, const int nc, const float *c,
                             const int ldc1, float *C, const int ldc2) {
  int nc1 = nc / 4;
  int _nc1 = nc % 4;

  const float *c_ptr;
  float *C_ptr;
  float32x4_t cv, cv2;
  for (int i = 0; i < mc; ++i) {
    c_ptr = c + i * ldc1;
    C_ptr = C + i * ldc2;
    for (int j = 0; j < nc1; ++j) {
      cv = vld1q_f32(c_ptr);
      cv2 = vld1q_f32(C_ptr);
      cv = vaddq_f32(cv, cv2);
      vst1q_f32(C_ptr, cv);
      c_ptr += 4;
      C_ptr += 4;
    }
    if (_nc1 != 0) {
      cv = vld1q_f32(c_ptr);
      cv2 = vld1q_f32(C_ptr);
      cv = vaddq_f32(cv, cv2);
      switch (_nc1) {
        case 3:
          vst1q_lane_f32(C_ptr + 2, cv, 2);
        case 2:
          vst1_f32(C_ptr, vget_low_f32(cv));
          break;
        case 1:
          vst1q_lane_f32(C_ptr, cv, 0);
          break;
      }
    }
  }
}

#else
void write_back_alpha1_beta0(const int mc, const int nc, const float *c,
                             const int ldc1, float *C, const int ldc2) {
  int nc1 = nc / 16;
  int nc2 = nc % 16;
  int step1 = 4 * (ldc1 - 16 * nc1);
  int step2 = 4 * ldc2;
  int volatile m = mc;

  const float *volatile c_ptr = c;
  float *volatile C_ptr = C;
  if (nc1 > 0) {
    asm volatile(
        "subs       %[mc], %[mc], #1              \n\t"
        "blt        end_mc_%=                     \n\t"
        "loop_mc_%=:                              \n\t"

        "mov        r6,   %[C_ptr]                \n\t"
        "mov        r5,   %[nc1]                  \n\t"
        "subs       r5,   r5,   #1                \n\t"
        "blt        end_nc1_%=                    \n\t"
        "loop_nc1_%=:                             \n\t"

        "vld1.32    {q0, q1}, [%[c_ptr]]!         \n\t"
        "vst1.32    {q0, q1}, [r6]!               \n\t"

        "vld1.32    {q2, q3}, [%[c_ptr]]!         \n\t"
        "vst1.32    {q2, q3}, [r6]!               \n\t"

        "subs       r5,   r5,   #1                \n\t"
        "bge        loop_nc1_%=                   \n\t"
        "end_nc1_%=:                              \n\t"

        "add        %[c_ptr], %[c_ptr], %[step1]  \n\t"
        "add        %[C_ptr], %[C_ptr], %[step2]  \n\t"
        "subs       %[mc], %[mc], #1              \n\t"
        "bge        loop_mc_%=                    \n\t"
        "end_mc_%=:                               \n\t"
        :
        : [C_ptr] "r"(C_ptr), [c_ptr] "r"(c_ptr), [mc] "r"(m), [nc1] "r"(nc1),
          [step1] "r"(step1), [step2] "r"(step2)
        : "memory", "r5", "r6", "q0", "q1", "q2", "q3");
  }

  if (nc2 != 0) {
    for (int i = 0; i < mc; i++) {
      const float *c0 = c_ptr + nc1 * 16 + i * ldc1;
      float *C0 = C_ptr + nc1 * 16 + i * ldc2;
      for (int j = 0; j < nc2; j++) {
        *C0++ = *c0++;
      }
    }
  }
}

void write_back_alpha1_beta1(const int mc, const int nc, const float *c,
                             const int ldc1, float *C, const int ldc2) {
  int nc1 = nc / 16;
  int nc2 = nc % 16;
  int step1 = 4 * (ldc1 - 16 * nc1);
  int step2 = 4 * ldc2;
  int volatile m = mc;

  const float *volatile c_ptr = c;
  float *volatile C_ptr = C;
  if (nc1 > 0) {
    asm volatile(
        "subs       %[mc], %[mc], #1              \n\t"
        "blt        end_mc_%=                     \n\t"
        "loop_mc_%=:                              \n\t"

        "mov        r6,   %[C_ptr]                \n\t"
        "mov        r5,   %[nc1]                  \n\t"
        "subs       r5,   r5,   #1                \n\t"
        "blt        end_nc1_%=                    \n\t"
        "loop_nc1_%=:                             \n\t"

        "vld1.32    {q0, q1}, [%[c_ptr]]!         \n\t"
        "vld1.32    {q2, q3}, [r6]                \n\t"
        "vadd.f32   q0, q0, q2                    \n\t"
        "vadd.f32   q1, q1, q3                    \n\t"
        "vst1.32    {q0, q1}, [r6]!               \n\t"

        "vld1.32    {q0, q1}, [%[c_ptr]]!         \n\t"
        "vld1.32    {q2, q3}, [r6]                \n\t"
        "vadd.f32   q0, q0, q2                    \n\t"
        "vadd.f32   q1, q1, q3                    \n\t"
        "vst1.32    {q0, q1}, [r6]!               \n\t"

        "subs       r5,   r5,   #1                \n\t"
        "bge        loop_nc1_%=                   \n\t"
        "end_nc1_%=:                              \n\t"

        "add        %[c_ptr], %[c_ptr], %[step1]  \n\t"
        "add        %[C_ptr], %[C_ptr], %[step2]  \n\t"
        "subs       %[mc], %[mc], #1              \n\t"
        "bge        loop_mc_%=                    \n\t"
        "end_mc_%=:                               \n\t"
        :
        : [C_ptr] "r"(C_ptr), [c_ptr] "r"(c_ptr), [mc] "r"(m), [nc1] "r"(nc1),
          [step1] "r"(step1), [step2] "r"(step2)
        : "memory", "r5", "r6", "q0", "q1", "q2", "q3");
  }

  if (nc2 != 0) {
    for (int i = 0; i < mc; i++) {
      const float *c0 = c_ptr + nc1 * 16 + i * ldc1;
      float *C0 = C_ptr + nc1 * 16 + i * ldc2;
      for (int j = 0; j < nc2; j++) {
        *C0++ += *c0++;
      }
    }
  }
}
#endif  // __aarch64__

void write_back(const int mc, const int nc, const float alpha, const float *c,
                const int ldc1, const float beta, float *C, const int ldc2) {
  if (alpha == 1.f && beta == 0.f) {
    write_back_alpha1_beta0(mc, nc, c, ldc1, C, ldc2);
  } else if (alpha == 1.f && beta == 1.f) {
    write_back_alpha1_beta1(mc, nc, c, ldc1, C, ldc2);
  } else {
    write_back_alpha_beta(mc, nc, alpha, c, ldc1, beta, C, ldc2);
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif  // __ARM_NEON__
