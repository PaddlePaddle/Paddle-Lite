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
#include "common/log.h"
#include "memory/t_malloc.h"
#ifndef X86
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {
namespace math {

std::vector<Gemmer *> Gemmer::gemmers;
// 将A矩阵分块复制到连续内存(ColMajor)
void Gemmer::PackMatrixA(int m, int k, int m_tail, const float *A, int lda,
                         float *buffer) {
  int i, j;
  const float *Aij;
  for (i = 0; i < m - m_tail; i += MR) {
    for (j = 0; j < k; ++j) {
      Aij = &A(i, j);
      *buffer++ = *Aij;
      *buffer++ = *(Aij + 1);
      *buffer++ = *(Aij + 2);
      *buffer++ = *(Aij + 3);
    }
  }
  if (m_tail != 0) {
    for (j = 0; j < k; ++j) {
      Aij = &A(m - m_tail, j);
      for (i = 0; i < m_tail; ++i) {
        *buffer++ = *(Aij + i);
      }
      for (i = m_tail; i < MR; ++i) {
        *buffer++ = 0;
      }
    }
  }
}

// 将A矩阵分块复制到连续内存(RowMajor)
void Gemmer::PackMatrixA_(int m, int k, int m_tail, const float *A, int lda,
                          float *buffer) {
  const float *a0, *a1, *a2, *a3;
  for (int i = 0; i < m - m_tail; i += MR) {
    a0 = A + i * lda;
    a1 = A + (i + 1) * lda;
    a2 = A + (i + 2) * lda;
    a3 = A + (i + 3) * lda;
    for (int j = 0; j < k; ++j) {
      *buffer++ = *a0++;
      *buffer++ = *a1++;
      *buffer++ = *a2++;
      *buffer++ = *a3++;
    }
  }
  int i = m - m_tail;
  a0 = &A(i, 0);
  a1 = a0 + lda;
  a2 = a0 + 2 * lda;
  a3 = a0 + 3 * lda;
  if (m_tail != 0) {
    if (m_tail <= 3) {
      a3 = zero;
    }
    if (m_tail <= 2) {
      a2 = zero;
    }
    if (m_tail <= 1) {
      a1 = zero;
    }
    for (int j = 0; j < k; ++j) {
      *buffer++ = *a0++;
      *buffer++ = *a1++;
      *buffer++ = *a2++;
      *buffer++ = *a3++;
    }
  }
}

// 将B矩阵分块复制到连续内存(ColMajor)
void Gemmer::PackMatrixB(int k, int n, int n_tail, const float *B, int ldb,
                         float *buffer) {
  int i, j;
  const float *Bj, *Bj1, *Bj2, *Bj3;
  for (j = 0; j < n - n_tail; j += NR) {
    Bj = &B(0, j);
    Bj1 = &B(0, j + 1);
    Bj2 = &B(0, j + 2);
    Bj3 = &B(0, j + 3);
    for (i = 0; i < k; ++i) {
      *buffer++ = *Bj++;
      *buffer++ = *Bj1++;
      *buffer++ = *Bj2++;
      *buffer++ = *Bj3++;
    }
  }
  if (n_tail != 0) {
    for (i = 0; i < k; ++i) {
      for (int j = n - n_tail; j < n; ++j) {
        *buffer++ = B(i, j);
      }
      for (int j = n; j < n + (NR - n_tail); ++j) {
        *buffer++ = 0;
      }
    }
  }
}

// 将B矩阵分块复制到连续内存(RowMajor)
void Gemmer::PackMatrixB_(int k, int n, int n_tail, const float *B, int ldb,
                          float *buffer) {
  const float *b0;
  for (int j = 0; j < n - n_tail; j += NR) {
    for (int i = 0; i < k; ++i) {
      b0 = &B(i, j);
      asm volatile(
          "pld        [%[b0]]               \n\t"
          "vld1.32    {q0, q1}, [%[b0]]         \n\t"
          "vst1.32    {q0, q1}, [%[buffer]]!    \n\t"
          : [buffer] "+r"(buffer)
          : [b0] "r"(b0)
          : "memory", "q0", "q0");
    }
  }
  if (n_tail != 0) {
    for (int i = 0; i < k; ++i) {
      b0 = &B(i, n - n_tail);
      for (int j = n - n_tail; j < n; ++j) {
        *buffer++ = *b0++;
      }
      for (int j = n; j < n + (NR - n_tail); ++j) {
        *buffer++ = 0;
      }
    }
  }
}

// 分块矩阵乘法
void Gemmer::InnerKernel(int mc, int nc, float alpha, const float *a,
                         const float *b, float beta, float *c, float *C,
                         int ldc, bool relu) {
  for (int j = 0; j < nc; j += NR) {
    for (int i = 0; i < mc; i += MR) {
      // AddDot4x4(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
      AddDot4x8(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
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
void Gemmer::InnerKernelWithBn(int mc, int nc, float alpha, const float *a,
                               const float *b, float beta, float *c, float *C,
                               int ldc, bool relu, float *new_scale,
                               float *new_bias) {
  for (int j = 0; j < nc; j += NR) {
    for (int i = 0; i < mc; i += MR) {
      // AddDot4x4(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
      AddDot4x8(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
    }
  }

  if (relu) {
    WriteWithBnRelu(mc, nc, c, C, ldc, new_scale, new_bias);
  } else {
    WriteWithBn(mc, nc, c, C, ldc, new_scale, new_bias);
  }
}

#if defined(IOS)
void Gemmer::AddDot4x4(int k, const float *a, const float *b, float *C,
                       int ldc) {
  // init C
  float32x4_t cv0 = vdupq_n_f32(0.0);
  float32x4_t cv1 = vdupq_n_f32(0.0);
  float32x4_t cv2 = vdupq_n_f32(0.0);
  float32x4_t cv3 = vdupq_n_f32(0.0);

  float32x4_t av;
  float32x4_t bv;

  float32x2_t av01;
  float32x2_t av23;

  for (int p = 0; p < k; p += 1) {
    av = vld1q_f32(a);
    bv = vld1q_f32(b);

    av01 = vget_low_f32(av);
    cv0 = vmlaq_lane_f32(cv0, bv, av01, 0);
    cv1 = vmlaq_lane_f32(cv1, bv, av01, 1);
    av23 = vget_high_f32(av);
    cv2 = vmlaq_lane_f32(cv2, bv, av23, 0);
    cv3 = vmlaq_lane_f32(cv3, bv, av23, 1);

    a += MR;
    b += NR;
  }
  float32x4x4_t cv = {cv0, cv1, cv2, cv3};
  int i, j;
  for (i = 0; i < mc; ++i) {
    for (j = 0; j < nc; ++j) {
      if (beta == 0.0) {
        C(i, j) = 0.0;
      } else if (beta != 1.0) {
        C(i, j) *= beta;
      }
      if (j == 0) {
        C(i, j) += alpha * vgetq_lane_f32(cv.val[i], 0);
      } else if (j == 1) {
        C(i, j) += alpha * vgetq_lane_f32(cv.val[i], 1);
      } else if (j == 2) {
        C(i, j) += alpha * vgetq_lane_f32(cv.val[i], 2);
      } else if (j == 3) {
        C(i, j) += alpha * vgetq_lane_f32(cv.val[i], 3);
      }
    }
  }
}
}  // namespace math

#elif defined(ARMV7)
void Gemmer::AddDot4x4(int k, const float *a, const float *b, float *c,
                       int ldc) {
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

#else
void Gemmer::AddDot4x4(int k, const float *a, const float *b, float *c,
                       int ldc) {
  float *c0, *c1, *c2, *c3;
  c0 = c;
  c1 = c + ldc;
  c2 = c + 2 * ldc;
  c3 = c + 3 * ldc;
  for (int p = 0; p < k; p += 1) {
    // first row
    c0[0] += a[0] * b[0];
    c0[1] += a[0] * b[1];
    c0[2] += a[0] * b[2];
    c0[3] += a[0] * b[3];

    // second row
    c1[0] += a[1] * b[0];
    c1[1] += a[1] * b[1];
    c1[2] += a[1] * b[2];
    c1[3] += a[1] * b[3];

    // third row
    c2[0] += a[2] * b[0];
    c2[1] += a[2] * b[1];
    c2[2] += a[2] * b[2];
    c2[3] += a[2] * b[3];

    // fourth row
    c3[0] += a[3] * b[0];
    c3[1] += a[3] * b[1];
    c3[2] += a[3] * b[2];
    c3[3] += a[3] * b[3];

    a += 4;
    b += 4;
  }
}

#endif

// 32位 float 矩阵乘法
void Gemmer::Sgemm(int m, int n, int k, float alpha, const float *A, int lda,
                   const float *B, int ldb, float beta, float *C, int ldc,
                   bool relu) {
  // L1 data cache is 32 kib (Per Contex-A57, Contex-A72, Contex-A73)
  // L2 cache is 0.5~4 Mib (Contex-A72 cluster)
  int L1 = 30 * 1024;
  int L2 = 1 * 1024 * 1024;

  KC = k;
  MC = L2 / (2 * KC * sizeof(float));
  NC = MC;

  // make sure MC is multiple of 4, and NC is multiple of 8
  int mblock_num = (m + MC - 1) / MC;
  MC = (m + mblock_num - 1) / mblock_num;
  MC = (MC + 4 - 1) / 4 * 4;
  //  DLOG << "mblock_num = " << mblock_num << ", MC = " << MC << "\n";

  int nblock_num = (n + NC - 1) / NC;
  NC = (n + nblock_num - 1) / nblock_num;
  NC = (NC + 8 - 1) / 8 * 8;
  //  DLOG << "nblock_num = " << nblock_num << ", NC = " << NC << "\n";

  packedA = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * MC * KC));
  packedB = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * KC * NC));
  packedC = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * MC * NC));
  zero = static_cast<float *>(paddle_mobile::memory::Alloc(sizeof(float) * KC));

  for (int l = 0; l < KC; ++l) {
    zero[l] = 0;
  }

  int mc, nc;
  for (int j = 0; j < n; j += NC) {
    nc = s_min(n - j, NC);
    PackMatrixB_(KC, nc, nc % NR, &B(0, j), ldb, packedB);
    for (int i = 0; i < m; i += MC) {
      mc = s_min(m - i, MC);
      PackMatrixA_(mc, KC, mc % MR, &A(i, 0), lda, packedA);
      InnerKernel(mc, nc, alpha, packedA, packedB, beta, packedC, &C(i, j), ldc,
                  relu);
    }
  }

  paddle_mobile::memory::Free(packedA);
  paddle_mobile::memory::Free(packedB);
  paddle_mobile::memory::Free(packedC);
  paddle_mobile::memory::Free(zero);
}

void Gemmer::SgemmWithBn(int m, int n, int k, float alpha, const float *A,
                         int lda, const float *B, int ldb, float beta, float *C,
                         int ldc, bool relu, float *new_scale,
                         float *new_bias) {
  // L1 data cache is 32 kib (Per Contex-A57, Contex-A72, Contex-A73)
  // L2 cache is 0.5~4 Mib (Contex-A72 cluster)
  int L1 = 30 * 1024;
  int L2 = 1 * 1024 * 1024;

  KC = k;
  MC = L2 / (2 * KC * sizeof(float));
  NC = MC;

  // make sure MC is multiple of 4, and NC is multiple of 8
  int mblock_num = (m + MC - 1) / MC;
  MC = (m + mblock_num - 1) / mblock_num;
  MC = (MC + 4 - 1) / 4 * 4;
  //  DLOG << "mblock_num = " << mblock_num << ", MC = " << MC << "\n";

  int nblock_num = (n + NC - 1) / NC;
  NC = (n + nblock_num - 1) / nblock_num;
  NC = (NC + 8 - 1) / 8 * 8;
  //  DLOG << "nblock_num = " << nblock_num << ", NC = " << NC << "\n";

  packedA = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * MC * KC));
  packedB = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * KC * NC));
  packedC = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * MC * NC));
  zero = static_cast<float *>(paddle_mobile::memory::Alloc(sizeof(float) * KC));

  for (int l = 0; l < KC; ++l) {
    zero[l] = 0;
  }

  int mc, nc;
  for (int j = 0; j < n; j += NC) {
    nc = s_min(n - j, NC);
    PackMatrixB_(KC, nc, nc % NR, &B(0, j), ldb, packedB);
    for (int i = 0; i < m; i += MC) {
      mc = s_min(m - i, MC);
      PackMatrixA_(mc, KC, mc % MR, &A(i, 0), lda, packedA);
      InnerKernelWithBn(mc, nc, alpha, packedA, packedB, beta, packedC,
                        &C(i, j), ldc, relu, new_scale + ldc * i + j,
                        new_bias + ldc * i + j);
    }
  }

  paddle_mobile::memory::Free(packedA);
  paddle_mobile::memory::Free(packedB);
  paddle_mobile::memory::Free(packedC);
  paddle_mobile::memory::Free(zero);
}

void Gemmer::VectorKernel(int m, int n, int k, float alpha, const float *A,
                          int lda, const float *B, int ldb, float beta,
                          float *C, int ldc, bool relu) {
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

void Gemmer::VectorKernelWithBn(int m, int n, int k, float alpha,
                                const float *A, int lda, const float *B,
                                int ldb, float beta, float *C, int ldc,
                                bool relu, float *new_scale, float *new_bias) {
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

void Gemmer::AddDot4x8(int k, const float *a, const float *b, float *c,
                       int ldc) {
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

// C = A * B
void Gemmer::WriteBasic(int mc, int nc, float *c, float *C, int ldc) {
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
void Gemmer::WriteWithAlphaBeta(int mc, int nc, float *c, float *C, int ldc) {}

// C = A * B + C
void Gemmer::WriteWithAdd(int mc, int nc, float *c, float *C, int ldc) {
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

// C = A * B + C, relu(C)
void Gemmer::WriteWithAddRelu(int mc, int nc, float *c, float *C, int ldc) {
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

// C = A * B, batchnorm(C)
void Gemmer::WriteWithBn(int mc, int nc, float *c, float *C, int ldc,
                         float *scale, float *bias) {
  int nc1 = nc / 16;
  int _nc1 = nc % 16;
  int nc2 = _nc1 / 4;
  int nc3 = 16 - 4 * (_nc1 % 4);
  int step = 4 * (ldc - nc);
  int step1 = 4 * (NC - nc);

  asm volatile(
      "subs       %[mc], %[mc], #1        \n\t"
      "blt        end_mc_%=               \n\t"
      "loop_mc_%=:                        \n\t"

      "mov        r5,   %[nc1]            \n\t"
      "mov        r6,   %[nc2]            \n\t"

      "subs       r5,   r5,   #1          \n\t"
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

      "subs       r5,   r5,   #1          \n\t"
      "bge        loop_nc1_%=             \n\t"
      "end_nc1_%=:                        \n\t"

      "subs       r6,  r6,   #1           \n\t"
      "blt        end_nc2_%=              \n\t"
      "loop_nc2_%=:                       \n\t"

      "vld1.32    {q0},   [%[c]]!           \n\t"
      "vld1.32    {q1},   [%[scale]]!       \n\t"
      "vld1.32    {q10},  [%[bias]]!        \n\t"
      "vmla.f32   q10,    q0,   q1          \n\t"
      "vst1.32    {q10},  [%[C]]!           \n\t"

      "subs       r6,   r6,   #1          \n\t"
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

      "add        %[c],     %[c],     %[step1]  \n\t"
      "add        %[scale], %[scale], %[step]   \n\t"
      "add        %[bias],  %[bias],  %[step]   \n\t"
      "add        %[C],     %[C],     %[step]   \n\t"

      "subs       %[mc], %[mc], #1        \n\t"
      "bge        loop_mc_%=              \n\t"
      "end_mc_%=:                         \n\t"

      :
      : [C] "r"(C), [c] "r"(c), [mc] "r"(mc), [nc1] "r"(nc1), [nc2] "r"(nc2),
        [nc3] "r"(nc3), [step] "r"(step), [step1] "r"(step1),
        [scale] "r"(scale), [bias] "r"(bias)
      : "memory", "cc", "r5", "r6", "r7", "r8", "q0", "q1", "q2", "q3", "q4",
        "q5", "q6", "q7", "q10", "q11", "q12", "q13");
}

// C = A * B, batchnorm(C), relu(C)
void Gemmer::WriteWithBnRelu(int mc, int nc, float *c, float *C, int ldc,
                             float *scale, float *bias) {
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

      "subs       r5,   r5,   #1          \n\t"
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

      "subs       r5,   r5,   #1          \n\t"
      "bge        loop_nc1_%=             \n\t"
      "end_nc1_%=:                        \n\t"

      "subs       r6,  r6,   #1           \n\t"
      "blt        end_nc2_%=              \n\t"
      "loop_nc2_%=:                       \n\t"

      "vld1.32    {q0},   [%[c]]!           \n\t"
      "vld1.32    {q1},   [%[scale]]!       \n\t"
      "vld1.32    {q10},  [%[bias]]!        \n\t"
      "vmla.f32   q10,    q0,   q1          \n\t"
      "vmax.f32   q10,    q10,  q14         \n\t"
      "vst1.32    {q10},  [%[C]]!           \n\t"

      "subs       r6,   r6,   #1          \n\t"
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

      "add        %[c],     %[c],     %[step1]  \n\t"
      "add        %[scale], %[scale], %[step]   \n\t"
      "add        %[bias],  %[bias],  %[step]   \n\t"
      "add        %[C],     %[C],     %[step]   \n\t"

      "subs       %[mc], %[mc], #1        \n\t"
      "bge        loop_mc_%=              \n\t"
      "end_mc_%=:                         \n\t"

      :
      : [C] "r"(C), [c] "r"(c), [mc] "r"(mc), [nc1] "r"(nc1), [nc2] "r"(nc2),
        [nc3] "r"(nc3), [step] "r"(step), [step1] "r"(step1),
        [scale] "r"(scale), [bias] "r"(bias)
      : "memory", "r5", "r6", "r7", "r8", "q0", "q1", "q2", "q3", "q4", "q5",
        "q6", "q7", "q10", "q11", "q12", "q13", "q14");
}

// C = A * B
void Gemmer::VecWriteBasic(int n, float *c, float *C, int ldc) {
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
void Gemmer::VecWriteWithAlphaBeta(int n, float *c, float *C, int ldc) {}

// C = A * B + C
void Gemmer::VecWriteWithAdd(int n, float *c, float *C, int ldc) {
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
void Gemmer::VecWriteWithAddRelu(int n, float *c, float *C, int ldc) {
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
void Gemmer::VecWriteWithBn(int n, float *c, float *C, int ldc, float *scale,
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
void Gemmer::VecWriteWithBnRelu(int n, float *c, float *C, int ldc,
                                float *scale, float *bias) {
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

}  // namespace operators
}  // namespace paddle_mobile
}  // namespace paddle_mobile
