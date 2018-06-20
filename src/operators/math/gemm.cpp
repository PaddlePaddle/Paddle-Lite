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
#ifndef X86
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {
namespace math {
alignas(64) float packedA[MC * KC];
alignas(64) float packedB[KC * NC];
alignas(64) float ab[MR * NR];
// 将A矩阵分块复制到连续内存(ColMajor)
void PackMatrixA(int m, int k, int paddingM, const float *A, int lda,
                 float *buffer) {
  int i, j;
  const float *Aij;
  for (i = 0; i < m - paddingM; i += MR) {
    for (int j = 0; j < k; ++j) {
      Aij = &A(i, j);
      *buffer++ = *Aij;
      *buffer++ = *(Aij + 1);
      *buffer++ = *(Aij + 2);
      *buffer++ = *(Aij + 3);
    }
  }
  if (paddingM != 0) {
    for (j = 0; j < k; ++j) {
      Aij = &A(m - paddingM, j);
      for (i = 0; i < paddingM; ++i) {
        *buffer++ = *(Aij + i);
      }
      for (i = paddingM; i < MR; ++i) {
        *buffer++ = 0;
      }
    }
  }
}

// 将A矩阵分块复制到连续内存(RowMajor)
void PackMatrixA_(int m, int k, int paddingM, const float *A, int lda,
                  float *buffer) {
  int i, j;
  const float *Ai, *Ai1, *Ai2, *Ai3;
  for (i = 0; i < m - paddingM; i += MR) {
    Ai = &A(i, 0);
    Ai1 = &A(i + 1, 0);
    Ai2 = &A(i + 2, 0);
    Ai3 = &A(i + 3, 0);
    for (int j = 0; j < k; ++j) {
      *buffer++ = *Ai++;
      *buffer++ = *Ai1++;
      *buffer++ = *Ai2++;
      *buffer++ = *Ai3++;
    }
  }
  if (paddingM != 0) {
    for (j = 0; j < k; ++j) {
      for (i = m - paddingM; i < m; ++i) {
        *buffer++ = A(i, j);
      }
      for (i = m; i < m + (MR - paddingM); ++i) {
        *buffer++ = 0;
      }
    }
  }
}

// 将B矩阵分块复制到连续内存(ColMajor)
void PackMatrixB(int k, int n, int paddingN, const float *B, int ldb,
                 float *buffer) {
  int i, j;
  const float *Bj, *Bj1, *Bj2, *Bj3;
  for (j = 0; j < n - paddingN; j += NR) {
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
  if (paddingN != 0) {
    for (i = 0; i < k; ++i) {
      for (int j = n - paddingN; j < n; ++j) {
        *buffer++ = B(i, j);
      }
      for (int j = n; j < n + (NR - paddingN); ++j) {
        *buffer++ = 0;
      }
    }
  }
}

// 将B矩阵分块复制到连续内存(RowMajor)
void PackMatrixB_(int k, int n, int paddingN, const float *B, int ldb,
                  float *buffer) {
  int i, j;
  const float *Bij;
  for (j = 0; j < n - paddingN; j += NR) {
    for (i = 0; i < k; ++i) {
      Bij = &B(i, j);
      asm volatile(
          "vld1.32    {q0}, [%[Bij]]        \n\t"
          "vst1.32    {q0}, [%[buffer]]!    \n\t"
          : [buffer] "+r"(buffer)
          : [Bij] "r"(Bij)
          : "memory", "q0");
    }
  }
  if (paddingN != 0) {
    for (i = 0; i < k; ++i) {
      Bij = &B(i, n - paddingN);
      for (int j = n - paddingN; j < n; ++j) {
        *buffer++ = *Bij++;
      }
      for (int j = n; j < n + (NR - paddingN); ++j) {
        *buffer++ = 0;
      }
    }
  }
}

// 分块矩阵乘法
void InnerKernel(int m, int n, int k, float alpha, const float *A, int lda,
                 const float *B, int ldb, float beta, float *C, int ldc,
                 int first_time) {
  int Buff_A_M = m;
  int Buff_B_N = n;

  int _mc = m % MR;
  int _nc = n % NR;

  if (_mc != 0) {
    Buff_A_M = m + (MR - _mc);
  }

  if (_nc != 0) {
    Buff_B_N = n + (NR - _nc);
  }

  if (first_time) {
    PackMatrixB_(k, n, _nc, B, ldb, packedB);
  }
  PackMatrixA_(m, k, _mc, A, lda, packedA);

  int i, j, mc, nc;

  // B 取 4 列, 打包预热
  for (j = 0; j < Buff_B_N; j += NR) {
    nc = (n - j) < NR ? _nc : NR;
    // A 取 4 行，打包预热
    for (i = 0; i < Buff_A_M; i += MR) {
      mc = (m - i) < MR ? _mc : MR;
      AddDot4x4(k, alpha, &packedA[i * k], 4, &packedB[j * k], k, beta,
                &C(i, j), ldc, mc, nc);
    }
  }
}

// 分块矩阵乘法
void InnerKernel_relu(int m, int n, int k, float alpha, const float *A, int lda,
                      const float *B, int ldb, float beta, float *C, int ldc,
                      int first_time, bool relu = false) {
  int Buff_A_M = m;
  int Buff_B_N = n;

  int _mc = m % MR;
  int _nc = n % NR;

  if (_mc != 0) {
    Buff_A_M = m + (MR - _mc);
  }

  if (_nc != 0) {
    Buff_B_N = n + (NR - _nc);
  }

  float packedA[MC * KC];
  static float packedB[KC * NC];

  if (first_time) {
    PackMatrixB_(k, n, _nc, B, ldb, packedB);
  }
  PackMatrixA_(m, k, _mc, A, lda, packedA);

  int i, j, mc, nc;

  // B 取 4 列, 打包预热
  for (j = 0; j < Buff_B_N; j += NR) {
    nc = (n - j) < NR ? _nc : NR;
    // A 取 4 行，打包预热
    for (i = 0; i < Buff_A_M; i += MR) {
      mc = (m - i) < MR ? _mc : MR;
      AddDot4x4_relu(k, alpha, &packedA[i * k], 4, &packedB[j * k], k, beta,
                     &C(i, j), ldc, mc, nc, relu);
    }
  }
}

//计算一个更小的 4 * 4 的 C 矩阵分块
#if defined(IOS)
void AddDot4x4(int k, float alpha, const float *a, int lda, const float *b,
               int ldb, float beta, float *C, int ldc, int mc, int nc) {
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

void AddDot4x4_relu(int k, float alpha, const float *a, int lda, const float *b,
                    int ldb, float beta, float *C, int ldc, int mc, int nc,
                    bool relu = false) {
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
      if (C(i, j) < 0) {
        C(i, j) = 0;
      }
    }
  }
}

#elif defined(ARMV7)
void AddDot4x4(int k, float alpha, const float *a, int lda, const float *b,
               int ldb, float beta, float *C, int ldc, int mc, int nc) {
  int kc1 = k / 4, kc2 = k % 4;
  int bytes_ldc = 4 * ldc;
  int flag_alpha = (alpha == 1.0) ? 1 : 2;
  int flag_beta;
  if (beta == 0.0) {
    flag_beta = 0;
  } else if (beta == 1.0) {
    flag_beta = 1;
  } else {
    flag_beta = 2;
  }
  asm volatile(
      "pld        [%[a]]              \n\t"
      "pld        [%[b]]              \n\t"
      "vmov.f32   q10,    #0.0        \n\t"
      "vmov.f32   q11,    #0.0        \n\t"
      "vmov.f32   q12,    #0.0        \n\t"
      "vmov.f32   q13,    #0.0        \n\t"

      "subs       %[kc1], %[kc1], #1  \n\t"
      "blt        end_kc1_%=          \n\t"
      "loop_kc1_%=:                   \n\t"
      "pld        [%[a], #64]         \n\t"
      "pld        [%[b], #64]         \n\t"
      "vld1.32    {q0, q1}, [%[a]]!   \n\t"
      "vld1.32    {q2, q3}, [%[b]]!   \n\t"
      "vmla.f32   q10, q2, d0[0]      \n\t"
      "vmla.f32   q11, q2, d0[1]      \n\t"
      "vmla.f32   q12, q2, d1[0]      \n\t"
      "vmla.f32   q13, q2, d1[1]      \n\t"
      "vmla.f32   q10, q3, d2[0]      \n\t"
      "vmla.f32   q11, q3, d2[1]      \n\t"
      "vmla.f32   q12, q3, d3[0]      \n\t"
      "vmla.f32   q13, q3, d3[1]      \n\t"
      "vld1.32    {q0, q1}, [%[a]]!   \n\t"
      "vld1.32    {q2, q3}, [%[b]]!   \n\t"
      "vmla.f32   q10, q2, d0[0]      \n\t"
      "vmla.f32   q11, q2, d0[1]      \n\t"
      "vmla.f32   q12, q2, d1[0]      \n\t"
      "vmla.f32   q13, q2, d1[1]      \n\t"
      "vmla.f32   q10, q3, d2[0]      \n\t"
      "vmla.f32   q11, q3, d2[1]      \n\t"
      "vmla.f32   q12, q3, d3[0]      \n\t"
      "vmla.f32   q13, q3, d3[1]      \n\t"
      "subs       %[kc1], %[kc1], #1  \n\t"
      "bge        loop_kc1_%=         \n\t"
      "end_kc1_%=:                    \n\t"

      "subs       %[kc2], %[kc2], #1  \n\t"
      "blt        end_kc2_%=          \n\t"
      "vld1.32    {q0}, [%[a]]!       \n\t"
      "vld1.32    {q1}, [%[b]]!       \n\t"
      "vmla.f32   q10, q1, d0[0]      \n\t"
      "vmla.f32   q11, q1, d0[1]      \n\t"
      "vmla.f32   q12, q1, d1[0]      \n\t"
      "vmla.f32   q13, q1, d1[1]      \n\t"
      "end_kc2_%=:                    \n\t"

      "cmp        %[mc],      #4      \n\t"
      "bne        temp_%=             \n\t"
      "cmp        %[nc],      #4      \n\t"
      "bne        temp_%=             \n\t"

      "vmov.f32   d8[0],    %[alpha]  \n\t"
      "vmov.f32   d8[1],    %[beta]   \n\t"

      "cmp        %[flag_alpha],  #1  \n\t"
      "bne        alpha_%=            \n\t"

      "alpha_%=:                      \n\t"
      "vmul.f32   q10, q10, d8[0]     \n\t"
      "vmul.f32   q11, q11, d8[0]     \n\t"
      "vmul.f32   q12, q12, d8[0]     \n\t"
      "vmul.f32   q13, q13, d8[0]     \n\t"

      "beta_%=:                       \n\t"
      "cmp        %[flag_beta],   #0  \n\t"
      "beq        memory_%=           \n\t"

      "mov        r4,     %[C]        \n\t"
      "mov        r6,     %[bytes_ldc]\n\t"
      "vld1.32    {q0}, [r4], r6      \n\t"
      "vld1.32    {q1}, [r4], r6      \n\t"
      "vld1.32    {q2}, [r4], r6      \n\t"
      "vld1.32    {q3}, [r4]          \n\t"
      "cmp        %[flag_beta],   #1  \n\t"
      "beq        beta_eq1_%=         \n\t"
      "bne        beta_ne1_%=         \n\t"

      "beta_eq1_%=:                   \n\t"
      "vadd.f32   q10, q10, q0        \n\t"
      "vadd.f32   q11, q11, q1        \n\t"
      "vadd.f32   q12, q12, q2        \n\t"
      "vadd.f32   q13, q13, q3        \n\t"
      "b          memory_%=           \n\t"

      "beta_ne1_%=:                   \n\t"
      "vmla.f32   q10, q0, d8[1]      \n\t"
      "vmla.f32   q11, q1, d8[1]      \n\t"
      "vmla.f32   q12, q2, d8[1]      \n\t"
      "vmla.f32   q13, q3, d8[1]      \n\t"

      "memory_%=:                     \n\t"
      "mov        r5,     %[C]        \n\t"
      "mov        r6,     %[bytes_ldc]\n\t"
      "vst1.32    {q10}, [r5], r6     \n\t"
      "vst1.32    {q11}, [r5], r6     \n\t"
      "vst1.32    {q12}, [r5], r6     \n\t"
      "vst1.32    {q13}, [r5]         \n\t"
      "b          end_%=              \n\t"

      "temp_%=:                       \n\t"
      "vst1.32    {q10, q11}, [%[ab]]!\n\t"
      "vst1.32    {q12, q13}, [%[ab]] \n\t"
      "end_%=:                        \n\t"
      :
      : [a] "r"(a), [b] "r"(b), [C] "r"(C), [ab] "r"(ab), [kc1] "r"(kc1),
        [kc2] "r"(kc2), [mc] "r"(mc), [nc] "r"(nc), [alpha] "r"(alpha),
        [beta] "r"(beta), [bytes_ldc] "r"(bytes_ldc),
        [flag_alpha] "r"(flag_alpha), [flag_beta] "r"(flag_beta)
      : "memory", "q0", "q1", "q2", "q3", "q4", "q10", "q11", "q12", "q13");

  if (mc != MR || nc != NR) {
    int i, j;
    for (i = 0; i < mc; ++i) {
      for (j = 0; j < nc; ++j) {
        if (beta == 0.0) {
          if (alpha != 1.0) {
            C(i, j) = alpha * ab[i * MR + j];
          } else {
            C(i, j) = ab[i * MR + j];
          }
        } else {
          if (beta != 1.0) {
            C(i, j) *= beta;
          }
          if (alpha != 1.0) {
            C(i, j) += alpha * ab[i * MR + j];
          } else {
            C(i, j) += ab[i * MR + j];
          }
        }
      }
    }
  }
}

void AddDot4x4_relu(int k, float alpha, const float *a, int lda, const float *b,
                    int ldb, float beta, float *C, int ldc, int mc, int nc,
                    bool relu = false) {
  int kc1 = k / 4, kc2 = k % 4;
  int bytes_ldc = 4 * ldc;
  int flag_alpha = (alpha == 1.0) ? 1 : 2;
  int flag_beta;
  if (beta == 0.0) {
    flag_beta = 0;
  } else if (beta == 1.0) {
    flag_beta = 1;
  } else {
    flag_beta = 2;
  }
  asm volatile(
      "pld        [%[a]]              \n\t"
      "pld        [%[b]]              \n\t"
      "vmov.f32   q10,    #0.0        \n\t"
      "vmov.f32   q11,    #0.0        \n\t"
      "vmov.f32   q12,    #0.0        \n\t"
      "vmov.f32   q13,    #0.0        \n\t"

      "subs       %[kc1], %[kc1], #1  \n\t"
      "blt        end_kc1_%=          \n\t"
      "loop_kc1_%=:                   \n\t"
      "pld        [%[a], #64]         \n\t"
      "pld        [%[b], #64]         \n\t"
      "vld1.32    {q0, q1}, [%[a]]!   \n\t"
      "vld1.32    {q2, q3}, [%[b]]!   \n\t"
      "vmla.f32   q10, q2, d0[0]      \n\t"
      "vmla.f32   q11, q2, d0[1]      \n\t"
      "vmla.f32   q12, q2, d1[0]      \n\t"
      "vmla.f32   q13, q2, d1[1]      \n\t"
      "vmla.f32   q10, q3, d2[0]      \n\t"
      "vmla.f32   q11, q3, d2[1]      \n\t"
      "vmla.f32   q12, q3, d3[0]      \n\t"
      "vmla.f32   q13, q3, d3[1]      \n\t"
      "vld1.32    {q0, q1}, [%[a]]!   \n\t"
      "vld1.32    {q2, q3}, [%[b]]!   \n\t"
      "vmla.f32   q10, q2, d0[0]      \n\t"
      "vmla.f32   q11, q2, d0[1]      \n\t"
      "vmla.f32   q12, q2, d1[0]      \n\t"
      "vmla.f32   q13, q2, d1[1]      \n\t"
      "vmla.f32   q10, q3, d2[0]      \n\t"
      "vmla.f32   q11, q3, d2[1]      \n\t"
      "vmla.f32   q12, q3, d3[0]      \n\t"
      "vmla.f32   q13, q3, d3[1]      \n\t"
      "subs       %[kc1], %[kc1], #1  \n\t"
      "bge        loop_kc1_%=         \n\t"
      "end_kc1_%=:                    \n\t"

      "subs       %[kc2], %[kc2], #1  \n\t"
      "blt        end_kc2_%=          \n\t"
      "vld1.32    {q0}, [%[a]]!       \n\t"
      "vld1.32    {q1}, [%[b]]!       \n\t"
      "vmla.f32   q10, q1, d0[0]      \n\t"
      "vmla.f32   q11, q1, d0[1]      \n\t"
      "vmla.f32   q12, q1, d1[0]      \n\t"
      "vmla.f32   q13, q1, d1[1]      \n\t"
      "end_kc2_%=:                    \n\t"

      "cmp        %[mc],      #4      \n\t"
      "bne        temp_%=             \n\t"
      "cmp        %[nc],      #4      \n\t"
      "bne        temp_%=             \n\t"

      "vmov.f32   d8[0],    %[alpha]  \n\t"
      "vmov.f32   d8[1],    %[beta]   \n\t"

      "cmp        %[flag_alpha],  #1  \n\t"
      "bne        alpha_%=            \n\t"

      "alpha_%=:                      \n\t"
      "vmul.f32   q10, q10, d8[0]     \n\t"
      "vmul.f32   q11, q11, d8[0]     \n\t"
      "vmul.f32   q12, q12, d8[0]     \n\t"
      "vmul.f32   q13, q13, d8[0]     \n\t"

      "beta_%=:                       \n\t"
      "cmp        %[flag_beta],   #0  \n\t"
      "beq        memory_%=           \n\t"

      "mov        r4,     %[C]        \n\t"
      "mov        r6,     %[bytes_ldc]\n\t"
      "vld1.32    {q0}, [r4], r6      \n\t"
      "vld1.32    {q1}, [r4], r6      \n\t"
      "vld1.32    {q2}, [r4], r6      \n\t"
      "vld1.32    {q3}, [r4]          \n\t"
      "cmp        %[flag_beta],   #1  \n\t"
      "beq        beta_eq1_%=         \n\t"
      "bne        beta_ne1_%=         \n\t"

      "beta_eq1_%=:                   \n\t"
      "vadd.f32   q10, q10, q0        \n\t"
      "vadd.f32   q11, q11, q1        \n\t"
      "vadd.f32   q12, q12, q2        \n\t"
      "vadd.f32   q13, q13, q3        \n\t"
      "b          memory_%=           \n\t"

      "beta_ne1_%=:                   \n\t"
      "vmla.f32   q10, q0, d8[1]      \n\t"
      "vmla.f32   q11, q1, d8[1]      \n\t"
      "vmla.f32   q12, q2, d8[1]      \n\t"
      "vmla.f32   q13, q3, d8[1]      \n\t"

      "memory_%=:                     \n\t"
      "vmax.f32 q10, q10, q14           \n\t"
      "vmax.f32 q11, q11, q14           \n\t"
      "vmax.f32 q12, q12, q14           \n\t"
      "vmax.f32 q13, q13, q14           \n\t"
      "mov        r5,     %[C]        \n\t"
      "mov        r6,     %[bytes_ldc]\n\t"
      "vst1.32    {q10}, [r5], r6     \n\t"
      "vst1.32    {q11}, [r5], r6     \n\t"
      "vst1.32    {q12}, [r5], r6     \n\t"
      "vst1.32    {q13}, [r5]         \n\t"
      "b          end_%=              \n\t"

      "temp_%=:                       \n\t"
      "vst1.32    {q10, q11}, [%[ab]]!\n\t"
      "vst1.32    {q12, q13}, [%[ab]] \n\t"
      "end_%=:                        \n\t"
      :
      : [a] "r"(a), [b] "r"(b), [C] "r"(C), [ab] "r"(ab), [kc1] "r"(kc1),
        [kc2] "r"(kc2), [mc] "r"(mc), [nc] "r"(nc), [alpha] "r"(alpha),
        [beta] "r"(beta), [bytes_ldc] "r"(bytes_ldc),
        [flag_alpha] "r"(flag_alpha), [flag_beta] "r"(flag_beta)
      : "memory", "q0", "q1", "q2", "q3", "q4", "q10", "q11", "q12", "q13");

  if (mc != MR || nc != NR) {
    int i, j;
    for (i = 0; i < mc; ++i) {
      for (j = 0; j < nc; ++j) {
        if (beta == 0.0) {
          if (alpha != 1.0) {
            C(i, j) = alpha * ab[i * MR + j];
          } else {
            C(i, j) = ab[i * MR + j];
          }
        } else {
          if (beta != 1.0) {
            C(i, j) *= beta;
          }
          if (alpha != 1.0) {
            C(i, j) += alpha * ab[i * MR + j];
          } else {
            C(i, j) += ab[i * MR + j];
          }
        }
        if (relu) {
          if (C(i, j) < 0) {
            C(i, j) = 0;
          }
        }
      }
    }
  }
}

#else
void AddDot4x4(int k, float alpha, const float *a, int lda, const float *b,
               int ldb, float beta, float *C, int ldc, int mc, int nc) {
  float c[16] = {0};
  float reg_a0, reg_a1, reg_a2, reg_a3, reg_b0, reg_b1, reg_b2, reg_b3;

  for (int p = 0; p < k; p += 1) {
    reg_b0 = *b++;
    reg_b1 = *b++;
    reg_b2 = *b++;
    reg_b3 = *b++;

    reg_a0 = *a++;
    reg_a1 = *a++;
    reg_a2 = *a++;
    reg_a3 = *a++;

    // first row
    c[0] += reg_a0 * reg_b0;
    c[1] += reg_a0 * reg_b1;
    c[2] += reg_a0 * reg_b2;
    c[3] += reg_a0 * reg_b3;

    // second row
    c[4] += reg_a1 * reg_b0;
    c[5] += reg_a1 * reg_b1;
    c[6] += reg_a1 * reg_b2;
    c[7] += reg_a1 * reg_b3;

    // third row
    c[8] += reg_a2 * reg_b0;
    c[9] += reg_a2 * reg_b1;
    c[10] += reg_a2 * reg_b2;
    c[11] += reg_a2 * reg_b3;

    // fourth row
    c[12] += reg_a3 * reg_b0;
    c[13] += reg_a3 * reg_b1;
    c[14] += reg_a3 * reg_b2;
    c[15] += reg_a3 * reg_b3;
  }
  int i, j;
  for (i = 0; i < mc; ++i) {
    for (j = 0; j < nc; ++j) {
      if (beta == 0.0) {
        C(i, j) = 0.0;
      } else if (beta != 1.0) {
        C(i, j) *= beta;
      }
      if (alpha != 1.0) {
        C(i, j) += alpha * c[i * MR + j];
      } else {
        C(i, j) += c[i * MR + j];
      }
    }
  }
}

void AddDot4x4_relu(int k, float alpha, const float *a, int lda, const float *b,
                    int ldb, float beta, float *C, int ldc, int mc, int nc,
                    bool relu) {
  float c[16] = {0};
  float reg_a0, reg_a1, reg_a2, reg_a3, reg_b0, reg_b1, reg_b2, reg_b3;

  for (int p = 0; p < k; p += 1) {
    reg_b0 = *b++;
    reg_b1 = *b++;
    reg_b2 = *b++;
    reg_b3 = *b++;

    reg_a0 = *a++;
    reg_a1 = *a++;
    reg_a2 = *a++;
    reg_a3 = *a++;

    // first row
    c[0] += reg_a0 * reg_b0;
    c[1] += reg_a0 * reg_b1;
    c[2] += reg_a0 * reg_b2;
    c[3] += reg_a0 * reg_b3;

    // second row
    c[4] += reg_a1 * reg_b0;
    c[5] += reg_a1 * reg_b1;
    c[6] += reg_a1 * reg_b2;
    c[7] += reg_a1 * reg_b3;

    // third row
    c[8] += reg_a2 * reg_b0;
    c[9] += reg_a2 * reg_b1;
    c[10] += reg_a2 * reg_b2;
    c[11] += reg_a2 * reg_b3;

    // fourth row
    c[12] += reg_a3 * reg_b0;
    c[13] += reg_a3 * reg_b1;
    c[14] += reg_a3 * reg_b2;
    c[15] += reg_a3 * reg_b3;
  }
  int i, j;
  for (i = 0; i < mc; ++i) {
    for (j = 0; j < nc; ++j) {
      if (beta == 0.0) {
        C(i, j) = 0.0;
      } else if (beta != 1.0) {
        C(i, j) *= beta;
      }
      if (alpha != 1.0) {
        C(i, j) += alpha * c[i * MR + j];
      } else {
        C(i, j) += c[i * MR + j];
      }
      if (relu) {
        if (C(i, j) < 0) {
          C(i, j) = 0;
        }
      }
    }
  }
}

#endif

// 32位 float 矩阵乘法
void sgemm(int m, int n, int k, float alpha, const float *A, int lda,
           const float *B, int ldb, float beta, float *C, int ldc) {
  int i, j, p, mc, nc, kc;
  float beta_;
  for (j = 0; j < n; j += NC) {
    nc = s_min(n - j, NC);
    for (p = 0; p < k; p += KC) {
      kc = s_min(k - p, KC);
      for (i = 0; i < m; i += MC) {
        mc = s_min(m - i, MC);
        if (p != 0) {
          beta_ = 1.0;
        } else {
          beta_ = beta;
        }
        InnerKernel(mc, nc, kc, alpha, &A(i, p), lda, &B(p, j), ldb, beta_,
                    &C(i, j), ldc, i == 0);
      }
    }
  }
}

void sgemm_relu(int m, int n, int k, float alpha, const float *A, int lda,
                const float *B, int ldb, float beta, float *C, int ldc) {
  int i, j, p, mc, nc, kc;
  float beta_;
  for (j = 0; j < n; j += NC) {
    nc = s_min(n - j, NC);
    for (p = 0; p < k; p += KC) {
      kc = s_min(k - p, KC);
      for (i = 0; i < m; i += MC) {
        mc = s_min(m - i, MC);
        if (p != 0) {
          beta_ = 1.0;
        } else {
          beta_ = beta;
        }

        if (p + KC >= k) {
          InnerKernel_relu(mc, nc, kc, alpha, &A(i, p), lda, &B(p, j), ldb,
                           beta_, &C(i, j), ldc, i == 0, true);
        } else {
          InnerKernel(mc, nc, kc, alpha, &A(i, p), lda, &B(p, j), ldb, beta_,
                      &C(i, j), ldc, i == 0);
        }
      }
    }
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
