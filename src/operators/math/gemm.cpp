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

namespace paddle_mobile {
namespace operators {
namespace math {
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
      *buffer++ = *Bij;
      *buffer++ = *(Bij + 1);
      *buffer++ = *(Bij + 2);
      *buffer++ = *(Bij + 3);
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
      AddDot4x4(k, alpha, &packedA[i * k], 4, &packedB[j * k], k, beta,
                &C(i, j), ldc, mc, nc);
    }
  }
}

// 计算一个更小的 4 * 4 的 C 矩阵分块
void AddDot4x4(int k, float alpha, const float *a, int lda, const float *b,
               int ldb, float beta, float *C, int ldc, int mc, int nc) {
  float c[16] = {0};
  float reg_a0, reg_a1, reg_a2, reg_a3, reg_b0, reg_b1, reg_b2, reg_b3;

  // // init C
  // float32x4_t cv0 = vdup_n_f32(0.0);
  // float32x4_t cv1 = vdup_n_f32(0.0);
  // float32x4_t cv2 = vdup_n_f32(0.0);
  // float32x4_t cv3 = vdup_n_f32(0.0);

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

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
