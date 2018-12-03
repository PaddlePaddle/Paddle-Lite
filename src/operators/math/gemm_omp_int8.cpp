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

#include <string.h>
#include "common/log.h"
#include "memory/t_malloc.h"
#include "operators/math/gemm.h"
#if __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace paddle_mobile {
namespace operators {
namespace math {

// 8 bits int matrix product (m*k x k*n)
void Gemm::Sgemm_omp(int32_t m, int32_t n, int32_t k, int8_t alpha,
                     const int8_t *A, int32_t lda, const int8_t *B, int32_t ldb,
                     int8_t beta, int32_t *C, int32_t ldc, bool relu,
                     int8_t *bias) {
#ifdef _OPENMP
  int32_t max_threads = omp_get_max_threads();
#else
  int32_t max_threads = 1;
#endif

  int32_t L1 = 64 / max_threads * 1024;
  KC = k;
  zero_int8 =
      static_cast<int8_t *>(paddle_mobile::memory::Alloc(sizeof(int8_t) * KC));
  memset(static_cast<void *>(zero_int8), 0, sizeof(int8_t) * KC);
  if (m > n) {
    // 对 A 分块
    MC = L1 / (KC * sizeof(int8_t));
    if (MC == 0) {
      MC = MR_INT8;
    } else {
      int32_t mblock_num = (m + MC - 1) / MC;
      MC = (m + mblock_num - 1) / mblock_num;
      MC = (MC + MR_INT8 - 1) / MR_INT8 * MR_INT8;
    }
    // 补齐 B
    NC = (n + NR - 1) / NR * NR;

    packedB_int8 = static_cast<int8_t *>(
        paddle_mobile::memory::Alloc(sizeof(int8_t) * KC * NC));
#if __aarch64__
    // TODO(wzzju)
#else
    PackMatrixB_omp_8c(KC, n, n % NR, B, ldb, packedB_int8);
#endif
    packedA_int8 = static_cast<int8_t *>(
        paddle_mobile::memory::Alloc(sizeof(int8_t) * MC * KC * max_threads));
  } else {
    // 对 B 分块
    NC = L1 / (KC * sizeof(int8_t));
    if (NC == 0) {
      NC = NR;
    } else {
      int32_t nblock_num = (n + NC - 1) / NC;
      NC = (n + nblock_num - 1) / nblock_num;
      NC = (NC + NR - 1) / NR * NR;
    }
    // 补齐 A
    MC = (m + MR_INT8 - 1) / MR_INT8 * MR_INT8;

    packedA_int8 = static_cast<int8_t *>(
        paddle_mobile::memory::Alloc(sizeof(int8_t) * MC * KC));
#if __aarch64__
    // TODO(wzzju)
#else
    PackMatrixA_omp_4r(m, KC, m % MR_INT8, A, lda, packedA_int8);
#endif
    packedB_int8 = static_cast<int8_t *>(
        paddle_mobile::memory::Alloc(sizeof(int8_t) * KC * NC * max_threads));
  }
  packedC_int8 = static_cast<int32_t *>(
      paddle_mobile::memory::Alloc(sizeof(int32_t) * MC * NC * max_threads));

  if (m > n) {
#pragma omp parallel for
    for (int32_t i = 0; i < m; i += MC) {
#ifdef _OPENMP
      int32_t local_threads = omp_get_thread_num();
#else
      int32_t local_threads = 0;
#endif

      int32_t mc;
      mc = s_min(m - i, MC);
      int8_t *local_A = packedA_int8 + MC * KC * local_threads;
      int32_t *local_C = packedC_int8 + MC * NC * local_threads;
#if __aarch64__
      // TODO(wzzju)
#else
      PackMatrixA_4r(mc, KC, mc % MR_INT8, &A(i, 0), lda, local_A);
#endif
      InnerKernelWithBias(mc, n, alpha, local_A, packedB_int8, beta, local_C,
                          &C(i, 0), ldc, relu, bias + i);
    }
  } else {
#pragma omp parallel for
    for (int32_t j = 0; j < n; j += NC) {
#ifdef _OPENMP
      int32_t local_threads = omp_get_thread_num();
#else
      int32_t local_threads = 0;
#endif
      int32_t nc;
      nc = s_min(n - j, NC);
      int8_t *local_B = packedB_int8 + KC * NC * local_threads;
      int32_t *local_C = packedC_int8 + MC * NC * local_threads;
#if __aarch64__
      // TODO(wzzju)
#else
      PackMatrixB_8c(KC, nc, nc % NR, &B(0, j), ldb, local_B);
#endif
      InnerKernelWithBias(m, nc, alpha, packedA_int8, local_B, beta, local_C,
                          &C(0, j), ldc, relu, bias);
    }
  }

  paddle_mobile::memory::Free(packedA_int8);
  paddle_mobile::memory::Free(packedB_int8);
  paddle_mobile::memory::Free(packedC_int8);
  paddle_mobile::memory::Free(zero_int8);
}

void Gemm::PackMatrixB_omp_8c(int32_t k, int32_t n, int32_t n_tail,
                              const int8_t *B, int32_t ldb, int8_t *buffer) {
  const int32_t j_length = n - n_tail;
#pragma omp parallel for
  for (int32_t j = 0; j < j_length; j += NR) {
    int8_t *local_buffer = buffer + j * k;
    for (int32_t i = 0; i < k; ++i) {
      const int8_t *b0 = &B(i, j);
#if __ARM_NEON
#if __aarch64__
      // TODO(wzzju)
#else
      asm volatile(
          //          "pld        [%[b0]]                     \n\t"
          "vld1.s8    {d0},   [%[b0]]         \n\t"
          "vst1.s8    {d0},   [%[local_buffer]]!    \n\t"
          : [local_buffer] "+r"(local_buffer)
          : [b0] "r"(b0)
          : "memory", "q0");
#endif  // __aarch64__
#else
      *local_buffer++ = *b0++;
      *local_buffer++ = *b0++;
      *local_buffer++ = *b0++;
      *local_buffer++ = *b0++;
      *local_buffer++ = *b0++;
      *local_buffer++ = *b0++;
      *local_buffer++ = *b0++;
      *local_buffer++ = *b0++;
#endif  // __ARM_NEON
    }
  }
  if (n_tail != 0) {
    int8_t *local_buffer = buffer + j_length * k;
    for (int32_t i = 0; i < k; ++i) {
      const int8_t *b0 = &B(i, j_length);
      for (int32_t j = j_length; j < n; ++j) {
        *local_buffer++ = *b0++;
      }
      for (int32_t j = n; j < j_length + NR; ++j) {
        *local_buffer++ = 0;
      }
    }
  }
}

void Gemm::PackMatrixA_omp_4r(int32_t m, int32_t k, int32_t m_tail,
                              const int8_t *A, int32_t lda, int8_t *buffer) {
  const int i_length = m - m_tail;
#pragma omp parallel for
  for (int32_t i = 0; i < i_length; i += MR_INT8) {
    const int8_t *a0 = A + i * lda;
    const int8_t *a1 = A + (i + 1) * lda;
    const int8_t *a2 = A + (i + 2) * lda;
    const int8_t *a3 = A + (i + 3) * lda;
    int8_t *local_buffer = buffer + i * k;
    for (int32_t j = 0; j < k; ++j) {
      *local_buffer++ = *a0++;
      *local_buffer++ = *a1++;
      *local_buffer++ = *a2++;
      *local_buffer++ = *a3++;
    }
  }

  if (m_tail != 0) {
    const int8_t *a0 = &A(i_length, 0);
    const int8_t *a1 = a0 + lda;
    const int8_t *a2 = a0 + 2 * lda;
    const int8_t *a3 = a0 + 3 * lda;
    int8_t *local_buffer = buffer + i_length * k;
    switch (m_tail) {
      case 1:
        a1 = zero_int8;
      case 2:
        a2 = zero_int8;
      case 3:
        a3 = zero_int8;
        break;
      default:
        break;
    }
    for (int j = 0; j < k; ++j) {
      *local_buffer++ = *a0++;
      *local_buffer++ = *a1++;
      *local_buffer++ = *a2++;
      *local_buffer++ = *a3++;
    }
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
