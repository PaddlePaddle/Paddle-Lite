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

void Gemm::PackMatrixB_omp_8c(int32_t k, int32_t n, int32_t n_tail,
                              const int8_t *B, int32_t ldb, int8_t *buffer) {
  const int32_t j_length = n - n_tail;
#pragma omp parallel for num_threads(framework::threads())
  for (int32_t j = 0; j < j_length; j += 8) {
    int8_t *local_buffer = buffer + j * k;
    for (int32_t i = 0; i < k; ++i) {
      const int8_t *b0 = &B(i, j);
#if __ARM_NEON
#if __aarch64__
      // TODO
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
      for (int32_t j = n; j < j_length + 8; ++j) {
        *local_buffer++ = 0;
      }
    }
  }
}

void Gemm::PackMatrixA_omp_4r(int32_t m, int32_t k, int32_t m_tail,
                              const int8_t *A, int32_t lda, int8_t *buffer) {
  const int32_t i_length = m - m_tail;
#pragma omp parallel for num_threads(framework::threads())
  for (int32_t i = 0; i < i_length; i += 4) {
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
    for (int32_t j = 0; j < k; ++j) {
      *local_buffer++ = *a0++;
      *local_buffer++ = *a1++;
      *local_buffer++ = *a2++;
      *local_buffer++ = *a3++;
    }
  }
}

// 8 bits int PackMatrixA_4r
void Gemm::PackMatrixA_omp_4r_16(int32_t m, int32_t k, int32_t m_tail,
                                 const int8_t *A, int32_t lda, int8_t *buffer) {
  const int32_t i_length = m - m_tail;
  const int32_t k_count = k >> 4;
  const int32_t k_tail = k & 15;
#pragma omp parallel for num_threads(framework::threads())
  for (int32_t i = 0; i < i_length; i += 4) {
    const int8_t *a0 = A + i * lda;
    const int8_t *a1 = A + (i + 1) * lda;
    const int8_t *a2 = A + (i + 2) * lda;
    const int8_t *a3 = A + (i + 3) * lda;
    int8_t *local_buffer = buffer + i * KC;
    for (int32_t j = 0; j < k_count; ++j) {
#if __ARM_NEON
#if __aarch64__
    // TODO
#else
      asm volatile(
          "vld1.s8    {d0, d1},   [%[a0]]!         \n\t"
          "vld1.s8    {d2, d3},   [%[a1]]!         \n\t"
          "vld1.s8    {d4, d5},   [%[a2]]!         \n\t"
          "vld1.s8    {d6, d7},   [%[a3]]!         \n\t"
          "vst1.s8    {d0, d1},   [%[local_buffer]]!    \n\t"
          "vst1.s8    {d2, d3},   [%[local_buffer]]!    \n\t"
          "vst1.s8    {d4, d5},   [%[local_buffer]]!    \n\t"
          "vst1.s8    {d6, d7},   [%[local_buffer]]!    \n\t"
          : [local_buffer] "+r"(local_buffer), [a0] "+r"(a0), [a1] "+r"(a1),
            [a2] "+r"(a2), [a3] "+r"(a3)
          :
          : "memory", "q0", "q1", "q2", "q3");
#endif  // __aarch64__
#else
      for (int32_t l = 0; l < 16; ++l) {
        *local_buffer++ = *a0++;
      }
      for (int32_t l = 0; l < 16; ++l) {
        *local_buffer++ = *a1++;
      }
      for (int32_t l = 0; l < 16; ++l) {
        *local_buffer++ = *a2++;
      }
      for (int32_t l = 0; l < 16; ++l) {
        *local_buffer++ = *a3++;
      }
#endif  // __ARM_NEON
    }
    if (k_tail != 0) {
      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *a0++;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }

      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *a1++;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }
      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *a2++;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }
      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *a3++;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }
    }
  }

  if (m_tail != 0) {
    const int8_t *a0 = &A(i_length, 0);
    const int8_t *a1 = a0 + lda;
    const int8_t *a2 = a0 + 2 * lda;
    const int8_t *a3 = a0 + 3 * lda;
    int8_t *local_buffer = buffer + i_length * KC;
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
    for (int32_t j = 0; j < k_count; ++j) {
#if __ARM_NEON
#if __aarch64__
    // TODO
#else
      asm volatile(
          "vld1.s8    {d0, d1},   [%[a0]]!         \n\t"
          "vld1.s8    {d2, d3},   [%[a1]]!         \n\t"
          "vld1.s8    {d4, d5},   [%[a2]]!         \n\t"
          "vld1.s8    {d6, d7},   [%[a3]]!         \n\t"
          "vst1.s8    {d0, d1},   [%[local_buffer]]!    \n\t"
          "vst1.s8    {d2, d3},   [%[local_buffer]]!    \n\t"
          "vst1.s8    {d4, d5},   [%[local_buffer]]!    \n\t"
          "vst1.s8    {d6, d7},   [%[local_buffer]]!    \n\t"
          : [local_buffer] "+r"(local_buffer), [a0] "+r"(a0), [a1] "+r"(a1),
            [a2] "+r"(a2), [a3] "+r"(a3)
          :
          : "memory", "q0", "q1", "q2", "q3");
#endif  // __aarch64__
#else
      for (int32_t l = 0; l < 16; ++l) {
        *local_buffer++ = *a0++;
      }
      for (int32_t l = 0; l < 16; ++l) {
        *local_buffer++ = *a1++;
      }
      for (int32_t l = 0; l < 16; ++l) {
        *local_buffer++ = *a2++;
      }
      for (int32_t l = 0; l < 16; ++l) {
        *local_buffer++ = *a3++;
      }
#endif  // __ARM_NEON
    }
    if (k_tail != 0) {
      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *a0++;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }

      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *a1++;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }
      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *a2++;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }
      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *a3++;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }
    }
  }
}

// 8 bits int PackMatrixB
void Gemm::PackMatrixB_omp_2c_16(int32_t k, int32_t n, int32_t n_tail,
                                 const int8_t *B, int32_t ldb, int8_t *buffer) {
  const int32_t j_length = n - n_tail;
  const int32_t k_count = k >> 4;
  const int32_t k_tail = k & 15;
#pragma omp parallel for num_threads(framework::threads())
  for (int32_t j = 0; j < j_length; j += 2) {
    int8_t *local_buffer = buffer + j * KC;
    for (int32_t i = 0; i < k_count; ++i) {
      const int8_t *b0 = &B((i << 4), j);
      const int8_t *b1 = &B((i << 4), j + 1);
      for (int m = 0; m < 16; ++m) {
        *local_buffer++ = *b0;
        b0 += ldb;
      }
      for (int m = 0; m < 16; ++m) {
        *local_buffer++ = *b1;
        b1 += ldb;
      }
    }
    if (k_tail != 0) {
      const int8_t *b0 = &B((k_count << 4), j);
      const int8_t *b1 = &B((k_count << 4), j + 1);
      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *b0;
        b0 += ldb;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }

      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *b1;
        b1 += ldb;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }
    }
  }
  if (n_tail != 0) {
    int8_t *local_buffer = buffer + j_length * KC;
    for (int32_t i = 0; i < k_count; ++i) {
      const int8_t *b0 = &B((i << 4), j_length);
      for (int m = 0; m < 16; ++m) {
        *local_buffer++ = *b0;
        b0 += ldb;
      }
      for (int m = 0; m < 16; ++m) {
        *local_buffer++ = 0;
      }
    }
    if (k_tail != 0) {
      const int8_t *b0 = &B((k_count << 4), j_length);
      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *b0;
        b0 += ldb;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }
      for (int32_t j = k_count << 4; j < KC; ++j) {
        *local_buffer++ = 0;
      }
    }
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
