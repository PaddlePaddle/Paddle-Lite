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

#pragma once
#include <cstring>
#include <string>
#include "common/log.h"
#include "memory/t_malloc.h"
#ifdef _OPENMP
#include <omp.h>
#endif

// 矩阵取值运算宏，假设矩阵按行存储
#define A(i, j) A[(i)*lda + (j)]
#define B(i, j) B[(i)*ldb + (j)]
#define C(i, j) C[(i)*ldc + (j)]

#if __aarch64__
#define MR_INT8 4
#define NR_INT8 2
#define MR 6
#define NR 16
#else
#define MR_INT8 4
#define NR_INT8 2
#define MR 6
#define NR 8
#endif

#define s_min(i, j) ((i) < (j) ? (i) : (j))

namespace paddle_mobile {
namespace operators {
namespace math {

class Gemm {
 public:
  typedef void (Gemm::*FnPack)(int, int, int, const float *, int, float *,
                               const bool);
  typedef void (Gemm::*FnAddDot)(int, const float *, const float *, float *,
                                 int);
  FnPack procPackA;
  FnPack procPackB;
  FnAddDot procAddDot;

  void PackMatrixA_6r(int m, int k, int m_tail, const float *A, int lda,
                      float *buffer, const bool parallel);
  void PackMatrixA_8r(int m, int k, int m_tail, const float *A, int lda,
                      float *buffer, const bool parallel);
  void PackMatrixB_8c(int k, int n, int n_tail, const float *B, int ldb,
                      float *buffer, const bool parallel);
#if __aarch64__
  void PackMatrixB_12c(int k, int n, int n_tail, const float *B, int ldb,
                       float *buffer, const bool parallel);
  void PackMatrixB_16c(int k, int n, int n_tail, const float *B, int ldb,
                       float *buffer, const bool parallel);
#endif

  // 分块矩阵乘法
  void InnerKernel(int mc, int nc, float alpha, const float *a, const float *b,
                   float beta, float *c, float *C, int ldc, bool relu);
  void InnerKernelWithBias(int mc, int nc, float alpha, const float *a,
                           const float *b, float beta, float *c, float *C,
                           int ldc, bool relu, float *bias);

  void InnerKernelWithBn(int mc, int nc, float alpha, const float *a,
                         const float *b, float beta, float *c, float *C,
                         int ldc, bool relu, float *new_scale, float *new_bias);
  void InnerKernelWithBnAdd(int mc, int nc, float alpha, const float *a,
                            const float *b, float beta, float *c, float *C,
                            int ldc, bool relu, float *new_scale,
                            float *new_bias, float *bias);
  void InnerKernelWithPRelu(int mc, int nc, const float *a, const float *b,
                            float *c, float *C, int ldc, float *p,
                            std::string mode, float *bias, float *bias1);

  // 计算一个更小的 C 矩阵分块
#if __aarch64__
  void AddDot6x8(int k, const float *a, const float *b, float *c, int ldc);
  void AddDot8x12(int k, const float *a, const float *b, float *c, int ldc);
  void AddDot6x16(int k, const float *a, const float *b, float *c, int ldc);
#else
  void AddDot4x4(int k, const float *a, const float *b, float *c, int ldc);
  void AddDot4x8(int k, const float *a, const float *b, float *c, int ldc);
  void AddDot6x8(int k, const float *a, const float *b, float *c, int ldc);
#endif

  // 分块矩阵乘法结果回写
  // C = A * B
  void WriteBasic(int mc, int nc, float *c, float *C, int ldc);
  // C = alpha * A * B + beta * C
  void WriteWithAlphaBeta(int mc, int nc, float *c, float *C, int ldc);
  // C = A * B + C
  void WriteWithAdd(int mc, int nc, float *c, float *C, int ldc);
  // C = A * B + bias
  void WriteWithAddV1(int mc, int nc, float *c, float *C, int ldc, float *bias);
  // C = A * B + C, relu(C)
  void WriteWithAddRelu(int mc, int nc, float *c, float *C, int ldc);
  // C = A * B + C,prelu(C)
  void WriteWithAddPRelu(int mc, int nc, float *c, float *C, int ldc, float *p,
                         std::string mode, float *bias, float *bias1);
  // C = A * B + bias ,relu(C)
  void WriteWithAddReluV1(int mc, int nc, float *c, float *C, int ldc,
                          float *bias);
  // C = A * B, batchnorm(C)
  void WriteWithBn(int mc, int nc, float *c, float *C, int ldc,
                   float *new_scale, float *new_bias);
  // C = A * B, batchnorm(C), relu(C)
  void WriteWithBnRelu(int mc, int nc, float *c, float *C, int ldc,
                       float *new_scale, float *new_bias);
  void WriteWithBnAddRelu(int mc, int nc, float *c, float *C, int ldc,
                          float *new_scale, float *new_bias, float *bias1);

  // 向量矩阵乘法 (M = 1)
#if __aarch64__
#else
  void VectorKernel(int m, int n, int k, float alpha, const float *A, int lda,
                    const float *B, int ldb, float beta, float *C, int ldc,
                    bool relu);

  void VectorKernelWithBn(int m, int n, int k, float alpha, const float *A,
                          int lda, const float *B, int ldb, float beta,
                          float *C, int ldc, bool relu, float *new_scale,
                          float *new_bias);

  // 向量矩阵乘法结果回写
  // C = A * B
  void VecWriteBasic(int n, float *c, float *C, int ldc);
  // C = alpha * A * B + beta * C
  void VecWriteWithAlphaBeta(int n, float *c, float *C, int ldc);
  // C = A * B + C
  void VecWriteWithAdd(int n, float *c, float *C, int ldc);
  // C = A * B + C, relu(C)
  void VecWriteWithAddRelu(int n, float *c, float *C, int ldc);
  // C = A * B, batchnorm(C)
  void VecWriteWithBn(int n, float *c, float *C, int ldc, float *new_scale,
                      float *new_bias);
  // C = A * B, batchnorm(C), relu(C)
  void VecWriteWithBnRelu(int n, float *c, float *C, int ldc, float *new_scale,
                          float *new_bias);
#endif

  // 32位 float 矩阵乘法
  void Sgemm(int m, int n, int k, float alpha, const float *A, int lda,
             const float *B, int ldb, float beta, float *C, int ldc, bool relu,
             float *bias);

  // 32位 float 矩阵乘法, 并对结果进行 batchnrom
  void SgemmWithBn(int m, int n, int k, float alpha, const float *A, int lda,
                   const float *B, int ldb, float beta, float *C, int ldc,
                   bool relu, float *new_scale, float *new_bias, float *bias);

  void SgemmWithPRelu(int m, int n, int k, const float *A, int lda,
                      const float *B, int ldb, float *C, int ldc, float *p,
                      std::string mode, float *bias, float *bias1);

  // 32位 float 矩阵乘法（openmp 多线程版本）
  void Sgemm_omp(int m, int n, int k, float alpha, const float *A, int lda,
                 const float *B, int ldb, float beta, float *C, int ldc,
                 bool relu, float *bias);

  // 32位 float 矩阵乘法, 并对结果进行 batchnrom（openmp 多线程版本）
  void SgemmWithBn_omp(int m, int n, int k, float alpha, const float *A,
                       int lda, const float *B, int ldb, float beta, float *C,
                       int ldc, bool relu, float *new_scale, float *new_bias,
                       float *bias);

  void SgemmWithPRelu_omp(int m, int n, int k, const float *A, int lda,
                          const float *B, int ldb, float *C, int ldc, float *p,
                          std::string mode, float *bias, float *bias1);

  // 8 bits function cluster begins
  // 8 bits int small block inner product
  void AddDot4x8(int32_t k, const int8_t *a, const int8_t *b, int32_t *c,
                 int32_t ldc);
  void AddDot4x2(int32_t k, const int8_t *a, const int8_t *b, int32_t *c,
                 int32_t ldc);
  void AddDot6x8(int32_t k, const int8_t *a, const int8_t *b, int32_t *c,
                 int32_t ldc);

  // 8 bits int inner product
  template <typename Otype>
  void InnerKernel(int32_t mc, int32_t nc, float alpha, const int8_t *a,
                   const int8_t *b, float beta, int32_t *c, Otype *C,
                   int32_t ldc, bool relu);
  template <typename Otype>
  void InnerKernelWithBias(int32_t mc, int32_t nc, float alpha, const int8_t *a,
                           const int8_t *b, float beta, int32_t *c, Otype *C,
                           int32_t ldc, bool relu, int32_t *bias,
                           bool addOnRow = false);

  // 8 bits int pack function
  void PackMatrixA_4r(int32_t m, int32_t k, int32_t m_tail, const int8_t *A,
                      int32_t lda, int8_t *buffer);
  void PackMatrixA_4r_16(int32_t m, int32_t k, int32_t m_tail, const int8_t *A,
                         int32_t lda, int8_t *buffer);
  void PackMatrixA_6r(int32_t m, int32_t k, int32_t m_tail, const int8_t *A,
                      int32_t lda, int8_t *buffer);
  void PackMatrixB_2c_16(int32_t k, int32_t n, int32_t n_tail, const int8_t *B,
                         int32_t ldb, int8_t *buffer);
  void PackMatrixB_8c(int32_t k, int32_t n, int32_t n_tail, const int8_t *B,
                      int32_t ldb, int8_t *buffer);
  void PackMatrixA_omp_4r(int32_t m, int32_t k, int32_t m_tail, const int8_t *A,
                          int32_t lda, int8_t *buffer);
  void PackMatrixB_omp_8c(int32_t k, int32_t n, int32_t n_tail, const int8_t *B,
                          int32_t ldb, int8_t *buffer);
  void PackMatrixA_omp_4r_16(int32_t m, int32_t k, int32_t m_tail,
                             const int8_t *A, int32_t lda, int8_t *buffer);
  void PackMatrixB_omp_2c_16(int32_t k, int32_t n, int32_t n_tail,
                             const int8_t *B, int32_t ldb, int8_t *buffer);

  // 8 bits int matrix product
  template <typename Itype, typename Btype, typename Otype>
  void Sgemm_omp(int32_t m, int32_t n, int32_t k, float alpha, const Itype *A,
                 int32_t lda, const Itype *B, int32_t ldb, float beta, Otype *C,
                 int32_t ldc, bool relu, Btype *bias, bool addOnRow = false);
  template <typename Otype>
  void Sgemm_omp(int32_t m, int32_t n, int32_t k, float alpha, const int8_t *A,
                 int32_t lda, const int8_t *B, int32_t ldb, float beta,
                 Otype *C, int32_t ldc, bool relu, int32_t *bias,
                 bool addOnRow = false);
  template <typename Itype, typename Btype, typename Otype>
  void Sgemm(int32_t m, int32_t n, int32_t k, float alpha, const Itype *A,
             int32_t lda, const Itype *B, int32_t ldb, float beta, Otype *C,
             int32_t ldc, bool relu, Btype *bias, bool addOnRow = false);
  template <typename Otype>
  void Sgemm(int32_t m, int32_t n, int32_t k, float alpha, const int8_t *A,
             int32_t lda, const int8_t *B, int32_t ldb, float beta, Otype *C,
             int32_t ldc, bool relu, int32_t *bias, bool addOnRow = false);
  // 8 bits int write back
  // C = A * B
  void WriteBasic(int32_t mc, int32_t nc, int32_t *c, int32_t *C, int32_t ldc);
  // C = A * B + bias, scale * relu(C)
  void WriteWithAddReluScale(int32_t mc, int32_t nc, int32_t *c, int8_t *C,
                             int32_t ldc, int32_t *bias, float scale);
  // C = A * B + bias, scale * C, bias is added on column
  void WriteWithAddScale(int32_t mc, int32_t nc, int32_t *c, int8_t *C,
                         int32_t ldc, int32_t *bias, float scale);
  // C = A * B + bias, scale * C, bias is added on row
  void WriteWithAddScaleT(int32_t mc, int32_t nc, int32_t *c, int8_t *C,
                          int32_t ldc, int32_t *bias, float scale);

 private:
  int MC = 0;
  int KC = 0;
  int NC = 0;

  // 32位 float
  float *packedA;
  float *packedB;
  float *packedC;

  // 8 bits int
  int8_t *packedA_int8;
  int8_t *packedB_int8;
  int32_t *packedC_int32;
  int8_t *zero_int8;
};

// 8 bits int matrix product (m*k x k*n)
template <typename Otype>
void Gemm::Sgemm(int32_t m, int32_t n, int32_t k, float alpha, const int8_t *A,
                 int32_t lda, const int8_t *B, int32_t ldb, float beta,
                 Otype *C, int32_t ldc, bool relu, int32_t *bias,
                 bool addOnRow) {
  // L1 data cache is 32 kib (Per Contex-A57, Contex-A72, Contex-A73)
  // L2 cache is 0.5~4 Mib (Contex-A72 cluster)
  int32_t L1 = 32 * 1024;
  int32_t L2 = 512 * 1024;

  const int32_t k_complete = (k + 15) - ((k + 15) & 15);
  KC = k_complete;
  MC = L1 / (KC * sizeof(int8_t));
  NC = L2 / (KC * sizeof(int8_t));

  // make sure MC is multiple of MR_INT8, and NC is multiple of NR_INT8
  if (MC == 0) {
    MC = MR_INT8;
  } else {
    int32_t mblock_num = (m + MC - 1) / MC;
    MC = (m + mblock_num - 1) / mblock_num;
    MC = (MC + MR_INT8 - 1) / MR_INT8 * MR_INT8;
  }
  // DLOG << "mblock_num = " << mblock_num << ", MC = " << MC << "\n";
  if (NC == 0) {
    NC = NR_INT8;
  } else {
    int32_t nblock_num = (n + NC - 1) / NC;
    NC = (n + nblock_num - 1) / nblock_num;
    NC = (NC + NR_INT8 - 1) / NR_INT8 * NR_INT8;
  }
  //  DLOG << "nblock_num = " << nblock_num << ", NC = " << NC << "\n";
  packedA_int8 = static_cast<int8_t *>(
      paddle_mobile::memory::Alloc(sizeof(int8_t) * MC * KC));
  packedB_int8 = static_cast<int8_t *>(
      paddle_mobile::memory::Alloc(sizeof(int8_t) * KC * NC));
  packedC_int32 = static_cast<int32_t *>(
      paddle_mobile::memory::Alloc(sizeof(int32_t) * MC * NC));
  zero_int8 =
      static_cast<int8_t *>(paddle_mobile::memory::Alloc(sizeof(int8_t) * k));

  memset(static_cast<void *>(zero_int8), 0, sizeof(int8_t) * k);
  int32_t mc, nc;
  for (int32_t j = 0; j < n; j += NC) {
    nc = s_min(n - j, NC);
    PackMatrixB_2c_16(k, nc, nc % NR_INT8, &B(0, j), ldb, packedB_int8);
    for (int32_t i = 0; i < m; i += MC) {
      mc = s_min(m - i, MC);
      PackMatrixA_4r_16(mc, k, mc % MR_INT8, &A(i, 0), lda, packedA_int8);
      if (bias == nullptr) {
        InnerKernel(mc, nc, alpha, packedA_int8, packedB_int8, beta,
                    packedC_int32, &C(i, j), ldc, relu);
      } else {
        if (addOnRow) {
          InnerKernelWithBias(mc, nc, alpha, packedA_int8, packedB_int8, beta,
                              packedC_int32, &C(i, j), ldc, relu, bias + j,
                              addOnRow);
        } else {
          InnerKernelWithBias(mc, nc, alpha, packedA_int8, packedB_int8, beta,
                              packedC_int32, &C(i, j), ldc, relu, bias + i,
                              addOnRow);
        }
      }
    }
  }

  paddle_mobile::memory::Free(packedA_int8);
  paddle_mobile::memory::Free(packedB_int8);
  paddle_mobile::memory::Free(packedC_int32);
  paddle_mobile::memory::Free(zero_int8);
}

// 8 bits int matrix product (m*k x k*n), omp version
template <typename Otype>
void Gemm::Sgemm_omp(int32_t m, int32_t n, int32_t k, float alpha,
                     const int8_t *A, int32_t lda, const int8_t *B, int32_t ldb,
                     float beta, Otype *C, int32_t ldc, bool relu,
                     int32_t *bias, bool addOnRow) {
#ifdef _OPENMP
  int32_t max_threads = omp_get_max_threads();
#else
  int32_t max_threads = 1;
#endif

  int32_t L1 = 64 / max_threads * 1024;
  const int32_t k_complete = (k + 15) - ((k + 15) & 15);
  KC = k_complete;
  zero_int8 =
      static_cast<int8_t *>(paddle_mobile::memory::Alloc(sizeof(int8_t) * k));
  memset(static_cast<void *>(zero_int8), 0, sizeof(int8_t) * k);
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
    NC = (n + NR_INT8 - 1) / NR_INT8 * NR_INT8;

    packedB_int8 = static_cast<int8_t *>(
        paddle_mobile::memory::Alloc(sizeof(int8_t) * KC * NC));
#if __aarch64__
    // TODO(paddle mobile)
#else
    PackMatrixB_omp_2c_16(k, n, n % NR_INT8, B, ldb, packedB_int8);
#endif
    packedA_int8 = static_cast<int8_t *>(
        paddle_mobile::memory::Alloc(sizeof(int8_t) * MC * KC * max_threads));
  } else {
    // 对 B 分块
    NC = L1 / (KC * sizeof(int8_t));
    if (NC == 0) {
      NC = NR_INT8;
    } else {
      int32_t nblock_num = (n + NC - 1) / NC;
      NC = (n + nblock_num - 1) / nblock_num;
      NC = (NC + NR_INT8 - 1) / NR_INT8 * NR_INT8;
    }
    // 补齐 A
    MC = (m + MR_INT8 - 1) / MR_INT8 * MR_INT8;

    packedA_int8 = static_cast<int8_t *>(
        paddle_mobile::memory::Alloc(sizeof(int8_t) * MC * KC));
#if __aarch64__
    // TODO(paddle mobile)
#else
    PackMatrixA_omp_4r_16(m, k, m % MR_INT8, A, lda, packedA_int8);
#endif
    packedB_int8 = static_cast<int8_t *>(
        paddle_mobile::memory::Alloc(sizeof(int8_t) * KC * NC * max_threads));
  }
  packedC_int32 = static_cast<int32_t *>(
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
      int32_t *local_C = packedC_int32 + MC * NC * local_threads;
#if __aarch64__
      // TODO(paddle mobile)
#else
      PackMatrixA_4r_16(mc, k, mc % MR_INT8, &A(i, 0), lda, local_A);
#endif
      if (bias == nullptr) {
        InnerKernel(mc, n, alpha, local_A, packedB_int8, beta, local_C,
                    &C(i, 0), ldc, relu);
      } else {
        if (addOnRow) {
          InnerKernelWithBias(mc, n, alpha, local_A, packedB_int8, beta,
                              local_C, &C(i, 0), ldc, relu, bias, addOnRow);
        } else {
          InnerKernelWithBias(mc, n, alpha, local_A, packedB_int8, beta,
                              local_C, &C(i, 0), ldc, relu, bias + i, addOnRow);
        }
      }
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
      int32_t *local_C = packedC_int32 + MC * NC * local_threads;
#if __aarch64__
      // TODO(paddle mobile)
#else
      PackMatrixB_2c_16(k, nc, nc % NR_INT8, &B(0, j), ldb, local_B);
#endif
      if (bias == nullptr) {
        InnerKernel(m, nc, alpha, packedA_int8, local_B, beta, local_C,
                    &C(0, j), ldc, relu);
      } else {
        if (addOnRow) {
          InnerKernelWithBias(m, nc, alpha, packedA_int8, local_B, beta,
                              local_C, &C(0, j), ldc, relu, bias + j, addOnRow);
        } else {
          InnerKernelWithBias(m, nc, alpha, packedA_int8, local_B, beta,
                              local_C, &C(0, j), ldc, relu, bias, addOnRow);
        }
      }
    }
  }

  paddle_mobile::memory::Free(packedA_int8);
  paddle_mobile::memory::Free(packedB_int8);
  paddle_mobile::memory::Free(packedC_int32);
  paddle_mobile::memory::Free(zero_int8);
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
