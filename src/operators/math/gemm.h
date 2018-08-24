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
#include <string>
#include "common/log.h"

// 矩阵取值运算宏，假设矩阵按行存储
#define A(i, j) A[(i)*lda + (j)]
#define B(i, j) B[(i)*ldb + (j)]
#define C(i, j) C[(i)*ldc + (j)]

#if __aarch64__
#define MR 6
#define NR 16
#else
#define MR 6
#define NR 8
#endif

#define s_min(i, j) ((i) < (j) ? (i) : (j))

namespace paddle_mobile {
namespace operators {
namespace math {

/*
// 将 A 矩阵分块复制到连续内存(ColMajor)
void PackMatrixA(int m, int k, int m_tail, const float *A, int lda,
                 float *buffer);

// 将 B 矩阵分块复制到连续内存(ColMajor)
void PackMatrixB(int k, int n, int n_tail, const float *B, int ldb,
                 float *buffer);
*/

// 将 A 矩阵分块复制到连续内存(RowMajor)
void PackMatrixA_4r(int m, int k, int m_tail, const float *A, int lda,
                    float *buffer);
void PackMatrixA_6r(int m, int k, int m_tail, const float *A, int lda,
                    float *buffer);
void PackMatrixA_8r(int m, int k, int m_tail, const float *A, int lda,
                    float *buffer);
void PackMatrixA_omp_6r(int m, int k, int m_tail, const float *A, int lda,
                        float *buffer);
void PackMatrixA_omp_8r(int m, int k, int m_tail, const float *A, int lda,
                        float *buffer);

// 将 B 矩阵分块复制到连续内存(RowMajor)
void PackMatrixB_8c(int k, int n, int n_tail, const float *B, int ldb,
                    float *buffer);
void PackMatrixB_12c(int k, int n, int n_tail, const float *B, int ldb,
                     float *buffer);
void PackMatrixB_16c(int k, int n, int n_tail, const float *B, int ldb,
                     float *buffer);
void PackMatrixB_omp_8c(int k, int n, int n_tail, const float *B, int ldb,
                        float *buffer);
void PackMatrixB_omp_12c(int k, int n, int n_tail, const float *B, int ldb,
                         float *buffer);
void PackMatrixB_omp_16c(int k, int n, int n_tail, const float *B, int ldb,
                         float *buffer);

// 分块矩阵乘法
void InnerKernel(int mc, int nc, float alpha, const float *a, const float *b,
                 float beta, float *c, float *C, int ldc, bool relu);
void InnerKernelWithBias(int mc, int nc, float alpha, const float *a,
                         const float *b, float beta, float *c, float *C,
                         int ldc, bool relu, float *bias);

void InnerKernelWithBn(int mc, int nc, float alpha, const float *a,
                       const float *b, float beta, float *c, float *C, int ldc,
                       bool relu, float *new_scale, float *new_bias);
void InnerKernelWithPRelu(int mc, int nc, const float *a, const float *b,
                          float *c, float *C, int ldc, float *p,
                          std::string mode, float *bias, float *bias1);
/*
// 向量矩阵乘法 (M = 1)
void VectorKernel(int m, int n, int k, float alpha, const float *A, int lda,
                  const float *B, int ldb, float beta, float *C, int ldc,
                  bool relu);

void VectorKernelWithBn(int m, int n, int k, float alpha, const float *A,
                        int lda, const float *B, int ldb, float beta, float *C,
                        int ldc, bool relu, float *new_scale, float *new_bias);
*/

// 计算一个更小的 C 矩阵分块
void AddDot4x4(int k, const float *a, const float *b, float *c, int ldc);
void AddDot4x8(int k, const float *a, const float *b, float *c, int ldc);
void AddDot6x8(int k, const float *a, const float *b, float *c, int ldc);
void AddDot8x12(int k, const float *a, const float *b, float *c, int ldc);
void AddDot6x16(int k, const float *a, const float *b, float *c, int ldc);

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
void WriteWithBn(int mc, int nc, float *c, float *C, int ldc, float *new_scale,
                 float *new_bias);
// C = A * B, batchnorm(C), relu(C)
void WriteWithBnRelu(int mc, int nc, float *c, float *C, int ldc,
                     float *new_scale, float *new_bias);

/*
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
*/

// 32位 float 矩阵乘法
void Sgemm(int m, int n, int k, float alpha, const float *A, int lda,
           const float *B, int ldb, float beta, float *C, int ldc, bool relu,
           float *bias);

// 32位 float 矩阵乘法, 并对结果进行 batchnrom
void SgemmWithBn(int m, int n, int k, float alpha, const float *A, int lda,
                 const float *B, int ldb, float beta, float *C, int ldc,
                 bool relu, float *new_scale, float *new_bias);

void SgemmWithPRelu(int m, int n, int k, const float *A, int lda,
                    const float *B, int ldb, float *C, int ldc, float *p,
                    std::string mode, float *bias, float *bias1);

// 32位 float 矩阵乘法（openmp 多线程版本）
void Sgemm_omp(int m, int n, int k, float alpha, const float *A, int lda,
               const float *B, int ldb, float beta, float *C, int ldc,
               bool relu, float *bias);

// 32位 float 矩阵乘法, 并对结果进行 batchnrom（openmp 多线程版本）
void SgemmWithBn_omp(int m, int n, int k, float alpha, const float *A, int lda,
                     const float *B, int ldb, float beta, float *C, int ldc,
                     bool relu, float *new_scale, float *new_bias);

void SgemmWithPRelu_omp(int m, int n, int k, const float *A, int lda,
                        const float *B, int ldb, float *C, int ldc, float *p,
                        std::string mode, float *bias, float *bias1);

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
