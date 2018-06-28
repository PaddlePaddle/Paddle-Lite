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

// 矩阵取值运算宏，假设矩阵按行存储
#define A(i, j) A[(i)*lda + (j)]
#define B(i, j) B[(i)*ldb + (j)]
#define C(i, j) C[(i)*ldc + (j)]

// 分块计算的块大小，mc 与 kc 分别对应分块计算时的 m 与 k
#define MC 128
#define KC 128
#define NC 1024
#define MR 4
#define NR 4

#define s_min(i, j) ((i) < (j) ? (i) : (j))

namespace paddle_mobile {
namespace operators {
namespace math {

// 将 A 矩阵分块复制到连续内存(ColMajor)
void PackMatrixA(int m, int k, int m_tail, const float *A, int lda,
                 float *buffer);

// 将 B 矩阵分块复制到连续内存(ColMajor)
void PackMatrixB(int k, int n, int n_tail, const float *B, int ldb,
                 float *buffer);

// 将 A 矩阵分块复制到连续内存(RowMajor)
void PackMatrixA_(int m, int k, int m_tail, const float *A, int lda,
                  float *buffer);

// 将 B 矩阵分块复制到连续内存(RowMajor)
void PackMatrixB_(int k, int n, int n_tail, const float *B, int ldb,
                  float *buffer);

// 分块矩阵乘法
void InnerKernel(int m, int n, int k, float alpha, const float *A, int lda,
                 const float *B, int ldb, float beta, float *C, int ldc,
                 int first_time);

// 向量矩阵乘法 (M = 1)
void VectorKernel(int m, int n, int k, float alpha, const float *A, int lda,
                  const float *B, int ldb, float beta, float *C, int ldc);

// 计算一个更小的 4 * 4 的 C 矩阵分块
void AddDot4x4(int k, float alpha, const float *A, int lda, const float *B,
               int ldb, float beta, float *C, int ldc, int mc, int nc);

void AddDot4x4_relu(int k, float alpha, const float *a, int lda, const float *b,
                    int ldb, float beta, float *C, int ldc, int mc, int nc,
                    bool relu);

// 32位 float 矩阵乘法
void sgemm(int m, int n, int k, float alpha, const float *A, int lda,
           const float *B, int ldb, float beta, float *C, int ldc);

void sgemm_relu(int m, int n, int k, float alpha, const float *A, int lda,
                const float *B, int ldb, float beta, float *C, int ldc);

// 64位 double 矩阵乘法
void dgemm(int m, int n, int k, float alpha, const double *A, int lda,
           const double *B, int ldb, float beta, double *C, int ldc);

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
