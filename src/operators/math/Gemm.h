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

// 矩阵取值运算宏，假设矩阵按列存储
#define A(i, j) A[(j)*lda + (i)]
#define B(i, j) B[(j)*ldb + (i)]
#define C(i, j) C[(j)*ldc + (i)]

// 分块计算的块大小，mc 与 kc 分别对应分块计算时的 m 与 k
#define MC 384
#define KC 384
#define NC 4096
#define MR 4
#define NR 4

#define s_min(i, j) ((i) < (j) ? (i) : (j))

namespace paddle_mobile {
namespace operators {
namespace math {

// 将 A 矩阵分块复制到连续内存
void PackMatrixA(int m, int k, int paddingM, const float *A, int lda,
                 float *buffer);

// 将 B 矩阵分块复制到连续内存
void PackMatrixB(int k, int n, int paddingN, const float *B, int ldb,
                 float *buffer);

// 分块矩阵乘法
void InnerKernel(int m, int n, int k, const float *A, int lda, const float *B,
                 int ldb, float *C, int ldc, int first_time);

// 计算一个更小的 4 * 4 的 C 矩阵分块
void AddDot4x4(int k, const float *A, int lda, const float *B, int ldb,
               float *C, int ldc, int mc, int nc);

// 32位 float 矩阵乘法
void sgemm(int m, int n, int k, float alpha, const float *A, int lda,
           const float *B, int ldb, float beta, float *C, int ldc);

// 64位 double 矩阵乘法
void dgemm(int m, int n, int k, float alpha, const double *A, int lda,
           const double *B, int ldb, float beta, double *C, int ldc);

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
