/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

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
