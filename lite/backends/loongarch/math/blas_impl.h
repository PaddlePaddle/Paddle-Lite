//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include <cmath>
#include <limits>
#include <vector>
#include "lite/backends/loongarch/math/math_function.h"

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {

template <typename T>
struct CBlas;

template <>
struct CBlas<float> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    cblas_sgemm(args...);
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    cblas_saxpy(args...);
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    cblas_scopy(args...);
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    cblas_sgemv(args...);
  }
};

template <>
struct CBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    cblas_dgemm(args...);
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    cblas_daxpy(args...);
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    cblas_dcopy(args...);
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    cblas_dgemv(args...);
  }
};

template <>
struct CBlas<lite::fluid::float16> {
  static void GEMM(...) { LOG(FATAL) << "float16 GEMM not supported on CPU"; }
  static void VMUL(...) { LOG(FATAL) << "float16 VMUL not supported on CPU"; }
  static void VEXP(...) { LOG(FATAL) << "float16 VEXP not supported on CPU"; }
  static void VSQUARE(...) {
    LOG(FATAL) << "float16 VSQUARE not supported on CPU";
  }
  static void VPOW(...) { LOG(FATAL) << "float16 VPOW not supported on CPU"; }
  static void DOT(...) { LOG(FATAL) << "float16 DOT not supported on CPU"; };
  static void SCAL(...) { LOG(FATAL) << "float16 SCAL not supported on CPU"; };
  static void ASUM(...) { LOG(FATAL) << "float16 ASUM not supported on CPU"; };
};

template <>
template <typename T>
void Blas<lite::TargetType::kLoongArch>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        T alpha,
                                        const T *A,
                                        const T *B,
                                        T beta,
                                        T *C) const {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  CBlas<T>::GEMM(CblasRowMajor,
                 transA,
                 transB,
                 M,
                 N,
                 K,
                 alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
}

template <>
template <typename T>
void Blas<lite::TargetType::kLoongArch>::GEMM(bool transA,
                                        bool transB,
                                        int M,
                                        int N,
                                        int K,
                                        T alpha,
                                        const T *A,
                                        int lda,
                                        const T *B,
                                        int ldb,
                                        T beta,
                                        T *C,
                                        int ldc) const {
  CBlas<T>::GEMM(CblasRowMajor,
                 transA == false ? CblasNoTrans : CblasTrans,
                 transB == false ? CblasNoTrans : CblasTrans,
                 M,
                 N,
                 K,
                 alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
}

template <>
template <typename T>
void Blas<lite::TargetType::kLoongArch>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        T alpha,
                                        const T *A,
                                        int lda,
                                        const T *B,
                                        int ldb,
                                        T beta,
                                        T *C,
                                        int ldc) const {
  CBlas<T>::GEMM(CblasRowMajor,
                 transA,
                 transB,
                 M,
                 N,
                 K,
                 alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
}

template <lite::TargetType Target>
template <typename T>
void Blas<Target>::MatMul(const lite::Tensor &mat_a,
                          bool trans_a,
                          const lite::Tensor &mat_b,
                          bool trans_b,
                          T alpha,
                          lite::Tensor *mat_out,
                          T beta) const {
  auto dim_a = mat_a.dims();
  auto dim_b = mat_b.dims();
  auto dim_out = mat_out->dims();
  CHECK(dim_a.size() == 2 && dim_b.size() == 2 && dim_out.size() == 2)
      << "The input and output of matmul be matrix";
  // CHECK(
  //    mat_a.target() == mat_b.target() && mat_a.target() == mat_out->target())
  //    << "The targets of matrices must be same";

  int M = dim_out[0];
  int N = dim_out[1];
  int K = !trans_a ? dim_a[1] : dim_a[0];

  CBLAS_TRANSPOSE transA = !trans_a ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !trans_b ? CblasNoTrans : CblasTrans;

  this->GEMM(transA,
             transB,
             M,
             N,
             K,
             alpha,
             mat_a.data<T>(),
             mat_b.data<T>(),
             beta,
             mat_out->template mutable_data<T>());
}

template <>
template <typename T>
void Blas<lite::TargetType::kLoongArch>::AXPY(int n,
                                        T alpha,
                                        const T *x,
                                        T *y) const {
  CBlas<T>::AXPY(n, alpha, x, 1, y, 1);
}

template <>
template <typename T>
void Blas<lite::TargetType::kLoongArch>::VCOPY(int n, const T *x, T *y) const {
  CBlas<T>::VCOPY(n, x, 1, y, 1);
}

template <>
template <typename T>
void Blas<lite::TargetType::kLoongArch>::VADD(int n,
                                        const T *x,
                                        const T *y,
                                        T *z) const {
  this->template VCOPY<T>(n, y, z);
  this->template AXPY<T>(n, 1., x, z);
}

template <>
template <typename T>
void Blas<lite::TargetType::kLoongArch>::VMUL(int n,
                                        const T *x,
                                        const T *y,
                                        T *z) const {
  // try to find if openblas support vmul
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] * y[i];
  }
}

template <>
template <typename T>
void Blas<lite::TargetType::kLoongArch>::VEXP(int n, const T *x, T *y) const {
  // try to find if openblas support vexp
  for (int i = 0; i < n; ++i) {
    y[i] = std::exp(x[i]);
  }
}

template <>
template <typename T>
void Blas<lite::TargetType::kLoongArch>::VSQUARE(int n, const T *x, T *y) const {
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] * x[i];
  }
}

template <>
template <typename T>
void Blas<lite::TargetType::kLoongArch>::VPOW(int n, const T *x, T a, T *y) const {
  for (int i = 0; i < n; ++i) {
    y[i] = std::pow(x[i], a);
  }
}

template <>
template <typename T>
T Blas<lite::TargetType::kLoongArch>::DOT(int n, const T *x, const T *y) const {
  // try to find if openblas support cblas_dot
  T sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += x[i] * y[i];
  }
  return sum;
}

template <>
template <typename T>
void Blas<lite::TargetType::kLoongArch>::SCAL(int n, const T a, T *x) const {
  // try to find if openblas support cblas_scal
  for (int i = 0; i < n; ++i) {
    x[i] = a * x[i];
  }
}

template <>
template <typename T>
T Blas<lite::TargetType::kLoongArch>::ASUM(int n, T *x, int inc) const {
  auto sum = static_cast<T>(0.0);
  // TODO(jczaja): check if openblas does provide cblas_sasum/cblas_dasum
  for (int c = 0; c < n; ++c) {
    sum += x[c];
  }
  return sum;
}

template <>
template <typename T>
void Blas<lite::TargetType::kLoongArch>::GEMV(bool trans_a,
                                        int M,
                                        int N,
                                        T alpha,
                                        const T *A,
                                        const T *B,
                                        T beta,
                                        T *C) const {
  CBLAS_TRANSPOSE transA = !trans_a ? CblasNoTrans : CblasTrans;
  CBlas<T>::GEMV(CblasRowMajor, transA, M, N, alpha, A, N, B, 1, beta, C, 1);
}

template <>
template <typename T>
void Blas<lite::TargetType::kLoongArch>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               T alpha,
                                               const T *A,
                                               const T *B,
                                               T beta,
                                               T *C,
                                               int batchCount,
                                               int64_t strideA,
                                               int64_t strideB) const {
  for (int k = 0; k < batchCount; ++k) {
    auto *Ak = &A[k * strideA];
    auto *Bk = &B[k * strideB];
    auto *Ck = &C[k * M * N];
    this->template GEMM<T>(transA, transB, M, N, K, alpha, Ak, Bk, beta, Ck);
  }
}

template <lite::TargetType Target>
template <typename T>
void Blas<Target>::MatMul(
    const int M, const int N, const int K, const T *A, const T *B, T *C) const {
  this->template GEMM<T>(CblasRowMajor,
                         CblasNoTrans,
                         CblasNoTrans,
                         M,
                         N,
                         K,
                         static_cast<T>(1),
                         A,
                         K,
                         B,
                         N,
                         static_cast<T>(0),
                         C,
                         N);
}

template <>
template <typename T>
void Blas<lite::TargetType::kLoongArch>::MatMul(
    const int M, const int N, const int K, const T *A, const T *B, T *C) const {
  CBlas<T>::GEMM(CblasRowMajor,
                 CblasNoTrans,
                 CblasNoTrans,
                 M,
                 N,
                 K,
                 static_cast<T>(1),
                 A,
                 K,
                 B,
                 N,
                 static_cast<T>(0),
                 C,
                 N);
}

template <lite::TargetType Target>
template <typename T>
void Blas<Target>::MatMul(const lite::Tensor &mat_a,
                          const MatDescriptor &dim_a,
                          const lite::Tensor &mat_b,
                          const MatDescriptor &dim_b,
                          T alpha,
                          lite::Tensor *mat_out,
                          T beta) const {
  CHECK_EQ(dim_a.width_, dim_b.height_);
  CBLAS_TRANSPOSE transA = !dim_a.trans_ ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !dim_b.trans_ ? CblasNoTrans : CblasTrans;
  if (dim_a.batch_size_ == 0 && dim_b.batch_size_ == 0) {
    this->template GEMM<T>(transA,
                           transB,
                           dim_a.height_,
                           dim_b.width_,
                           dim_a.width_,
                           alpha,
                           mat_a.data<T>(),
                           mat_b.data<T>(),
                           beta,
                           mat_out->template mutable_data<T>());
  } else {
    CHECK(dim_a.batch_size_ == dim_b.batch_size_ || dim_a.batch_size_ == 0 ||
          dim_b.batch_size_ == 0);
    this->template BatchedGEMM<T>(
        transA,
        transB,
        dim_a.height_,
        dim_b.width_,
        dim_a.width_,
        alpha,
        mat_a.data<T>(),
        mat_b.data<T>(),
        beta,
        mat_out->template mutable_data<T>(),
        dim_a.batch_size_ == 0 ? dim_b.batch_size_ : dim_a.batch_size_,
        dim_a.stride_,
        dim_b.stride_);
  }
}
template <lite::TargetType Target>
template <typename T>
void Blas<Target>::VINV(int n, const T *a, T *y) const {
  for (int i = 0; i < n; ++i) {
    y[i] = 1.0 / a[i];
  }
}

template <>
template <typename T>
void Blas<lite::TargetType::kLoongArch>::VMERF(int n,
                                         const T *a,
                                         T *y,
                                         int64_t mode) const {
  for (int i = 0; i < n; ++i) {
    y[i] = std::erf(a[i]);
  }
}

}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
