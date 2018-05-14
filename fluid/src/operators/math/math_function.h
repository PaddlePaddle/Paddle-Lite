/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <cmath>
#include <cblas.h>
#include "framework/tensor.h"

namespace paddle_mobile {
namespace operators {
namespace math {

template <typename T>
void gemm(const CBLAS_TRANSPOSE transA,
          const CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
          const T alpha, const T* A, const T* B, const T beta, T* C);

template <typename T>
void gemm(const bool transA, const bool transB,
          const int M, const int N, const int K, const T alpha, const T* A,
          const int lda, const T* B, const int ldb, const T beta, T* C,
          const int ldc);

// matrix multiply with continuous memory
template <typename T>
void matmul(const framework::Tensor& matrix_a,
            bool trans_a, const framework::Tensor& matrix_b, bool trans_b,
            T alpha, framework::Tensor* matrix_out, T beta);
}  // namespace math
}  // namespace operators
}  // namespace paddle
