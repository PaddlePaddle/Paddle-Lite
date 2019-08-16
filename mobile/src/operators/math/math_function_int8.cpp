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

#include <cstring>
#include <string>
#include "operators/math/gemm.h"
#include "operators/math/math_function.h"

namespace paddle_mobile {
namespace operators {
namespace math {

template <>
void MatMul<int8_t, int32_t>(const framework::Tensor &matrix_a, bool trans_a,
                             const framework::Tensor &matrix_b, bool trans_b,
                             float alpha, framework::Tensor *matrix_out,
                             float beta, bool relu, int32_t *bias,
                             bool addOnRow) {
  auto dim_a = matrix_a.dims();
  auto dim_b = matrix_b.dims();
  auto dim_out = matrix_out->dims();
  PADDLE_MOBILE_ENFORCE(
      dim_a.size() == 2 && dim_b.size() == 2 && dim_out.size() == 2,
      "The input and output of MatMul be matrix");

  int32_t M = dim_out[0];
  int32_t N = dim_out[1];
  int32_t K = (!trans_a) ? dim_a[1] : dim_a[0];
  Gemm gemm;

  if (trans_a) {
    int32_t numel = matrix_a.numel();
    int32_t m = matrix_a.dims()[0];
    int32_t n = matrix_a.dims()[1];
    int8_t *tmp = (int8_t *)(matrix_a.data<int8_t>());  // NOLINT
    int8_t *a = static_cast<int8_t *>(
        paddle_mobile::memory::Alloc(sizeof(int8_t) * numel));
    int32_t index = 0;
    for (int32_t j = 0; j < n; j++) {
      for (int32_t i = 0; i < m; i++) {
        a[index++] = tmp[i * n + j];
      }
    }

#ifdef _OPENMP
    if (bias != nullptr) {
      gemm.Sgemm_omp(M, N, K, alpha, a, K, matrix_b.data<int8_t>(), N, beta,
                     matrix_out->data<int8_t>(), N, relu, bias, addOnRow);
    } else {
      gemm.Sgemm_omp(M, N, K, alpha, a, K, matrix_b.data<int8_t>(), N, beta,
                     matrix_out->data<int32_t>(), N, relu, bias, addOnRow);
    }
#else
    if (bias != nullptr) {
      gemm.Sgemm(M, N, K, alpha, a, K, matrix_b.data<int8_t>(), N, beta,
                 matrix_out->data<int8_t>(), N, relu, bias, addOnRow);
    } else {
      gemm.Sgemm(M, N, K, alpha, a, K, matrix_b.data<int8_t>(), N, beta,
                 matrix_out->data<int32_t>(), N, relu, bias, addOnRow);
    }
#endif
  } else {
#ifdef _OPENMP
    if (bias != nullptr) {
      gemm.Sgemm_omp(M, N, K, alpha, matrix_a.data<int8_t>(), K,
                     matrix_b.data<int8_t>(), N, beta,
                     matrix_out->data<int8_t>(), N, relu, bias, addOnRow);
    } else {
      gemm.Sgemm_omp(M, N, K, alpha, matrix_a.data<int8_t>(), K,
                     matrix_b.data<int8_t>(), N, beta,
                     matrix_out->data<int32_t>(), N, relu, bias, addOnRow);
    }
#else
    if (bias != nullptr) {
      gemm.Sgemm(M, N, K, alpha, matrix_a.data<int8_t>(), K,
                 matrix_b.data<int8_t>(), N, beta, matrix_out->data<int8_t>(),
                 N, relu, bias, addOnRow);
    } else {
      gemm.Sgemm(M, N, K, alpha, matrix_a.data<int8_t>(), K,
                 matrix_b.data<int8_t>(), N, beta, matrix_out->data<int32_t>(),
                 N, relu, bias, addOnRow);
    }
#endif
  }
}

template <>
void MatMul<int8_t, int32_t>(const framework::Tensor &matrix_a, bool trans_a,
                             const framework::Tensor &matrix_b, bool trans_b,
                             float alpha, framework::Tensor *matrix_out,
                             float beta, bool relu, int32_t *bias) {
  MatMul<int8_t, int32_t>(matrix_a, trans_a, matrix_b, trans_b, alpha,
                          matrix_out, beta, relu, bias, false);
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
