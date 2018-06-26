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

#include "operators/math/math_function.h"
#include "operators/math/gemm.h"

namespace paddle_mobile {
namespace operators {
namespace math {

template <>
void matmul<float>(const framework::Tensor &matrix_a, bool trans_a,
                   const framework::Tensor &matrix_b, bool trans_b, float alpha,
                   framework::Tensor *matrix_out, float beta, bool relu) {
  auto dim_a = matrix_a.dims();
  auto dim_b = matrix_b.dims();
  auto dim_out = matrix_out->dims();
  //  PADDLE_ENFORCE(dim_a.size() == 2 && dim_b.size() == 2 &&
  //  dim_out.size() ==
  //  2,
  //                 "The input and output of matmul be matrix");
  //
  //  PADDLE_ENFORCE(platform::is_cpu_place(matrix_a.place()) &&
  //                     platform::is_cpu_place(matrix_b.place())
  //                     &&
  //                     platform::is_cpu_place(matrix_out->place()),
  //                 "Matrix must all be in CPUPlace");

  int M = dim_out[0];
  int N = dim_out[1];
  int K = (trans_a == false) ? dim_a[1] : dim_a[0];

  if (relu) {
    sgemm_relu(M, N, K, alpha, matrix_a.data<float>(), K,
               matrix_b.data<float>(), N, beta, matrix_out->data<float>(), N);
  } else {
    sgemm(M, N, K, alpha, matrix_a.data<float>(), K, matrix_b.data<float>(), N,
          beta, matrix_out->data<float>(), N);
  }
}

template <>
void matmul<double>(const framework::Tensor &matrix_a, bool trans_a,
                    const framework::Tensor &matrix_b, bool trans_b,
                    double alpha, framework::Tensor *matrix_out, double beta,
                    bool relu) {
  auto dim_a = matrix_a.dims();
  auto dim_b = matrix_b.dims();
  auto dim_out = matrix_out->dims();
  //  PADDLE_ENFORCE(dim_a.size() == 2 && dim_b.size() == 2 &&
  //  dim_out.size() ==
  //  2,
  //                 "The input and output of matmul be matrix");
  //
  //  PADDLE_ENFORCE(platform::is_cpu_place(matrix_a.place()) &&
  //                     platform::is_cpu_place(matrix_b.place())
  //                     &&
  //                     platform::is_cpu_place(matrix_out->place()),
  //                 "Matrix must all be in CPUPlace");

  int M = dim_out[0];
  int N = dim_out[1];
  int K = (trans_a == false) ? dim_a[1] : dim_a[0];
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
