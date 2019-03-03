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

namespace paddle_mobile {
namespace operators {
namespace math {

void cblas_sgemm(const bool transA, const bool transB, const int M, const int N,
                 const int K, const float alpha, const float *A, const int lda,
                 const float *B, const int ldb, const float beta, float *C,
                 const int ldc);

void cblas_sgemv(const bool trans, const int M, const int N, const float alpha,
                 const float *A, const int lda, const float *B,
                 const float beta, float *C);

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
