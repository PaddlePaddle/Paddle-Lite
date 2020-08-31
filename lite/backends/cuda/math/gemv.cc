// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/backends/cuda/math/gemv.h"

#include <iostream>

#include "lite/core/device_info.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename PTypeIn, typename PTypeOut>
bool Gemv<PTypeIn, PTypeOut>::init(const bool trans,
                                   const int m,
                                   const int n,
                                   const int lda,
                                   const int ldb,
                                   const int ldc,
                                   Context<TARGET(kCUDA)> *ctx) {
  if (cu_handle_ == nullptr) {
    this->exe_stream_ = ctx->exec_stream();
    CUBLAS_CALL(cublasCreate(&cu_handle_));
    CUBLAS_CALL(cublasSetMathMode(cu_handle_, CUBLAS_TENSOR_OP_MATH));
    CUBLAS_CALL(cublasSetStream(cu_handle_, this->exe_stream_));
  }
  m_ = m;
  n_ = n;
  lda_ = lda;
  ldb_ = ldb;
  ldc_ = ldc;
  cu_trans_ = trans ? CUBLAS_OP_N : CUBLAS_OP_T;
  return true;
}

template <>
bool Gemv<float, float>::run(const float alpha,
                             const float beta,
                             const float *a,
                             const float *b,
                             float *c) {
  CUBLAS_CALL(cublasSgemv(
      cu_handle_, cu_trans_, n_, m_, &alpha, a, lda_, b, ldb_, &beta, c, ldc_));
  return true;
}

template <>
bool Gemv<half, half>::run(
    const half alpha, const half beta, const half *a, const half *b, half *c) {
  LOG(FATAL) << "not supported";
  return false;
}

template class Gemv<float, float>;
template class Gemv<half, half>;

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
