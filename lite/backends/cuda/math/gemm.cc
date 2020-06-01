// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/backends/cuda/math/gemm.h"
#include <iostream>
#include "lite/core/device_info.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename PTypeIn, typename PTypeOut>
bool Gemm<PTypeIn, PTypeOut>::init(const bool trans_a,
                                   bool trans_b,
                                   const int m,
                                   const int n,
                                   const int k,
                                   Context<TARGET(kCUDA)> *ctx) {
  if (cu_handle_ == nullptr) {
    this->exe_stream_ = ctx->exec_stream();
    CUBLAS_CALL(cublasCreate(&cu_handle_));
    CUBLAS_CALL(cublasSetMathMode(cu_handle_, CUBLAS_TENSOR_OP_MATH));
    CUBLAS_CALL(cublasSetStream(cu_handle_, this->exe_stream_));
  }
  lda_ = (!trans_a) ? k : m;
  ldb_ = (!trans_b) ? n : k;
  ldc_ = n;
  m_ = m;
  n_ = n;
  k_ = k;
  cu_trans_a_ = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cu_trans_b_ = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  return true;
}

template <typename PTypeIn, typename PTypeOut>
bool Gemm<PTypeIn, PTypeOut>::init(const bool trans_a,
                                   bool trans_b,
                                   const int m,
                                   const int n,
                                   const int k,
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
  k_ = k;
  lda_ = lda;
  ldb_ = ldb;
  ldc_ = ldc;
  cu_trans_a_ = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cu_trans_b_ = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  return true;
}

template <>
bool Gemm<float, float>::run(const float alpha,
                             const float beta,
                             const float *a,
                             const float *b,
                             float *c,
                             Context<TARGET(kCUDA)> *ctx) {
  CUBLAS_CALL(cublasSgemm(cu_handle_,
                          cu_trans_b_,
                          cu_trans_a_,
                          n_,
                          m_,
                          k_,
                          &alpha,
                          b,
                          ldb_,
                          a,
                          lda_,
                          &beta,
                          c,
                          ldc_));
  return true;
}

template <>
bool Gemm<half, half>::run(const half alpha,
                           const half beta,
                           const half *a,
                           const half *b,
                           half *c,
                           Context<TARGET(kCUDA)> *ctx) {
  CUBLAS_CALL(cublasHgemm(cu_handle_,
                          cu_trans_b_,
                          cu_trans_a_,
                          n_,
                          m_,
                          k_,
                          &alpha,
                          b,
                          ldb_,
                          a,
                          lda_,
                          &beta,
                          c,
                          ldc_));
  return true;
}

template class Gemm<float, float>;
template class Gemm<half, half>;

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
