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

#include "lite/backends/cuda/math/batched_gemm.h"
#include <iostream>
#include "lite/core/device_info.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename PtypeIn, typename PtypeOut>
bool BatchedGemm<PtypeIn, PtypeOut>::init(const bool trans_a,
                                          const bool trans_b,
                                          const int max_batch_size,
                                          Context<TARGET(kCUDA)> *ctx) {
  if (cu_handle_ == nullptr) {
    this->exe_stream_ = ctx->exec_stream();
    CUBLAS_CALL(cublasCreate(&cu_handle_));
    CUBLAS_CALL(cublasSetStream(cu_handle_, this->exe_stream_));
  }
  cu_trans_a_ = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cu_trans_b_ = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  if (A_ != nullptr) {
    cudaFree(A_);
  }
  cudaMalloc(reinterpret_cast<void **>(&A_),
             3 * max_batch_size * sizeof(PtypeIn *));
  return true;
}

template <>
bool BatchedGemm<float, float>::run(const float alpha,
                                    const float beta,
                                    const float *a[],
                                    const float *b[],
                                    float *c[],
                                    const int m,
                                    const int n,
                                    const int k,
                                    const int batch_size) {
  CHECK(a != nullptr);
  CHECK(b != nullptr);
  CHECK(c != nullptr);
  lda_ = (cu_trans_a_ == CUBLAS_OP_N) ? k : m;
  ldb_ = (cu_trans_b_ == CUBLAS_OP_N) ? n : k;
  ldc_ = n;
  m_ = m;
  n_ = n;
  k_ = k;
  cudaMemcpyAsync(A_,
                  a,
                  batch_size * sizeof(const float *),
                  cudaMemcpyHostToDevice,
                  exe_stream_);
  cudaMemcpyAsync(A_ + batch_size,
                  b,
                  batch_size * sizeof(const float *),
                  cudaMemcpyHostToDevice,
                  exe_stream_);
  cudaMemcpyAsync(A_ + batch_size * 2,
                  c,
                  batch_size * sizeof(float *),
                  cudaMemcpyHostToDevice,
                  exe_stream_);
  CUBLAS_CALL(cublasSgemmBatched(cu_handle_,
                                 cu_trans_b_,
                                 cu_trans_a_,
                                 n_,
                                 m_,
                                 k_,
                                 &alpha,
                                 const_cast<const float **>(A_ + batch_size),
                                 ldb_,
                                 const_cast<const float **>(A_),
                                 lda_,
                                 &beta,
                                 A_ + batch_size * 2,
                                 ldc_,
                                 batch_size));
  return true;
}

template <>
bool BatchedGemm<half, half>::run(const half alpha,
                                  const half beta,
                                  const half *a[],
                                  const half *b[],
                                  half *c[],
                                  const int m,
                                  const int n,
                                  const int k,
                                  const int batch_size) {
  CHECK(a != nullptr);
  CHECK(b != nullptr);
  CHECK(c != nullptr);
  lda_ = (cu_trans_a_ == CUBLAS_OP_N) ? k : m;
  ldb_ = (cu_trans_b_ == CUBLAS_OP_N) ? n : k;
  ldc_ = n;
  m_ = m;
  n_ = n;
  k_ = k;
  cudaMemcpyAsync(A_,
                  a,
                  batch_size * sizeof(const half *),
                  cudaMemcpyHostToDevice,
                  exe_stream_);
  cudaMemcpyAsync(A_ + batch_size,
                  b,
                  batch_size * sizeof(const half *),
                  cudaMemcpyHostToDevice,
                  exe_stream_);
  cudaMemcpyAsync(A_ + batch_size * 2,
                  c,
                  batch_size * sizeof(half *),
                  cudaMemcpyHostToDevice,
                  exe_stream_);
  CUBLAS_CALL(cublasHgemmBatched(cu_handle_,
                                 cu_trans_b_,
                                 cu_trans_a_,
                                 n_,
                                 m_,
                                 k_,
                                 &alpha,
                                 const_cast<const half **>(A_ + batch_size),
                                 ldb_,
                                 const_cast<const half **>(A_),
                                 lda_,
                                 &beta,
                                 A_ + batch_size * 2,
                                 ldc_,
                                 batch_size));
  return true;
}

template <>
bool BatchedGemm<float, float>::run(const float alpha,
                                    const float beta,
                                    const float *a[],
                                    const int m,
                                    const int n,
                                    const int k,
                                    const int batch_size) {
  CHECK(a != nullptr);
  lda_ = (cu_trans_a_ == CUBLAS_OP_N) ? k : m;
  ldb_ = (cu_trans_b_ == CUBLAS_OP_N) ? n : k;
  ldc_ = n;
  m_ = m;
  n_ = n;
  k_ = k;
  cudaMemcpyAsync(A_,
                  a,
                  3 * batch_size * sizeof(const float *),
                  cudaMemcpyDefault,
                  exe_stream_);
  CUBLAS_CALL(cublasSgemmBatched(cu_handle_,
                                 cu_trans_b_,
                                 cu_trans_a_,
                                 n_,
                                 m_,
                                 k_,
                                 &alpha,
                                 const_cast<const float **>(A_ + batch_size),
                                 ldb_,
                                 const_cast<const float **>(A_),
                                 lda_,
                                 &beta,
                                 A_ + batch_size * 2,
                                 ldc_,
                                 batch_size));
  return true;
}

template <>
bool BatchedGemm<half, half>::run(const half alpha,
                                  const half beta,
                                  const half *a[],
                                  const int m,
                                  const int n,
                                  const int k,
                                  const int batch_size) {
  CHECK(a != nullptr);
  lda_ = (cu_trans_a_ == CUBLAS_OP_N) ? k : m;
  ldb_ = (cu_trans_b_ == CUBLAS_OP_N) ? n : k;
  ldc_ = n;
  m_ = m;
  n_ = n;
  k_ = k;
  cudaMemcpyAsync(A_,
                  a,
                  3 * batch_size * sizeof(const half *),
                  cudaMemcpyDefault,
                  exe_stream_);
  CUBLAS_CALL(cublasHgemmBatched(cu_handle_,
                                 cu_trans_b_,
                                 cu_trans_a_,
                                 n_,
                                 m_,
                                 k_,
                                 &alpha,
                                 const_cast<const half **>(A_ + batch_size),
                                 ldb_,
                                 const_cast<const half **>(A_),
                                 lda_,
                                 &beta,
                                 A_ + batch_size * 2,
                                 ldc_,
                                 batch_size));
  return true;
}

template class BatchedGemm<float, float>;
template class BatchedGemm<half, half>;

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
