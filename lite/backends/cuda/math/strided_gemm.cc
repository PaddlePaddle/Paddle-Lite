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

#include "lite/backends/cuda/math/strided_gemm.h"

#include <iostream>

#include "lite/core/device_info.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename PtypeIn, typename PtypeOut>
bool StridedGemm<PtypeIn, PtypeOut>::init(const bool trans_a,
                                          const bool trans_b,
                                          Context<TARGET(kCUDA)>* ctx) {
  if (cu_handle_ == nullptr) {
    this->exe_stream_ = ctx->exec_stream();
    CUBLAS_CALL(cublasCreate(&cu_handle_));
    CUBLAS_CALL(cublasSetStream(cu_handle_, this->exe_stream_));
  }
  cu_trans_a_ = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cu_trans_b_ = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  return true;
}

template <>
bool StridedGemm<float, float>::run(const float alpha,
                                    const float beta,
                                    const int m,
                                    const int n,
                                    const int k,
                                    const float* a_data,
                                    const float* b_data,
                                    float* c_data,
                                    const int batch_size,
                                    const int64_t stride_a,
                                    const int64_t stride_b) {
  lda_ = (cu_trans_a_ == CUBLAS_OP_N) ? k : m;
  ldb_ = (cu_trans_b_ == CUBLAS_OP_N) ? n : k;
  ldc_ = n;
  m_ = m;
  n_ = n;
  k_ = k;
  const int64_t stride_c = m_ * n_;
  CUBLAS_CALL(cublasGemmStridedBatchedEx(cu_handle_,
                                         cu_trans_b_,
                                         cu_trans_a_,
                                         n_,
                                         m_,
                                         k_,
                                         &alpha,
                                         b_data,
                                         CUDA_R_32F,
                                         ldb_,
                                         stride_b,
                                         a_data,
                                         CUDA_R_32F,
                                         lda_,
                                         stride_a,
                                         &beta,
                                         c_data,
                                         CUDA_R_32F,
                                         ldc_,
                                         stride_c,
                                         batch_size,
                                         CUDA_R_32F,
                                         algo_));
  return true;
}

template <>
bool StridedGemm<half, half>::run(const half alpha,
                                  const half beta,
                                  const int m,
                                  const int n,
                                  const int k,
                                  const half* a_data,
                                  const half* b_data,
                                  half* c_data,
                                  const int batch_size,
                                  const int64_t stride_a,
                                  const int64_t stride_b) {
  lda_ = (cu_trans_a_ == CUBLAS_OP_N) ? k : m;
  ldb_ = (cu_trans_b_ == CUBLAS_OP_N) ? n : k;
  ldc_ = n;
  m_ = m;
  n_ = n;
  k_ = k;
  const int64_t stride_c = m_ * n_;
  CUBLAS_CALL(cublasGemmStridedBatchedEx(cu_handle_,
                                         cu_trans_b_,
                                         cu_trans_a_,
                                         n_,
                                         m_,
                                         k_,
                                         &alpha,
                                         b_data,
                                         CUDA_R_16F,
                                         ldb_,
                                         stride_b,
                                         a_data,
                                         CUDA_R_16F,
                                         lda_,
                                         stride_a,
                                         &beta,
                                         c_data,
                                         CUDA_R_16F,
                                         ldc_,
                                         stride_c,
                                         batch_size,
                                         CUDA_R_16F,
                                         algo_));
  return true;
}

template class StridedGemm<float, float>;
template class StridedGemm<half, half>;

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
