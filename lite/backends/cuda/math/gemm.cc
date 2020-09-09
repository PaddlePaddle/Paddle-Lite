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

// LtGemm
template <typename T>
class cublasTypeWrapper;

template <>
class cublasTypeWrapper<float> {
 public:
  static const cudaDataType_t type = CUDA_R_32F;
};

template <>
class cublasTypeWrapper<half> {
 public:
  static const cudaDataType_t type = CUDA_R_16F;
};

#if (CUBLAS_VER_MAJOR * 10 + CUBLAS_VER_MINOR) >= 101

template <typename PTypeIn, typename PTypeOut>
bool LtGemm<PTypeIn, PTypeOut>::init(const bool trans_a,
                                     const bool trans_b,
                                     const int m,
                                     const int n,
                                     const int k,
                                     Context<TARGET(kCUDA)> *ctx) {
  int lda = (!trans_a) ? k : m;
  int ldb = (!trans_b) ? n : k;
  int ldc = n;

  return this->init(trans_a, trans_b, m, n, k, lda, ldb, ldc, ctx);
}

template <typename PTypeIn, typename PTypeOut>
bool LtGemm<PTypeIn, PTypeOut>::init(const bool trans_a,
                                     const bool trans_b,
                                     const int m,
                                     const int n,
                                     const int k,
                                     const int lda,
                                     const int ldb,
                                     const int ldc,
                                     Context<TARGET(kCUDA)> *ctx) {
  if (handle_ == nullptr) {
    this->exe_stream_ = ctx->exec_stream();
    CUBLAS_CALL(cublasLtCreate(&handle_));
  }
  m_ = m;
  n_ = n;
  k_ = k;
  lda_ = lda;
  ldb_ = ldb;
  ldc_ = ldc;
  cu_trans_a_ = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cu_trans_b_ = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

  // create operation desciriptor; see cublasLtMatmulDescAttributes_t for
  // details about defaults; here we just need to set the transforms for A and B
  CUBLAS_CALL(cublasLtMatmulDescCreate(&matmul_desc_,
                                       cublasTypeWrapper<PTypeOut>::type));
  CUBLAS_CALL(cublasLtMatmulDescSetAttribute(matmul_desc_,
                                             CUBLASLT_MATMUL_DESC_TRANSA,
                                             &cu_trans_b_,
                                             sizeof(cu_trans_b_)));
  CUBLAS_CALL(cublasLtMatmulDescSetAttribute(matmul_desc_,
                                             CUBLASLT_MATMUL_DESC_TRANSA,
                                             &cu_trans_a_,
                                             sizeof(cu_trans_a_)));

  // create matrix descriptors, we are good with the details here so no need to
  // set any extra attributes
  CUBLAS_CALL(cublasLtMatrixLayoutCreate(&a_desc_,
                                         cublasTypeWrapper<PTypeOut>::type,
                                         trans_a == false ? k : m,
                                         trans_a == false ? m : k,
                                         lda));
  CUBLAS_CALL(cublasLtMatrixLayoutCreate(&b_desc_,
                                         cublasTypeWrapper<PTypeOut>::type,
                                         trans_b == false ? n : k,
                                         trans_b == false ? k : n,
                                         ldb));
  CUBLAS_CALL(cublasLtMatrixLayoutCreate(
      &c_desc_, cublasTypeWrapper<PTypeOut>::type, n, m, ldc));

  // create preference handle; here we could use extra attributes to disable
  // tensor ops or to make sure algo selected will work with badly aligned A, B,
  // C; here for simplicity we just assume A,B,C are always well aligned (e.g.
  // directly come from cudaMalloc)
  CUBLAS_CALL(cublasLtMatmulPreferenceCreate(&preference_));

  if (!workspace_) {
    CUDA_CALL(cudaMalloc(&this->workspace_, workspace_size_));
  }
  CUBLAS_CALL(cublasLtMatmulPreferenceSetAttribute(
      preference_,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_size_,
      sizeof(workspace_size_)));

  // we just need the best available heuristic to try and run matmul. There is
  // no guarantee this will work, e.g. if A is badly aligned, you can request
  // more (e.g. 32) algos and try to run them one by one until something works
  CUBLAS_CALL(cublasLtMatmulAlgoGetHeuristic(handle_,
                                             matmul_desc_,
                                             b_desc_,
                                             a_desc_,
                                             c_desc_,
                                             c_desc_,
                                             preference_,
                                             1,
                                             &heuristic_result_,
                                             &returned_results_));
  if (returned_results_ == 0) {
    LOG(FATAL) << "cuBLAS API failed with status "
               << CUBLAS_STATUS_NOT_SUPPORTED;
  }

  return true;
}

template <typename PTypeIn, typename PTypeOut>
bool LtGemm<PTypeIn, PTypeOut>::run(const PTypeOut alpha,
                                    const PTypeOut beta,
                                    const PTypeIn *a,
                                    const PTypeIn *b,
                                    PTypeOut *c,
                                    Context<TARGET(kCUDA)> *ctx) {
  CUBLAS_CALL(cublasLtMatmul(handle_,
                             matmul_desc_,
                             &alpha,
                             b,
                             b_desc_,
                             a,
                             a_desc_,
                             &beta,
                             c,
                             c_desc_,
                             c,
                             c_desc_,
                             &heuristic_result_.algo,
                             workspace_,
                             workspace_size_,
                             this->exe_stream_));
  return true;
}

template class LtGemm<float, float>;
template class LtGemm<half, half>;

#endif

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
