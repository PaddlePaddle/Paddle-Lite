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

#pragma once
#include <string>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/core/context.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename PtypeIn, typename PtypeOut>
class Gemm {
 public:
  Gemm() : cu_handle_(nullptr) {}
  ~Gemm() {}
  bool init(const bool trans_a,
            const bool trans_b,
            const int m,
            const int n,
            const int k,
            Context<TARGET(kCUDA)>* ctx);
  bool init(const bool trans_a,
            const bool trans_b,
            const int m,
            const int n,
            const int k,
            const int lda,
            const int ldb,
            const int ldc,
            Context<TARGET(kCUDA)>* ctx);

  bool run(const PtypeOut alpha,
           const PtypeOut beta,
           const PtypeIn* a,
           const PtypeIn* b,
           PtypeOut* c,
           Context<TARGET(kCUDA)>* ctx);

  cublasHandle_t get_handle() const { return cu_handle_; }

 private:
  cudaStream_t exe_stream_;
  cublasHandle_t cu_handle_;
  cublasOperation_t cu_trans_a_;
  cublasOperation_t cu_trans_b_;
  int m_{-1};
  int n_{-1};
  int k_{-1};
  int lda_{-1};
  int ldb_{-1};
  int ldc_{-1};
};

#if (CUBLAS_VER_MAJOR * 10 + CUBLAS_VER_MINOR) >= 101

template <typename PtypeIn, typename PtypeOut>
class LtGemm {
 public:
  LtGemm()
      : handle_(nullptr),
        matmul_desc_(nullptr),
        a_desc_(nullptr),
        b_desc_(nullptr),
        c_desc_(nullptr),
        preference_(nullptr),
        returned_results_(0),
        workspace_size_(4 * 1024 * 1024),
        workspace_{nullptr} {}

  ~LtGemm() {
    if (this->workspace_) {
      CUDA_CALL(cudaFree(this->workspace_));
    }
    this->workspace_ = nullptr;
  }
  bool init(const bool trans_a,
            const bool trans_b,
            const int m,
            const int n,
            const int k,
            Context<TARGET(kCUDA)>* ctx);
  bool init(const bool trans_a,
            const bool trans_b,
            const int m,
            const int n,
            const int k,
            const int lda,
            const int ldb,
            const int ldc,
            Context<TARGET(kCUDA)>* ctx);

  bool run(const PtypeOut alpha,
           const PtypeOut beta,
           const PtypeIn* a,
           const PtypeIn* b,
           PtypeOut* c,
           Context<TARGET(kCUDA)>* ctx);

  cublasLtHandle_t get_handle() const { return handle_; }

 private:
  cudaStream_t exe_stream_;

  cublasLtHandle_t handle_;
  cublasLtMatmulDesc_t matmul_desc_;
  cublasLtMatrixLayout_t a_desc_;
  cublasLtMatrixLayout_t b_desc_;
  cublasLtMatrixLayout_t c_desc_;
  cublasLtMatmulPreference_t preference_;
  int returned_results_;
  cublasLtMatmulHeuristicResult_t heuristic_result_{};

  cublasOperation_t cu_trans_a_;
  cublasOperation_t cu_trans_b_;
  int m_{-1};
  int n_{-1};
  int k_{-1};
  int lda_{-1};
  int ldb_{-1};
  int ldc_{-1};

  size_t workspace_size_;
  void* workspace_;
};
#endif

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
