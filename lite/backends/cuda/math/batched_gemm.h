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
#include <cudnn.h>
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
class BatchedGemm {
 public:
  BatchedGemm() : cu_handle_(nullptr) {}
  ~BatchedGemm() {
    if (A_ != nullptr) {
      cudaFree(A_);
    }
  }

  bool init(const bool trans_a,
            const bool trans_b,
            const int max_batch_size,
            Context<TARGET(kCUDA)>* ctx);

  bool run(const PtypeOut alpha,
           const PtypeOut beta,
           const PtypeIn* a[],
           const PtypeIn* b[],
           PtypeOut* c[],
           const int m,
           const int n,
           const int k,
           const int batch_size);

  bool run(const PtypeOut alpha,
           const PtypeOut beta,
           const PtypeIn* a[],
           const int m,
           const int n,
           const int k,
           const int batch_size);

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
  PtypeIn** A_{nullptr};
};

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
