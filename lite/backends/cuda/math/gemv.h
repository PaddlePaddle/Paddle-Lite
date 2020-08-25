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

#pragma once
#include <cudnn.h>
#include <string>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/core/context.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename PtypeIn, typename PtypeOut>
class Gemv {
 public:
  Gemv() : cu_handle_(nullptr) {}
  ~Gemv() {}

  bool init(const bool trans_,
            const int m,
            const int n,
            const int lda,
            const int ldb,
            const int ldc,
            Context<TARGET(kCUDA)>* ctx);

  bool run(const PtypeOut alpha,
           const PtypeOut beta,
           const PtypeIn* a,
           const PtypeIn* b,
           PtypeOut* c);

  cublasHandle_t get_handle() const { return cu_handle_; }

 private:
  cudaStream_t exe_stream_;
  cublasHandle_t cu_handle_;
  cublasOperation_t cu_trans_;
  int m_{-1};
  int n_{-1};
  int lda_{-1};
  int ldb_{-1};
  int ldc_{-1};
};

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
