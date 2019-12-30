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
#include <memory>
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/backends/cuda/math/gemm.h"
#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

const int CUDA_NUM_THREADS = 512;
inline int CUDA_GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
inline int CUDA_GET_BLOCKS(const int N, const int base) {
  return (N + base - 1) / base;
}

template <typename T>
class SearchFcCompute : public KernelLite<TARGET(kCUDA), PRECISION(kFloat)> {
 public:
  using param_t = operators::SearchFcParam;
  void PrepareForRun() override;
  void Run() override;
  virtual ~SearchFcCompute() = default;

 private:
  std::unique_ptr<lite::cuda::math::Gemm<float, float>> gemm_impl_{nullptr};
  int _M;
  int _K;
  int _N;
};

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
