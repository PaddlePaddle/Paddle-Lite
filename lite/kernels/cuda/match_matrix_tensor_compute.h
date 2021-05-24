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
#include <memory>
#include "lite/backends/cuda/blas.h"
#include "lite/backends/cuda/math/gemm.h"
#include "lite/backends/cuda/math/transpose.h"
#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T, PrecisionType PType>
class MatchMatrixTensorCompute
    : public KernelLite<TARGET(kCUDA), PType, DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::MatchMatrixTensorParam;

  void PrepareForRun() override;
  void Run() override;
  virtual ~MatchMatrixTensorCompute() = default;

 private:
  std::unique_ptr<lite::cuda::math::Gemm<T, T>> gemm_impl_;
  lite::cuda::math::Transpose<T> trans_;

  lite::Tensor input_l_transform_;
  lite::Tensor input_l_transform_reorganize_;
  lite::Tensor offset_r_;
  lite::Tensor offset_l_;
};

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
