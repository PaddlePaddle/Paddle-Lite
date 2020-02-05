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
#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class MatchMatrixTensorCompute
    : public KernelLite<TARGET(kCUDA), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::MatchMatrixTensorParam;

  void PrepareForRun() override;
  void Run() override;
  virtual ~MatchMatrixTensorCompute() = default;

 private:
  std::unique_ptr<lite::cuda::math::Gemm<float, float>> gemm_impl_;
  lite::Tensor _input_l_transform;
  lite::Tensor _input_l_transform_reorganize;
  lite::Tensor _output_tmp;
  lite::Tensor _offset_r;
};

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
