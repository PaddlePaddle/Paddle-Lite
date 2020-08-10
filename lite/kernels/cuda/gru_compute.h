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
#include <memory>

#include "lite/backends/cuda/math/gemm.h"
#include "lite/backends/cuda/math/sequence_helper.h"
#include "lite/core/kernel.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T, PrecisionType PType>
class GRUCompute : public KernelLite<TARGET(kCUDA), PType> {
 public:
  using param_t = operators::GRUParam;

  void PrepareForRun() override;

  void Run() override;

  virtual ~GRUCompute() = default;

 private:
  std::unique_ptr<lite::cuda::math::Gemm<T, T>> gemm_impl_{nullptr};
  lite::Tensor ordered_h0_;
  lite::cuda::math::SeqSortedseqTranseUtil seq_utils_;
};

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
