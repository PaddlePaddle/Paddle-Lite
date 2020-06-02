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
#include "lite/backends/cuda/math/cudnn_conv.h"
#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T, PrecisionType PType>
class VarConv2DCompute : public KernelLite<TARGET(kCUDA), PType> {
 public:
  using param_t = operators::VarConv2DParam;

  void Run() override;
  void PrepareForRun() override;
  virtual ~VarConv2DCompute() = default;

 private:
  mutable operators::ConvParam conv_param_;
  std::unique_ptr<lite::cuda::math::CudnnConv2D<T, PType>> conv_impl_;
  lite::Tensor offset_;
};

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
