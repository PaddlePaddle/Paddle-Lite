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
#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename Dtype>
class ConcatCompute
    : public KernelLite<TARGET(kCUDA), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::ConcatParam;

  void Run() override;
  virtual ~ConcatCompute() = default;
};

template <typename Dtype>
class ConcatComputeNHWC
    : public KernelLite<TARGET(kCUDA), PRECISION(kFloat), DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::ConcatParam;

  void Run() override {}
  virtual ~ConcatComputeNHWC() = default;
};

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
