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
#include "lite/backends/cuda/math/transpose.h"
#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class NCHWToNHWCCompute : public KernelLite<TARGET(kCUDA), PRECISION(kFloat)> {
 public:
  using param_t = operators::LayoutParam;
  void Run() override;
  virtual ~NCHWToNHWCCompute() = default;

 private:
  lite::cuda::math::Transpose<float> trans;
};

class NCHWToNHWCComputeInt8
    : public KernelLite<TARGET(kCUDA), PRECISION(kInt8)> {
 public:
  using param_t = operators::LayoutParam;
  void Run() override;
  virtual ~NCHWToNHWCComputeInt8() = default;

 private:
  lite::cuda::math::Transpose<int8_t> trans;
};

class NHWCToNCHWCompute : public KernelLite<TARGET(kCUDA), PRECISION(kFloat)> {
 public:
  using param_t = operators::LayoutParam;
  void Run() override;
  virtual ~NHWCToNCHWCompute() = default;

 private:
  lite::cuda::math::Transpose<float> trans;
};

class NHWCToNCHWComputeInt8
    : public KernelLite<TARGET(kCUDA), PRECISION(kInt8)> {
 public:
  using param_t = operators::LayoutParam;
  void Run() override;
  virtual ~NHWCToNCHWComputeInt8() = default;

 private:
  lite::cuda::math::Transpose<int8_t> trans;
};

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
