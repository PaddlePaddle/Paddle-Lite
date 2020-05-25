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

class ElementwiseAddCompute
    : public KernelLite<TARGET(kCUDA), PRECISION(kFloat)> {
 public:
  using param_t = operators::ElementwiseParam;

  void Run() override;
  virtual ~ElementwiseAddCompute() = default;
};

class ElementwiseAddComputeNHWC
    : public KernelLite<TARGET(kCUDA), PRECISION(kFloat), DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::ElementwiseParam;

  void Run() override;
  virtual ~ElementwiseAddComputeNHWC() = default;
};

class ElementwiseSubCompute
    : public KernelLite<TARGET(kCUDA), PRECISION(kFloat)> {
 public:
  using param_t = operators::ElementwiseParam;

  void Run() override;
  virtual ~ElementwiseSubCompute() = default;
};

class ElementwiseSubComputeNHWC
    : public KernelLite<TARGET(kCUDA), PRECISION(kFloat), DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::ElementwiseParam;

  void Run() override;
  virtual ~ElementwiseSubComputeNHWC() = default;
};

class ElementwiseMulCompute
    : public KernelLite<TARGET(kCUDA), PRECISION(kFloat)> {
 public:
  using param_t = operators::ElementwiseParam;

  void Run() override;
  virtual ~ElementwiseMulCompute() = default;
};

class ElementwiseMulComputeNHWC
    : public KernelLite<TARGET(kCUDA), PRECISION(kFloat), DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::ElementwiseParam;

  void Run() override;
  virtual ~ElementwiseMulComputeNHWC() = default;
};

class ElementwiseAddActivationCompute
    : public KernelLite<TARGET(kCUDA), PRECISION(kFloat)> {
 public:
  using param_t = operators::FusionElementwiseActivationParam;

  void Run() override;
  virtual ~ElementwiseAddActivationCompute() = default;
};

class ElementwiseAddActivationComputeNHWC
    : public KernelLite<TARGET(kCUDA), PRECISION(kFloat), DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::FusionElementwiseActivationParam;

  void Run() override;
  virtual ~ElementwiseAddActivationComputeNHWC() = default;
};

class ElementwiseSubActivationCompute
    : public KernelLite<TARGET(kCUDA), PRECISION(kFloat)> {
 public:
  using param_t = operators::FusionElementwiseActivationParam;

  void Run() override;
  virtual ~ElementwiseSubActivationCompute() = default;
};

class ElementwiseSubActivationComputeNHWC
    : public KernelLite<TARGET(kCUDA), PRECISION(kFloat), DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::FusionElementwiseActivationParam;

  void Run() override;
  virtual ~ElementwiseSubActivationComputeNHWC() = default;
};

class ElementwiseMulActivationCompute
    : public KernelLite<TARGET(kCUDA), PRECISION(kFloat)> {
 public:
  using param_t = operators::FusionElementwiseActivationParam;

  void Run() override;
  virtual ~ElementwiseMulActivationCompute() = default;
};

class ElementwiseMulActivationComputeNHWC
    : public KernelLite<TARGET(kCUDA), PRECISION(kFloat), DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::FusionElementwiseActivationParam;

  void Run() override;
  virtual ~ElementwiseMulActivationComputeNHWC() = default;
};

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
