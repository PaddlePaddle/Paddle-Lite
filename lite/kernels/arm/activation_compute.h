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
#include <algorithm>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

class ReluCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~ReluCompute() = default;
};

class LeakyReluCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~LeakyReluCompute() = default;
};

class ReluClippedCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~ReluClippedCompute() = default;
};

class PReluCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~PReluCompute() = default;
};

class SigmoidCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~SigmoidCompute() = default;
};

class TanhCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~TanhCompute() = default;
};

class SwishCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~SwishCompute() = default;
};

class Relu6Compute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~Relu6Compute() = default;
};

class LogCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~LogCompute() = default;
};

class ExpCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~ExpCompute() = default;
};

class FloorCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~FloorCompute() = default;
};

class HardSigmoidCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~HardSigmoidCompute() = default;
};

class RsqrtCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~RsqrtCompute() = default;
};

class SquareCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~SquareCompute() = default;
};

class HardSwishCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~HardSwishCompute() = default;
};

class ReciprocalCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~ReciprocalCompute() = default;
};

class AbsCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~AbsCompute() = default;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
