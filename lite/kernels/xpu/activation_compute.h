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
namespace xpu {

class ReluCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~ReluCompute() = default;
};

class Relu6Compute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~Relu6Compute() = default;
};

class TanhCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~TanhCompute() = default;
};

class SigmoidCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~SigmoidCompute() = default;
};

class AbsCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~AbsCompute() = default;
};

class ExpCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~ExpCompute() = default;
};

class SquareCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~SquareCompute() = default;
};

class ReciprocalCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~ReciprocalCompute() = default;
};

class SqrtCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~SqrtCompute() = default;
};

class PowCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::PowParam;

  virtual void Run();

  virtual ~PowCompute() = default;
};

class SignCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::SignParam;

  virtual void Run();

  virtual ~SignCompute() = default;
};

class HardSwishCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~HardSwishCompute() = default;
};

class HardSigmoidCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~HardSigmoidCompute() = default;
};

class LeakyReluCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~LeakyReluCompute() = default;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
