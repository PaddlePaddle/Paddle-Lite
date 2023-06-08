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

template <typename T, PrecisionType PType>
class ReluCompute : public KernelLite<TARGET(kXPU), PType> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~ReluCompute() = default;
};

template <typename T, PrecisionType PType>
class Relu6Compute : public KernelLite<TARGET(kXPU), PType> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~Relu6Compute() = default;
};

template <typename T, PrecisionType PType>
class GeluCompute : public KernelLite<TARGET(kXPU), PType> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~GeluCompute() = default;
};

template <typename T, PrecisionType PType>
class TanhCompute : public KernelLite<TARGET(kXPU), PType> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~TanhCompute() = default;
};

template <typename T, PrecisionType PType>
class SigmoidCompute : public KernelLite<TARGET(kXPU), PType> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~SigmoidCompute() = default;
};

template <typename T, PrecisionType PType>
class SiluCompute : public KernelLite<TARGET(kXPU), PType> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~SiluCompute() = default;
};

template <typename T, PrecisionType PType>
class EluCompute : public KernelLite<TARGET(kXPU), PType> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~EluCompute() = default;
};

template <typename T, PrecisionType PType>
class SoftplusCompute : public KernelLite<TARGET(kXPU), PType> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~SoftplusCompute() = default;
};

template <typename T, PrecisionType PType>
class AbsCompute : public KernelLite<TARGET(kXPU), PType> {
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

class RsqrtCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~RsqrtCompute() = default;
};

class PowCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::PowParam;

  virtual void Run();

  virtual ~PowCompute() = default;
};

class LogCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~LogCompute() = default;
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

template <typename T, PrecisionType PType>
class LeakyReluCompute : public KernelLite<TARGET(kXPU), PType> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~LeakyReluCompute() = default;
};

class SoftsignCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~SoftsignCompute() = default;
};

class SwishCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~SwishCompute() = default;
};

class PReluCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  virtual void Run();

  virtual ~PReluCompute() = default;
};

class FloorCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~FloorCompute() = default;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
