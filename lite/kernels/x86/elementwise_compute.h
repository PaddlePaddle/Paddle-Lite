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
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class ElementwiseAddCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseAddCompute() = default;
};

template <typename T>
class ElementwiseAddActivationCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseAddActivationCompute() = default;
};

template <typename T>
class ElementwiseSubCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseSubCompute() = default;
};

template <typename T>
class ElementwiseSubActivationCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseSubActivationCompute() = default;
};

template <typename T>
class ElementwiseMulCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseMulCompute() = default;
};

template <typename T>
class ElementwiseMulActivationCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseMulActivationCompute() = default;
};

template <typename T>
class ElementwiseMaxCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseMaxCompute() = default;
};

template <typename T>
class ElementwiseMaxActivationCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseMaxActivationCompute() = default;
};

template <typename T>
class ElementwiseMinCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseMinCompute() = default;
};

template <typename T>
class ElementwiseMinActivationCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseMinActivationCompute() = default;
};

template <typename T>
class ElementwiseDivCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseDivCompute() = default;
};

template <typename T>
class ElementwiseDivActivationCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseDivActivationCompute() = default;
};

template <typename T>
class ElementwiseFloorDivCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseFloorDivCompute() = default;
};

template <typename T>
class ElementwiseFloorDivActivationCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseFloorDivActivationCompute() = default;
};

template <typename T>
class ElementwiseModCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseModCompute() = default;
};

template <typename T>
class ElementwiseModActivationCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseModActivationCompute() = default;
};

template <typename T>
class ElementwisePowCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwisePowCompute() = default;
};

template <typename T>
class ElementwisePowActivationCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwisePowActivationCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
