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

template <typename T, PrecisionType PType>
class ElementwiseAddCompute : public KernelLite<TARGET(kARM), PType> {
 public:
  void Run() override;

  virtual ~ElementwiseAddCompute() = default;
};

template <typename T, PrecisionType PType>
class ElementwiseAddActivationCompute : public KernelLite<TARGET(kARM), PType> {
 public:
  void Run() override;

  virtual ~ElementwiseAddActivationCompute() = default;
};

template <typename T, PrecisionType PType>
class ElementwiseSubCompute : public KernelLite<TARGET(kARM), PType> {
 public:
  void Run() override;

  virtual ~ElementwiseSubCompute() = default;
};

template <typename T, PrecisionType PType>
class ElementwiseSubActivationCompute : public KernelLite<TARGET(kARM), PType> {
 public:
  void Run() override;

  virtual ~ElementwiseSubActivationCompute() = default;
};

template <typename T, PrecisionType PType>
class ElementwiseMulCompute : public KernelLite<TARGET(kARM), PType> {
 public:
  void Run() override;

  virtual ~ElementwiseMulCompute() = default;
};

template <typename T, PrecisionType PType>
class ElementwiseMulActivationCompute : public KernelLite<TARGET(kARM), PType> {
 public:
  void Run() override;

  virtual ~ElementwiseMulActivationCompute() = default;
};

class ElementwiseMaxCompute
    : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseMaxCompute() = default;
};

class ElementwiseMaxActivationCompute
    : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseMaxActivationCompute() = default;
};

class ElementwiseMinCompute
    : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseMinCompute() = default;
};

class ElementwiseMinActivationCompute
    : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseMinActivationCompute() = default;
};

template <typename T, PrecisionType PType>
class ElementwiseDivCompute : public KernelLite<TARGET(kARM), PType> {
 public:
  void Run() override;

  virtual ~ElementwiseDivCompute() = default;
};

template <typename T, PrecisionType PType>
class ElementwiseFloorDivCompute : public KernelLite<TARGET(kARM), PType> {
 public:
  void Run() override;

  virtual ~ElementwiseFloorDivCompute() = default;
};

template <typename T, PrecisionType PType>
class ElementwiseDivActivationCompute : public KernelLite<TARGET(kARM), PType> {
 public:
  void Run() override;

  virtual ~ElementwiseDivActivationCompute() = default;
};

template <typename T, PrecisionType PType>
class ElementwiseModCompute : public KernelLite<TARGET(kARM), PType> {
 public:
  void Run() override;

  virtual ~ElementwiseModCompute() = default;
};

template <typename T, PrecisionType PType>
class ElementwisePowCompute : public KernelLite<TARGET(kARM), PType> {
 public:
  void Run() override;

  virtual ~ElementwisePowCompute() = default;
};

// class ElementwiseModActivationCompute
//     : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
//  public:
//   void Run() override;

//   virtual ~ElementwiseModActivationCompute() = default;
// };

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
