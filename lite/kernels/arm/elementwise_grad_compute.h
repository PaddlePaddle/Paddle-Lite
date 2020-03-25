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
#include <algorithm>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

class ElementwiseAddGradCompute
    : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseAddGradCompute() = default;
};

class ElementwiseSubGradCompute
    : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseSubGradCompute() = default;
};

template <typename T, PrecisionType PType>
class ElementwiseMulGradCompute : public KernelLite<TARGET(kARM), PType> {
 public:
  void Run() override;

  virtual ~ElementwiseMulGradCompute() = default;
};

class ElementwiseMaxGradCompute
    : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseMaxGradCompute() = default;
};

class ElementwiseDivGradCompute
    : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~ElementwiseDivGradCompute() = default;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
