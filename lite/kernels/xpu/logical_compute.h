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

typedef enum { LOGICAL_NOT = 0, LOGICAL_AND = 1 } LogicalType;

template <int LogicType>
class BinaryLogicalCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void Run() override;

  // void LogicalData(const T* x, const T* y, bool* z, int len);

  virtual ~BinaryLogicalCompute() = default;
};

template <int LogicType>
class UnaryLogicalCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void Run() override;

  // void LogicalData(const T* x, const T* y, bool* z, int len);

  virtual ~UnaryLogicalCompute() = default;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
