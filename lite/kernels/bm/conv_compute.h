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
namespace bm {

class ConvCompute : public KernelLite<TARGET(kBM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ConvParam;

  void PrepareForRun() {};
  void Run() {};
  virtual ~ConvCompute() = default;
};

template <PrecisionType Ptype_out>
class ConvComputeInt8
    : public KernelLite<TARGET(kBM), PRECISION(kInt8), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::ConvParam;

  void PrepareForRun() {};
  void Run() {};
  virtual ~ConvComputeInt8() = default;
};

}  // namespace bm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
