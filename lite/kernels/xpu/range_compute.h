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
namespace xpu {

template <typename T, PrecisionType PType>
class RangeCompute : public KernelLite<TARGET(kXPU), PType, DATALAYOUT(kAny)> {
 public:
  void Run() override;
  void InferShapeImpl(T start, T step, T end, int64_t* len) {
    CHECK(!std::equal_to<T>()(step, 0))
        << "The step of range op should not be 0.";
    CHECK(((start < end) && (step > 0)) || ((start > end) && (step < 0)))
        << "The step should be greater than 0 while start < end. And the "
           "step should be less than 0 while start > end.";
    *len = std::is_integral<T>::value
               ? ((std::abs(end - start) + std::abs(step) - 1) / std::abs(step))
               : std::ceil(std::abs((end - start) / step));
  }

  virtual ~RangeCompute() = default;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
