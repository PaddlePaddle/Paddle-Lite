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
#include <vector>

#include "lite/backends/arm/math/elementwise.h"
#include "lite/operators/sum_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

// Sum
template <typename T>
class SumCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  void Run() {
    operators::SumParam& param = this->template Param<operators::SumParam>();
    auto& out = param.Out;
    std::vector<lite::Tensor*>& inputs = param.X;
    auto num = inputs.front()->dims().production();
    auto* out_data = param.Out->mutable_data<T>();
    if (inputs.size() == 1) {
      if (!param.inplace) {
        param.Out->CopyDataFrom(*inputs[0]);
      }
      return;
    }
    int start_index = 0;

    if (param.inplace) {  // inplace add
      start_index = 1;
    } else {
      lite::arm::math::elementwise_add<T>(
          inputs[0]->data<T>(), inputs[1]->data<T>(), out_data, num);
      start_index = 2;
    }
    for (auto it = inputs.begin() + start_index; it != inputs.end(); ++it) {
      const auto& x_data = (*it)->data<T>();
      lite::arm::math::elementwise_add<T>(x_data, out_data, out_data, num);
    }
  }

  virtual ~SumCompute() = default;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
