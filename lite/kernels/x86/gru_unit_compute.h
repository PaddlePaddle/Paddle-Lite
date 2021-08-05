// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/backends/x86/fluid/eigen.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/x86/activation_compute.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

enum GRUActivationType { identity = 0, sigmoid = 1, tanh = 2, relu = 3 };

template <class T>
class GRUUnitCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::GRUUnitParam;

  void Run() override;

  virtual ~GRUUnitCompute() = default;

  template <typename Device, typename X, typename Y>
  void ActCompute(const int act_type, const Device& d, X x, Y y) const {
    switch (GRUActivationType(act_type)) {
      case identity:
        y.device(d) = x;
        break;
      case sigmoid:
        SigmoidFunctor<T>()(d, x, y);
        break;
      case tanh:
        TanhFunctor<T>()(d, x, y);
        break;
      case relu:
        ReluFunctor<T>()(d, x, y);
        break;
      default:
        LOG(FATAL) << "Unsupported activation type, only supports identity, "
                      "sigmoid, tanh and relu.";
    }
  }
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
