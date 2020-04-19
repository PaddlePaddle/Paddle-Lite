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

#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class ShapeCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ShapeParam;

  void Run() override {
    auto& param = *param_.get_mutable<operators::ShapeParam>();
    // auto& context = context_->As<X86Context>();
    auto out_data = param.Out->template mutable_data<int32_t>();
    auto in_dims = param.X->dims();
    for (size_t i = 0; i < in_dims.size(); ++i) {
      out_data[i] = in_dims[i];
    }
  }

  virtual ~ShapeCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
