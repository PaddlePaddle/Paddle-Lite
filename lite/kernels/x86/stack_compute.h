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
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/operators/stack_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class StackCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::StackParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto x = param.X;
    auto y = param.Out;

    int axis = param.axis;
    if (axis < 0) axis += (x[0]->dims().size() + 1);

    int n = static_cast<int>(x.size());
    auto y_data = y->template mutable_data<T>();
    std::vector<const T*> x_datas(n);
    for (int i = 0; i < n; ++i) x_datas[i] = x[i]->template data<T>();

    int pre = 1, post = 1;
    auto dim = x[0]->dims();
    for (int i = 0; i < axis; ++i) pre *= dim[i];
    for (size_t i = axis; i < dim.size(); ++i) post *= dim[i];

    auto x_data_arr = x_datas.data();

    size_t x_offset = 0;
    size_t y_offset = 0;
    for (int i = 0; i < pre; i++) {
      for (int j = 0; j < n; j++) {
        std::memcpy(
            y_data + y_offset, x_data_arr[j] + x_offset, post * sizeof(T));
        y_offset += post;
      }
      x_offset += post;
    }
  }

  virtual ~StackCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
