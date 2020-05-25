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
#include "lite/fluid/eigen.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class LookupTableCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::LookupTableParam;

  void Run() override {
    auto &param = *param_.get_mutable<operators::LookupTableParam>();
    auto *ids_t = param.Ids;
    auto *output_t = param.Out;
    int64_t padding_idx = param.padding_idx;
    const int64_t *ids = ids_t->template data<int64_t>();
    int64_t ids_numel = ids_t->dims().production();

    auto *table_t = param.W;
    int64_t row_number = table_t->dims()[0];
    int64_t row_width = table_t->dims()[1];

    const T *table = table_t->template data<T>();
    T *output = output_t->template mutable_data<T>();
    memset(output, 0, output_t->dims().production() * sizeof(T));
    for (int64_t i = 0; i < ids_numel; ++i) {
      if (padding_idx != -1 && ids[i] == padding_idx) {
        memset(output + i * row_width, 0, row_width * sizeof(T));
      } else {
        CHECK_LT(ids[i], row_number);
        CHECK_GE(ids[i], 0);
        memcpy(output + i * row_width,
               table + ids[i] * row_width,
               row_width * sizeof(T));
      }
    }
  }

  virtual ~LookupTableCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
