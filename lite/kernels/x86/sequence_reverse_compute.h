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

template <typename T, PrecisionType Ptype>
class SequenceReverseCompute : public KernelLite<TARGET(kX86), Ptype> {
 public:
  using param_t = operators::SequenceReverseParam;

  void Run() override {
    auto& param = this->template Param<param_t>();
    auto* output = param.Out;
    const auto* din = param.X->template data<T>();

    T* dout = output->template mutable_data<T>();
    CHECK_NE(din, dout)
        << "SequenceReverse Op does not support in-place operation";
    const auto lod = param.X->lod()[param.X->lod().size() - 1];
    const size_t lod_count = lod.size();

    size_t limit = static_cast<size_t>(param.X->numel());
    size_t row_numel = static_cast<size_t>(limit / param.X->dims()[0]);

    for (size_t idx = 0; idx < lod_count - 1; ++idx) {
      auto start_pos = lod[idx];
      auto end_pos = lod[idx + 1];
      for (auto pos = start_pos; pos < end_pos; ++pos) {
        auto cur_pos = end_pos - pos - 1 + start_pos;
        std::memcpy(dout + pos * row_numel,
                    din + cur_pos * row_numel,
                    row_numel * sizeof(T));
      }
    }
    output->set_lod(param.X->lod());
  }

  virtual ~SequenceReverseCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
