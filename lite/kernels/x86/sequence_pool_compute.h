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
#include "lite/backends/x86/math/math_function.h"
#include "lite/backends/x86/math/sequence_pooling.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class SequencePoolCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::SequencePoolParam;

  void Run() override {
    auto& param = *param_.get_mutable<operators::SequencePoolParam>();
    auto& context = ctx_->As<X86Context>();
    auto* out = param.Out;
    auto dims = param.X->dims();
    auto lod = param.X->lod();
    CHECK_EQ(lod.size(), 1UL);
    CHECK_GE(dims[0], static_cast<int64_t>(lod[0].size() - 1));

    dims[0] = lod[0].size() - 1;
    out->Resize({dims});
    out->template mutable_data<T>();
    lite::Tensor* index = nullptr;

    const bool is_test = true;
    float pad_value = 0.0;

    lite::x86::math::SequencePoolFunctor<lite::TargetType::kX86, T> pool;
    pool(context, param.pool_type, pad_value, *param.X, out, is_test, index);
  }

  virtual ~SequencePoolCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
