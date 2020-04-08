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

#include "lite/backends/x86/math/sequence_padding.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

namespace math = paddle::lite::x86::math;

template <typename T>
class SequenceUnpadCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::SequenceUnpadParam;

  void Run() override {
    auto& param = this->template Param<param_t>();
    auto& ctx = this->ctx_->template As<X86Context>();

    param.Out->template mutable_data<T>();
    int64_t padded_length = param.X->dims()[1];
    math::UnpaddingLoDTensorFunctor<lite::TargetType::kX86, T>()(
        ctx,
        *param.X,
        param.Out,
        padded_length,
        0,
        false,
        math::kBatchLengthWidth);
  }

  virtual ~SequenceUnpadCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
