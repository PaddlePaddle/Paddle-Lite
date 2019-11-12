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
#include "lite/backends/x86/math/sequence_softmax.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class SequenceSoftmaxCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override {
    auto& context = ctx_->As<X86Context>();
    auto& param = *param_.get_mutable<operators::SequenceSoftmaxParam>();

    auto x_data = param.X->data<T>();
    auto o_data = param.Out->mutable_data<T>();
    auto input_dims = param.X->dims();
    int in_h = input_dims[0];
    int in_w = param.X->numel() / in_h;
    CHECK_EQ(in_w, 1) << "input dims is not valid";
    auto seq_offset = param.X->lod()[0];
    CHECK_EQ(in_h, seq_offset.back()) << "input dims is not valid";

    lite::x86::math::SequenceSoftmaxFunctor<lite::TargetType::kX86, T> softmax;
    softmax(x_data, seq_offset, o_data, context);
  }

  virtual ~SequenceSoftmaxCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
