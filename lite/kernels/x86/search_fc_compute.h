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

#include "lite/backends/x86/math/search_fc.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class SearchFcCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::SearchFcParam;
  void Run() override {
    auto& context = ctx_->As<X86Context>();
    auto& param = *param_.get_mutable<param_t>();

    param.Out->Resize({param.X->dims()[0], param.out_size});
    lite::x86::math::SearchFcFunctor<lite::TargetType::kX86, T> search_fc;
    search_fc(context, *param.X, *param.W, *param.b, param.Out, param.out_size);
  }
  virtual ~SearchFcCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
