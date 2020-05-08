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

#include <Eigen/Core>
#include "lite/backends/x86/math/blas.h"
#include "lite/backends/x86/math/math_function.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class FillConstantBatchSizeLikeCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::FillConstantBatchSizeLikeParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& ctx = ctx_->As<X86Context>();
    auto* out = param.out;
    auto* in = param.input;
    if (in->lod().size() && param.input_dim_idx == 0) {
      // set the correct batch size for the LoDTensor.
      auto odims = out->dims();
      int output_dim_idx = param.output_dim_idx;
      odims[output_dim_idx] = static_cast<int>(in->lod().back().size()) - 1;
      out->Resize(odims);
      // out->template mutable_data<T>();
    }
    out->template mutable_data<T>();
    auto value = param.value;

    paddle::lite::x86::math::SetConstant<lite::TargetType::kX86, T> setter;
    setter(ctx, out, static_cast<T>(value));
  }

  virtual ~FillConstantBatchSizeLikeCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
