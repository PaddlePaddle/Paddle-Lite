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

#include <algorithm>
#include <vector>
#include "lite/backends/x86/math/blas.h"
#include "lite/backends/x86/math/context_project.h"
#include "lite/backends/x86/math/math_function.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

namespace math = paddle::lite::x86::math;

template <typename T>
class SequenceConvCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::SequenceConvParam;

  void Run() override {
    auto& param = this->template Param<param_t>();
    auto& ctx = this->ctx_->template As<X86Context>();

    auto* in = param.X;
    auto* filter = param.Filter;
    auto* out = param.Out;
    out->template mutable_data<T>();
    CHECK(in->lod().size() == 1) << "Only support one level sequence now";

    int context_start = param.contextStart;
    int context_stride = param.contextStride;
    int context_length = param.contextLength;
    bool padding_trainable = false;
    const Tensor* padding_data = nullptr;

    int up_pad = (std::max)(0, -context_start);
    int down_pad = (std::max)(0, context_start + context_length - 1);
    auto sequence_width = static_cast<int64_t>(in->dims()[1]);

    std::vector<int64_t> col_shape{in->dims()[0],
                                   context_length * sequence_width};
    Tensor col;
    col.Resize(col_shape);
    col.mutable_data<T>();

    // Because if padding_trainable is false, padding data should be zeros.
    math::SetConstant<TARGET(kX86), T> set_zero;
    auto blas = math::GetBlas<TARGET(kX86), T>(ctx);
    set_zero(ctx, &col, static_cast<T>(0));
    math::ContextProjectFunctor<TARGET(kX86), T> seq_project_functor;

    seq_project_functor(ctx,
                        *in,
                        padding_data,
                        padding_trainable,
                        context_start,
                        context_length,
                        context_stride,
                        up_pad,
                        down_pad,
                        &col);

    blas.MatMul(col, *filter, out);
  }

  virtual ~SequenceConvCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
