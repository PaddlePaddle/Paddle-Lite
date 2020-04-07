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

#include "lite/backends/x86/math/blas.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class SearchSeqFcCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::SearchSeqFcParam;

  void Run() override {
    auto& context = ctx_->As<X86Context>();
    auto& param = *param_.get_mutable<operators::SearchSeqFcParam>();

    auto x = param.x;
    auto w = param.w;
    auto b = param.b;
    auto out = param.out;
    auto out_size = param.out_size;
    const auto x_dims = x->dims();
    const auto w_dims = w->dims();
    const auto out_dims = out->dims();
    CHECK_EQ(x_dims.size(), 2) << "The Input(X) should be 2-D tensor.";
    CHECK_EQ(w_dims.size(), 2) << "W should be 2-D tensor.";
    CHECK_EQ(out_dims.size(), 2) << "The Output(Out) should be 2-D tensor.";
    CHECK_EQ(x_dims[1], w_dims[1]) << "Wrong shape: x_dims[1] != w_dims[1]";
    CHECK_EQ(w_dims[0], out_size) << "Wrong shape: w_dims[0] != out_size";
    CHECK_EQ(out_dims[0], x_dims[0]) << "Wrong shape: out_dims[0] != x_dims[0]";
    CHECK_EQ(out_dims[1], out_size) << "Wrong shape: out_dims[1] != out_size";

    auto blas = lite::x86::math::GetBlas<lite::TargetType::kX86, T>(context);
    blas.MatMul(*x, false, *w, true, out);

    if (b != nullptr) {
      auto b_dims = b->dims();
      CHECK_EQ(b_dims.size(), 1) << "b should be 1-D tensor.";
      CHECK_EQ(b_dims[0], w_dims[0]) << "Wrong shape: b_dims[0] != w_dims[0]";
      int M = x_dims[0];
      int N = w_dims[0];
      for (int i = 0; i < M; i++) {
        blas.AXPY(N,
                  static_cast<T>(1),
                  b->template data<T>(),
                  out->template mutable_data<T>() + i * N);
      }
    }
  }

  virtual ~SearchSeqFcCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
