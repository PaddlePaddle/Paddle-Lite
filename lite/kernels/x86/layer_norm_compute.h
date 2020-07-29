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

#include "lite/backends/x86/jit/helper.h"
#include "lite/backends/x86/jit/kernel_base.h"
#include "lite/backends/x86/jit/kernels.h"
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/operators/layer_norm_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class LayerNormCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::LayerNormParam;

  void Run() override {
    auto &param = *param_.get_mutable<param_t>();
    float epsilon = param.epsilon;
    auto Scale = param.Scale;
    auto Bias = param.Bias;
    auto x = param.X;

    auto y = param.Y;
    auto Mean = param.Mean;
    auto Var = param.Variance;
    auto begin_norm_axis = param.begin_norm_axis;

    auto x_dims = x->dims();

    y->template mutable_data<T>();
    Mean->template mutable_data<T>();
    Var->template mutable_data<T>();

    auto matrix_dim = x_dims.Flatten2D(begin_norm_axis);
    int left = static_cast<int>(matrix_dim[0]);
    int right = static_cast<int>(matrix_dim[1]);
    lite::DDim matrix_shape({left, right});

    lite::Tensor in;
    in.ShareDataWith(*x);
    in.Resize(matrix_shape);
    lite::Tensor out;
    out.ShareDataWith(*y);
    out.Resize(matrix_shape);

    CHECK_EQ(Mean->numel(), left);
    CHECK_EQ(Var->numel(), left);
    CHECK_EQ(Scale->numel(), right);
    CHECK_EQ(Bias->numel(), right);

    auto ker = paddle::lite::jit::KernelFuncs<jit::LayerNormTuple<T>,
                                              lite::fluid::CPUPlace>::Cache()
                   .At(right);
    ker(in.mutable_data<T>(),
        out.mutable_data<T>(),
        Mean->template mutable_data<T>(),
        Var->template mutable_data<T>(),
        Scale->template data<T>(),
        Bias->template data<T>(),
        static_cast<int>(left),
        epsilon,
        right);
  }

  virtual ~LayerNormCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
