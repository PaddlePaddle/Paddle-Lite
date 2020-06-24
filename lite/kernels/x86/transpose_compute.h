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
#include <vector>
#include "lite/backends/x86/math/math_function.h"
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/operators/transpose_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <lite::TargetType Target, typename T>
inline void TransCompute(const int dim,
                         const lite::Context<Target>& context,
                         const lite::Tensor& in,
                         lite::Tensor* out,
                         const std::vector<int>& axis) {
  switch (dim) {
    case 1:
      paddle::lite::x86::math::Transpose<lite::TargetType::kX86, T, 1> trans1;
      trans1(context, in, out, axis);
      break;
    case 2:
      paddle::lite::x86::math::Transpose<lite::TargetType::kX86, T, 2> trans2;
      trans2(context, in, out, axis);
      break;
    case 3:
      paddle::lite::x86::math::Transpose<lite::TargetType::kX86, T, 3> trans3;
      trans3(context, in, out, axis);
      break;
    case 4:
      paddle::lite::x86::math::Transpose<lite::TargetType::kX86, T, 4> trans4;
      trans4(context, in, out, axis);
      break;
    case 5:
      paddle::lite::x86::math::Transpose<lite::TargetType::kX86, T, 5> trans5;
      trans5(context, in, out, axis);
      break;
    case 6:
      paddle::lite::x86::math::Transpose<lite::TargetType::kX86, T, 6> trans6;
      trans6(context, in, out, axis);
      break;
    default:
      LOG(FATAL) << "Tensors with rank at most 6 are supported";
  }
}

template <typename T>
class TransposeCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::TransposeParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto* x = param.x;
    auto* out = param.output;
    out->template mutable_data<T>();
    int ndims = param.axis.size();
    auto& context = ctx_->As<X86Context>();
    TransCompute<lite::TargetType::kX86, T>(
        ndims, context, *x, out, param.axis);
  }

  virtual ~TransposeCompute() = default;
};

template <typename T>
class Transpose2Compute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::TransposeParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto* x = param.x;
    auto* out = param.output;
    out->template mutable_data<T>();
    int ndims = param.axis.size();
    auto& context = ctx_->As<X86Context>();
    TransCompute<lite::TargetType::kX86, T>(
        ndims, context, *x, out, param.axis);
  }

  virtual ~Transpose2Compute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
