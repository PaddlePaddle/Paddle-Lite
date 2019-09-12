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
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/operators/squeeze_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class SqueezeCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::SqueezeParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto x = param.X;
    auto output = param.Out;
    auto x_dims = x->dims();
    auto* x_data = x->data<T>();
    auto* out_data = output->mutable_data<T>();
    memcpy(out_data, x_data, x_dims.production() * sizeof(T));
  }

  virtual ~SqueezeCompute() = default;
};

template <typename T>
class Squeeze2Compute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::SqueezeParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto x = param.X;
    auto output = param.Out;
    auto xshape = param.XShape;
    auto x_dims = x->dims();
    auto* x_data = x->data<T>();
    auto* out_data = output->mutable_data<T>();
    auto* xshape_data = xshape->mutable_data<T>();
    memcpy(out_data, x_data, x_dims.production() * sizeof(T));
    memcpy(xshape_data, x_data, x_dims.production() * sizeof(T));
  }

  virtual ~Squeeze2Compute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
