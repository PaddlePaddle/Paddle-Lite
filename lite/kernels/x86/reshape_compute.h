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
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/operators/reshape_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
void Compute(const lite::Tensor* in, lite::Tensor* out) {
  // In CopyDataFrom, the target tensor's dims will be set to the source
  // tensor's dims.
  auto out_dims = out->dims();
  out->CopyDataFrom(*in);
  out->Resize(out_dims);
}

template <typename T>
class ReshapeCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ReshapeParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    Compute<T>(param.x, param.output);
  }

  virtual ~ReshapeCompute() = default;
};

template <typename T>
void reshape2_compute() {}

template <typename T>
class Reshape2Compute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ReshapeParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    Compute<T>(param.x, param.output);
  }

  virtual ~Reshape2Compute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
