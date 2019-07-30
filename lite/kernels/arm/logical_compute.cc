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

#include "lite/kernels/arm/logical_compute.h"
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void LogicalXorCompute::PrepareForRun() {}

void LogicalXorCompute::Run() {
  auto& param = this->Param<operators::LogicalParam>();

  ///  using LogicalFunctor = Functor<bool>;

  const size_t count = param.X->numel();
  bool* z = param.Out->mutable_data<bool>();
  const bool* x = param.X->data<bool>();
  const bool* y = param.Y->data<bool>();

  for (int i = 0; i < count; ++i) {
    // z[i] = LogicalFunctor()(x[i], y[i]);
    z[i] = (x[i] || y[i]) && !(x[i] && y[i]);
  }
}

void LogicalAndCompute::PrepareForRun() {}

void LogicalAndCompute::Run() {
  auto& param = this->Param<operators::LogicalParam>();

  ///  using LogicalFunctor = Functor<bool>;

  const size_t count = param.X->numel();
  bool* z = param.Out->mutable_data<bool>();
  const bool* x = param.X->data<bool>();
  const bool* y = param.Y->data<bool>();

  for (int i = 0; i < count; ++i) {
    // z[i] = LogicalFunctor()(x[i], y[i]);
    z[i] = (x[i] && y[i]);
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
REGISTER_LITE_KERNEL(logical_xor,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::LogicalXorCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(logical_and,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::LogicalAndCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
