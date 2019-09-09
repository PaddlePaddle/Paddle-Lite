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
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

#define LOGICAL_FUNCTOR(name, op)                                           \
  template <typename T>                                                     \
  struct _##name##Functor {                                                 \
    inline bool operator()(const T& a, const T& b) const { return a op b; } \
  };

LOGICAL_FUNCTOR(LogicalAnd, &&);
LOGICAL_FUNCTOR(LogicalOr, ||);

template <typename T>
struct _LogicalXorFunctor {
  inline bool operator()(const T& a, const T& b) const {
    return (a || b) && !(a && b);
  }
};

template <typename T>
struct _LogicalNotFunctor {
  inline bool operator()(const T& a) const { return !a; }
};

// template<typename Functor>
template <template <typename T> class Functor>
void BinaryLogicalCompute<Functor>::PrepareForRun() {}

template <template <typename T> class Functor>
// template<typename Functor>
void BinaryLogicalCompute<Functor>::Run() {
  auto& param = this->Param<operators::LogicalParam>();
  const size_t count = param.X->numel();
  bool* z = param.Out->template mutable_data<bool>();
  const bool* x = param.X->template data<bool>();
  const bool* y = param.Y->template data<bool>();
  using LogicalFunctor = Functor<bool>;
  for (int i = 0; i < count; ++i) {
    z[i] = LogicalFunctor()(x[i], y[i]);
  }
}

template <template <typename> class Functor>
void UnaryLogicalCompute<Functor>::PrepareForRun() {}

template <template <typename> class Functor>
void UnaryLogicalCompute<Functor>::Run() {
  auto& param = this->Param<operators::LogicalParam>();
  const size_t count = param.X->numel();
  bool* z = param.Out->template mutable_data<bool>();
  const auto x = param.X->template data<bool>();
  using LogicalFunctor = Functor<bool>;
  for (int i = 0; i < count; ++i) {
    z[i] = LogicalFunctor()(x[i]);
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
REGISTER_LITE_KERNEL(
    logical_xor,
    kARM,
    kFloat,
    kNCHW,
    paddle::lite::kernels::arm::BinaryLogicalCompute<
        paddle::lite::kernels::arm::_LogicalXorFunctor>,
    //  paddle::lite::kernels::arm::BinaryLogicalCompute<paddle::lite::kernels::arm::_LogicalXorFunctor<bool>>,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .Finalize();
REGISTER_LITE_KERNEL(
    logical_and,
    kARM,
    kFloat,
    kNCHW,
    // paddle::lite::kernels::arm::BinaryLogicalCompute<paddle::lite::kernels::arm::_LogicalAndFunctor<bool>>,
    paddle::lite::kernels::arm::BinaryLogicalCompute<
        paddle::lite::kernels::arm::_LogicalAndFunctor>,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .Finalize();
REGISTER_LITE_KERNEL(logical_or,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::BinaryLogicalCompute<
                         paddle::lite::kernels::arm::_LogicalOrFunctor>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .Finalize();
REGISTER_LITE_KERNEL(logical_not,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::UnaryLogicalCompute<
                         paddle::lite::kernels::arm::_LogicalNotFunctor>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .Finalize();
