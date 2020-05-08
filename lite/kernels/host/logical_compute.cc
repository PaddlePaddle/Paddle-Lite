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

#include "lite/kernels/host/logical_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

#define LOGICAL_FUNCTOR(name, op)                                \
  struct _##name##Functor {                                      \
    inline bool operator()(const bool& a, const bool& b) const { \
      return a op b;                                             \
    }                                                            \
  };

LOGICAL_FUNCTOR(LogicalAnd, &&);
LOGICAL_FUNCTOR(LogicalOr, ||);

struct _LogicalXorFunctor {
  inline bool operator()(const bool& a, const bool& b) const {
    return (a || b) && !(a && b);
  }
};

struct _LogicalNotFunctor {
  inline bool operator()(const bool& a) const { return !a; }
};

template <class Functor>
// template<typename Functor>
void BinaryLogicalCompute<Functor>::Run() {
  auto& param = this->Param<operators::LogicalParam>();
  const size_t count = param.X->numel();
  bool* z = param.Out->template mutable_data<bool>();
  const bool* x = param.X->template data<bool>();
  const bool* y = param.Y->template data<bool>();
  for (int i = 0; i < count; ++i) {
    z[i] = Functor()(x[i], y[i]);
  }
}

template <class Functor>
void UnaryLogicalCompute<Functor>::Run() {
  auto& param = this->Param<operators::LogicalParam>();
  const size_t count = param.X->numel();
  bool* z = param.Out->template mutable_data<bool>();
  const auto x = param.X->template data<bool>();
  for (int i = 0; i < count; ++i) {
    z[i] = Functor()(x[i]);
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(logical_xor,
                     kHost,
                     kAny,
                     kAny,
                     paddle::lite::kernels::host::BinaryLogicalCompute<
                         paddle::lite::kernels::host::_LogicalXorFunctor>,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(logical_and,
                     kHost,
                     kAny,
                     kAny,
                     paddle::lite::kernels::host::BinaryLogicalCompute<
                         paddle::lite::kernels::host::_LogicalAndFunctor>,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(logical_or,
                     kHost,
                     kAny,
                     kAny,
                     paddle::lite::kernels::host::BinaryLogicalCompute<
                         paddle::lite::kernels::host::_LogicalOrFunctor>,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(logical_not,
                     kHost,
                     kAny,
                     kAny,
                     paddle::lite::kernels::host::UnaryLogicalCompute<
                         paddle::lite::kernels::host::_LogicalNotFunctor>,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .Finalize();
