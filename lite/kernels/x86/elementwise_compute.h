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
#include <cmath>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/fluid/eigen.h"
#include "lite/kernels/x86/elementwise_op_function.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
struct SubFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a - b; }
};

template <typename T>
struct AddFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct MulFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a * b; }
};

template <typename T>
struct DivFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a / b; }
};

template <typename T>
struct FloorDivFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const {
    return static_cast<T>(std::trunc(a / b));
  }
};

template <typename T>
struct PowFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return std::pow(a, b); }
};

template <typename T>
struct ModFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const {
    T res = a % b;
    if ((res != 0) && ((res < 0) != (b < 0))) res += b;
    return res;
  }
};

template <typename T>
struct MaxFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a > b ? a : b; }
};

template <typename T>
struct MinFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a < b ? a : b; }
};

template <typename T>
class ElementwiseAddCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ElementwiseParam;
  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<X86Context>();
    param.Out->template mutable_data<T>();
    ElementwiseComputeEx<AddFunctor<T>, lite::TargetType::kX86, T>(
        context, param.X, param.Y, param.axis, AddFunctor<T>(), param.Out);
  }

  virtual ~ElementwiseAddCompute() = default;
};

template <typename T>
class ElementwiseSubCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ElementwiseParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<X86Context>();

    param.Out->template mutable_data<T>();
    ElementwiseComputeEx<SubFunctor<T>, lite::TargetType::kX86, T>(
        context, param.X, param.Y, param.axis, SubFunctor<T>(), param.Out);
  }

  virtual ~ElementwiseSubCompute() = default;
};

template <typename T>
class ElementwiseMulCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ElementwiseParam;
  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<X86Context>();
    param.Out->template mutable_data<T>();
    ElementwiseComputeEx<MulFunctor<T>, lite::TargetType::kX86, T>(
        context, param.X, param.Y, param.axis, MulFunctor<T>(), param.Out);
  }

  virtual ~ElementwiseMulCompute() = default;
};

template <typename T>
class ElementwiseDivCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ElementwiseParam;
  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<X86Context>();
    param.Out->template mutable_data<T>();
    ElementwiseComputeEx<DivFunctor<T>, lite::TargetType::kX86, T>(
        context, param.X, param.Y, param.axis, DivFunctor<T>(), param.Out);
  }

  virtual ~ElementwiseDivCompute() = default;
};

template <typename T>
class ElementwiseFloorDivCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ElementwiseParam;
  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<X86Context>();
    param.Out->template mutable_data<T>();
    ElementwiseComputeEx<FloorDivFunctor<T>, lite::TargetType::kX86, T>(
        context, param.X, param.Y, param.axis, FloorDivFunctor<T>(), param.Out);
  }

  virtual ~ElementwiseFloorDivCompute() = default;
};

template <typename T>
class ElementwisePowCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ElementwiseParam;
  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<X86Context>();
    param.Out->template mutable_data<T>();
    ElementwiseComputeEx<PowFunctor<T>, lite::TargetType::kX86, T>(
        context, param.X, param.Y, param.axis, PowFunctor<T>(), param.Out);
  }

  virtual ~ElementwisePowCompute() = default;
};

template <typename T>
class ElementwiseModCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ElementwiseParam;
  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<X86Context>();
    param.Out->template mutable_data<T>();
    ElementwiseComputeEx<ModFunctor<T>, lite::TargetType::kX86, T>(
        context, param.X, param.Y, param.axis, ModFunctor<T>(), param.Out);
  }

  virtual ~ElementwiseModCompute() = default;
};

template <typename T>
class ElementwiseMaxCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ElementwiseParam;
  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<X86Context>();
    param.Out->template mutable_data<T>();
    ElementwiseComputeEx<MaxFunctor<T>, lite::TargetType::kX86, T>(
        context, param.X, param.Y, param.axis, MaxFunctor<T>(), param.Out);
  }

  virtual ~ElementwiseMaxCompute() = default;
};

template <typename T>
class ElementwiseMinCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ElementwiseParam;
  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<X86Context>();
    param.Out->template mutable_data<T>();
    ElementwiseComputeEx<MinFunctor<T>, lite::TargetType::kX86, T>(
        context, param.X, param.Y, param.axis, MinFunctor<T>(), param.Out);
  }

  virtual ~ElementwiseMinCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
