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
#include <utility>
#include <vector>

#include <cmath>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include "lite/backends/x86/math/blas.h"
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/fluid/eigen.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

enum ActBwdOpFwdDeps {
  kNoDeps = 0x00,  // Do not need any forward input/output
  kDepX = 0x01,    // Only need forward input X
  kDepOut = 0x02,  // Only need forward output Out

  // Never add kDepXOut, because Out can be always calculated
  // by forward input X in backward part.
  // FIXME(zjl): but in MKLDNN abs, X and Out are all needed...
  // Developers should not rely on this enum value!
  kDepXOut = 0x03
};

template <typename T>
struct BaseActivationFunctor {
  using ELEMENT_TYPE = T;

  using AttrPair = std::vector<std::pair<const char*, float*>>;

  AttrPair GetAttrs() { return AttrPair(); }

  /* NOTE(*): Output reuse X memory if X is not dependented by its Gradient.
     For example, sigmoid op's gradient didn't involve x, so its output can
     reuse
     input memory. But abs op's gradient use x, it can not be inplaced.
     gradient did use x.
   */
  bool Inplace() const { return false; }
};

template <typename Functor>
bool Activate(const lite::Tensor* X, lite::Tensor* Out) {
  using T = typename Functor::ELEMENT_TYPE;
  auto place = lite::fluid::EigenDeviceType<TARGET(kX86)>();
  CHECK_OR_FALSE(X)
  CHECK_OR_FALSE(Out)
  auto x = lite::fluid::EigenVector<T>::Flatten(*X);
  auto out = lite::fluid::EigenVector<T>::Flatten(*Out);
  Functor()(place, x, out);
  return true;
}

// square(x) = x^2
template <typename T>
struct SquareFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.square();
  }
};

template <typename T>
class SquareCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override {
    auto& param = *param_.get_mutable<operators::ActivationParam>();

    param.Out->template mutable_data<T>();
    Activate<SquareFunctor<T>>(param.X, param.Out);
  }

  virtual ~SquareCompute() = default;
};

// relu(x) = max(x, 0)
template <typename T>
struct ReluFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.cwiseMax(static_cast<T>(0));
  }
};

template <typename T>
class ReluCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override {
    auto& param = *param_.get_mutable<operators::ActivationParam>();

    param.Out->template mutable_data<T>();
    Activate<ReluFunctor<T>>(param.X, param.Out);
  }

  virtual ~ReluCompute() = default;
};

template <typename T>
struct LeakyReluFunctor {
  float alpha;
  explicit LeakyReluFunctor(float alpha_) : alpha(alpha_) {}

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.cwiseMax(static_cast<T>(alpha) * x);
  }
};

template <typename T>
class LeakyReluCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override {
    auto& param = *param_.get_mutable<operators::ActivationParam>();

    param.Out->template mutable_data<T>();
    auto X = param.X;
    auto Out = param.Out;
    auto place = lite::fluid::EigenDeviceType<TARGET(kX86)>();
    CHECK(X);
    CHECK(Out);
    auto x = lite::fluid::EigenVector<T>::Flatten(*X);
    auto out = lite::fluid::EigenVector<T>::Flatten(*Out);
    LeakyReluFunctor<T> functor(param.Leaky_relu_alpha);
    functor(place, x, out);
  }

  virtual ~LeakyReluCompute() = default;
};

// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
template <typename T>
struct TanhFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.tanh();
  }
};

template <typename T>
class TanhCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override {
    auto& param = *param_.get_mutable<operators::ActivationParam>();

    param.Out->template mutable_data<T>();
    Activate<TanhFunctor<T>>(param.X, param.Out);
  }

  virtual ~TanhCompute() = default;
};

// gelu(x) = 0.5 * x *  (1 + erf(x / sqrt(2)))
template <typename T>
struct GeluFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
// Because the execute or device context can not be deliver here, it keep the
// marco for NVCC.
#if defined(PADDLE_WITH_MKLML) && !defined(_WIN32) && !defined(__APPLE__) && \
    !defined(__OSX__) && !defined(PADDLE_WITH_CUDA)
    auto x_data = x.data();
    auto out_data = out.data();
    int n = std::min(x.size(), out.size());

    std::memset(out_data, 0, n * sizeof(T));
    paddle::lite::x86::math::CBlas<T>::AXPY(
        n, static_cast<T>(M_SQRT1_2), x_data, 1, out_data, 1);
    paddle::lite::x86::math::CBlas<T>::VMERF(n, out_data, out_data, VML_LA);
    for (int i = 0; i < n; i++) {
      out_data[i] += static_cast<T>(1);
    }
    paddle::lite::x86::math::CBlas<T>::VMUL(n, x_data, out_data, out_data);
    for (int i = 0; i < n; i++) {
      out_data[i] *= static_cast<T>(0.5);
    }
#else
    auto temp = (x * static_cast<T>(M_SQRT1_2)).erf();
    out.device(d) = x * static_cast<T>(0.5) * (static_cast<T>(1) + temp);
#endif
  }
};

template <typename T>
class GeluCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override {
    auto& param = *param_.get_mutable<operators::ActivationParam>();

    param.Out->template mutable_data<T>();
    Activate<GeluFunctor<T>>(param.X, param.Out);
  }

  virtual ~GeluCompute() = default;
};

// softsign(x) = x / (1 + |x|)
template <typename T>
class SoftsignCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override {
    // auto& context = ctx_->As<X86Context>();
    auto& param = *param_.get_mutable<operators::ActivationParam>();

    const T* x_data = param.X->template data<T>();
    T* out_data = param.Out->template mutable_data<T>();
    size_t x_size = param.X->numel();
    for (size_t i = 0; i < x_size; i++) {
      out_data[i] = x_data[i] / (static_cast<T>(1) + std::abs(x_data[i]));
    }
  }

  virtual ~SoftsignCompute() = default;
};

// sigmoid(x) = 1 / (1 + exp(-x))
template <typename T>
struct SigmoidFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = (static_cast<T>(1) + (-x).exp()).inverse();
  }
};

template <typename T>
class SigmoidCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override {
    auto& param = this->Param<param_t>();
    param.Out->template mutable_data<T>();
    Activate<SigmoidFunctor<T>>(param.X, param.Out);
  }

  virtual ~SigmoidCompute() = default;
};

// relu6(x) = min(max(0, x), 6)
template <typename T>
struct Relu6Functor {
  float threshold;
  explicit Relu6Functor(float threshold_) : threshold(threshold_) {}

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) =
        x.cwiseMax(static_cast<T>(0)).cwiseMin(static_cast<T>(threshold));
  }
};

template <typename T>
class Relu6Compute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override {
    auto& param = *param_.get_mutable<operators::ActivationParam>();

    param.Out->template mutable_data<T>();
    auto X = param.X;
    auto Out = param.Out;
    auto place = lite::fluid::EigenDeviceType<TARGET(kX86)>();
    CHECK(X);
    CHECK(Out);
    auto x = lite::fluid::EigenVector<T>::Flatten(*X);
    auto out = lite::fluid::EigenVector<T>::Flatten(*Out);
    Relu6Functor<T> functor(param.threshold);
    functor(place, x, out);
  }

  virtual ~Relu6Compute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
