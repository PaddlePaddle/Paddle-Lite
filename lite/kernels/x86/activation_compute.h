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

#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/fluid/eigen.h"
#include "lite/core/op_lite.h"
#include "lite/operators/activation_ops.h"

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
bool Activate(const lite::Tensor* X,
              lite::Tensor* Out) {
  using T = typename Functor::ELEMENT_TYPE;
  auto place = lite::fluid::EigenDeviceType<TARGET(kX86)>();
  CHECK_OR_FALSE(X)
  CHECK_OR_FALSE(Out)
  auto x =
      lite::fluid::EigenVector<T>::Flatten(*X);
  auto out =
      lite::fluid::EigenVector<T>::Flatten(*Out);
  Functor()(place, x, out);
}

template <typename Functor>
bool ActivateGrad(const lite::Tensor* X,
                  const lite::Tensor* Out,
                  const lite::Tensor* Out_grad,
                  lite::Tensor* X_grad) {
  using T = typename Functor::ELEMENT_TYPE;
  auto place = lite::fluid::EigenDeviceType<TARGET(kX86)>();
  CHECK_OR_FALSE(X)
  CHECK_OR_FALSE(Out)
  CHECK_OR_FALSE(Out_grad)
  CHECK_OR_FALSE(X_grad)
  auto x =
      lite::fluid::EigenVector<T>::Flatten(*X);
  auto out =
      lite::fluid::EigenVector<T>::Flatten(*Out);
  auto x_grad = 
      lite::fluid::EigenVector<T>::Flatten(*X_grad);
  auto out_grad = 
      lite::fluid::EigenVector<T>::Flatten(*Out_grad);
  Functor()(place, x, out, out_grad, x_grad);
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
struct SquareGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * static_cast<T>(2) * x;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
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

template <typename T>
class SquareGradCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationGradParam;

  void Run() override {
    auto& param = *param_.get_mutable<operators::ActivationGradParam>();
    param.X_grad->template mutable_data<T>();

    ActivateGrad<SquareGradFunctor<T>>(
        param.X, param.Out,
        param.Out_grad, param.X_grad);
  }

  virtual ~SquareGradCompute() = default;
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
struct ReluGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * (out > static_cast<T>(0)).template cast<T>();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
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
class ReluGradCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationGradParam;

  void Run() override {
    auto& param = *param_.get_mutable<operators::ActivationGradParam>();
    param.X_grad->template mutable_data<T>();

    ActivateGrad<ReluGradFunctor<T>>(
        param.X, param.Out,
        param.Out_grad, param.X_grad);
  }

  virtual ~ReluGradCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
