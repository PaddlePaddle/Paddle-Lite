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

#include <vector>
#include "lite/backends/x86/fluid/eigen.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/x86/reduce_op_function.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

struct SumFunctor {
  template <typename X, typename Y, typename Dim>
  void operator()(X* x, Y* y, const Dim& dim) {
    y->device(lite::fluid::EigenDeviceType<TARGET(kX86)>()) = x->sum(dim);
  }
};

struct ProdFunctor {
  template <typename X, typename Y, typename Dim>
  void operator()(X* x, Y* y, const Dim& dim) {
    y->device(lite::fluid::EigenDeviceType<TARGET(kX86)>()) = x->prod(dim);
  }
};

struct MeanFunctor {
  template <typename X, typename Y, typename Dim>
  void operator()(X* x, Y* y, const Dim& dim) {
    y->device(lite::fluid::EigenDeviceType<TARGET(kX86)>()) = x->mean(dim);
  }
};

struct MaxFunctor {
  template <typename X, typename Y, typename Dim>
  void operator()(X* x, Y* y, const Dim& dim) {
    y->device(lite::fluid::EigenDeviceType<TARGET(kX86)>()) = x->maximum(dim);
  }
};

struct MinFunctor {
  template <typename X, typename Y, typename Dim>
  void operator()(X* x, Y* y, const Dim& dim) {
    y->device(lite::fluid::EigenDeviceType<TARGET(kX86)>()) = x->minimum(dim);
  }
};

#define HANDLE_DIM(NDIM, RDIM, FUNCTOR)                                \
  if (ndim == NDIM && rdim == RDIM) {                                  \
    paddle::lite::kernels::x86::                                       \
        ReduceFunctor<lite::TargetType::kX86, T, NDIM, RDIM, FUNCTOR>( \
            *x, out, dims, keep_dim);                                  \
  }

template <typename T, typename Functor>
class ReduceCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ReduceParam;

  void Run() override {
    auto& param = *param_.get_mutable<operators::ReduceParam>();
    auto* x = param.X;
    auto* out = param.Out;
    out->template mutable_data<T>();
    auto x_dims = x->dims();

    const auto& dims = param.dim;
    bool keep_dim = param.keep_dim;
    bool reduce_all = param.reduce_all;
    if (reduce_all || dims.empty() || x_dims.size() == 1 ||
        x_dims.size() == dims.size()) {
      // Flatten and reduce 1-D tensor
      auto x_e = lite::fluid::EigenVector<T>::Flatten(*x);
      auto out_e = lite::fluid::EigenScalar<T>::From(out);
      auto reduce_dim = Eigen::array<int, 1>({{0}});
      Functor functor;
      functor(&x_e, &out_e, reduce_dim);
    } else {
      int ndim = x_dims.size();
      int rdim = dims.size();
      HANDLE_DIM(4, 3, Functor);
      HANDLE_DIM(4, 2, Functor);
      HANDLE_DIM(4, 1, Functor);
      HANDLE_DIM(3, 2, Functor);
      HANDLE_DIM(3, 1, Functor);
      HANDLE_DIM(2, 1, Functor);
    }
  }

  virtual ~ReduceCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
