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

#include <string>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"
#include "lite/fluid/eigen.h"
#include "lite/x86/math/blas.h"
#include "lite/x86/math/gru_compute.h"
#include "lite/x86/math/math_function.h"
#include "lite/x86/math/sequence2batch.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

using Tensor = lite::Tensor;

template <typename T>
struct SequenceExpandFunctor {
  void operator()(
      const Tensor &x, 
      const std::vector<size_t> &ref_lod, /*expand referenced lod*/
      Tensor *out) {
    int64_t hight = x.dims()[0];
    int64_t width = x.data_size() / hight;

    const T *in_data = x.data<T>();
    T *out_data = out->mutable_data<T, T>();

    for (int h_id = 0; h_id < hight; ++h_id) {
      size_t span = ref_lod[h_id + 1] - ref_lod[h_id];
      if (span == 0) continue;
      const T *src = in_data + h_id * width;
      for (int64_t w_id = 0; w_id < width; ++w_id) {
        T ele = src[w_id];
        size_t offset = ref_lod[h_id] * width;
        for (size_t k = 0; k < span; ++k) {
          out_data[offset + k * width + w_id] = ele;
        }
      }
    }
  }
};

template <typename T>
class SequenceExpandAsCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override {
    auto& param = *param_.get_mutable<operators::SequenceExpandAsParam>();

    auto* x = param.x;
    auto* y = param.y;
    auto* out = param.out;

    auto &y_lod = y->lod();
    CHECK_EQ_OR_RETURN(y_lod.size(), 1)
    CHECK_GT_OR_RETURN(y_lod[0].size(), 1)

    out->mutable_data<T, T>();

    SequenceExpandFunctor<T> seq_espand_functor;
    seq_espand_functor(*x, y_lod[0], out);
  }
};

/*
 *Given Grad(Out)
 *
 *    Grad(Out).lod = [[0,              3,            6]]
 *    Grad(Out).data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
 * Then
 *    Grad(X).data = [(0.1 + 0.2 + 0.3), (0.4 + 0.5 + 0.6)]
 *                 = [0.6, 1.5]
 *    Grad(X).lod = Input(X).lod
 *
 * */
template <typename T>
struct SequenceExpandAsGradFunctor {
  void operator()(
      const Tensor &dout,
      const std::vector<size_t> &ref_lod, /*expand referenced lod*/
      Tensor *dx) {
    int64_t hight = dx->dims()[0];
    int64_t width = dx->data_size() / hight;

    const T *dout_data = dout.data<T>();
    T *dx_data = dx->mutable_data<T, T>();

    for (int64_t h_id = 0; h_id < hight; ++h_id) {
      T *dst = dx_data + h_id * width;
      size_t span = ref_lod[h_id + 1] - ref_lod[h_id];
      for (int64_t w_id = 0; w_id < width; ++w_id) {
        T result = 0;
        for (size_t k = 0; k < span; ++k) {
          size_t offset = (ref_lod[h_id] + k) * width;
          result += dout_data[offset + w_id];
        }
        dst[w_id] = result;
      }
    }
  }
};

template <typename T>
class SequenceExpandAsGradCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override {
    auto& param = *param_.get_mutable<operators::SequenceExpandAsGradParam>();

    auto* y = param.y;
    auto* out_grad = param.out_grad;
    auto* x_grad = param.x_grad;
    x_grad->mutable_data<T, T>();

    SequenceExpandAsGradFunctor<T> functor;
    functor(*out_grad, y->lod()[0], x_grad);
  }
};

}  // x86
}  // kernels
}  // lite
}  // paddle
