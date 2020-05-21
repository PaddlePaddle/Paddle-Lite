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
#include "lite/backends/x86/jit/helper.h"
#include "lite/backends/x86/jit/kernel_base.h"
#include "lite/backends/x86/jit/kernels.h"
#include "lite/backends/x86/math/blas.h"
#include "lite/backends/x86/parallel.h"
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/operators/fc_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <lite::TargetType Target, typename T>
class FCFunctor {
 public:
  void operator()(const lite::X86Context& context,
                  const int M,
                  const int N,
                  const int K,
                  const T* X,
                  const T* W,
                  T* Y,
                  const T* B = nullptr,
                  bool relu = false,
                  bool padding_weights = false) {
    auto blas = lite::x86::math::GetBlas<lite::TargetType::kX86, T>(context);
    T* Y1_data = nullptr;

    auto compute =
        relu
            ? jit::KernelFuncs<jit::VAddReluTuple<T>, fluid::CPUPlace>::Cache()
                  .At(N)
            : jit::KernelFuncs<jit::VAddTuple<T>, fluid::CPUPlace>::Cache().At(
                  N);
    auto parallel_compute = [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; i++) {
        T* dst = Y + i * N;
        T* src = Y1_data ? Y1_data + i * (N + 4) : dst;
        compute(B, src, dst, N);
      }
    };

    // Because of the overhead of memcpy, we only do padding for GEMM
    //  when weights is already padded in fc_fuse_pass.
    if (padding_weights) {
      const int NN = N + 4;
      const int KK = K + 4;

      // NOTE: here need to mutable_data for temporary Tensor X1 and Y1,
      //  the overhead is unmeasured.
      lite::Tensor X1;
      X1.Resize(std::vector<int64_t>{M * KK});
      T* X1_data = X1.mutable_data<T>();

      lite::Tensor Y1;
      Y1.Resize(std::vector<int64_t>{M * NN});
      Y1_data = Y1.mutable_data<T>();

      auto parallel_memcpy_x = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          memcpy(X1_data + i * KK, X + i * K, K * sizeof(T));
        }
      };
      lite::x86::RunParallelFor(0, M, parallel_memcpy_x);

      blas.GEMM(false,
                false,
                M,
                N,
                K,
                static_cast<T>(1.0),
                X1_data,
                KK,
                W,
                NN,
                static_cast<T>(0.0),
                Y1_data,
                NN);

      if (!B) {
        auto parallel_memcpy_y = [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; i++) {
            memcpy(Y + i * N, Y1_data + i * NN, N * sizeof(T));
          }
        };
        lite::x86::RunParallelFor(0, M, parallel_memcpy_y);
        return;
      }

      lite::x86::RunParallelFor(0, M, parallel_compute);
    } else {
      blas.MatMul(M, N, K, X, W, Y);
      if (!B) {
        return;
      }

      lite::x86::RunParallelFor(0, M, parallel_compute);
    }
  }
};

template <typename T>
class FcCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::FcParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto* input = param.input;
    auto* w = param.w;
    auto* bias = param.bias;
    auto* output = param.output;
    bool with_relu = (param.activation_type == "relu") ? true : false;

    bool padding_weights = param.padding_weights;
    const auto& w_dims = w->dims();
    auto w_dims0 = padding_weights ? w_dims[0] - 4 : w_dims[0];
    auto w_dims1 = padding_weights ? w_dims[1] - 4 : w_dims[1];

    int M = output->dims().production() / w_dims1;

    const T* input_data = input->template data<T>();
    const T* w_data = w->template data<T>();
    T* output_data = output->template mutable_data<T>();

    auto& context = ctx_->As<X86Context>();
    FCFunctor<lite::TargetType::kX86, T> fc;
    fc(context,
       M,
       w_dims1,
       w_dims0,
       input_data,
       w_data,
       output_data,
       bias ? bias->template data<T>() : NULL,
       with_relu,
       padding_weights);
  }

  virtual ~FcCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
