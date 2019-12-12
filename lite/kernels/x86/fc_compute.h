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
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/operators/fc_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

inline void FCOutputSize(const lite::DDim& in_dims,
                         const lite::DDim& w_dims,
                         std::vector<int64_t>& out_dims,  // NOLINT
                         int in_num_col_dims,
                         bool padding_weights) {
  auto w_dims1 = padding_weights ? w_dims[1] - 4 : w_dims[1];

  out_dims.reserve(static_cast<size_t>(in_num_col_dims + 1));
  for (int i = 0; i < in_num_col_dims; ++i) {
    out_dims.push_back(in_dims[i]);
  }
  out_dims.push_back(w_dims1);
}

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
    lite::Tensor Y1;
    T* Y1_data = nullptr;
    if (N % 128 == 0 && K % 128 == 0) {
      const int NN = N + 4;
      const int KK = K + 4;
      lite::Tensor X1;
      X1.Resize({M * KK});
      Y1.Resize({M * (N + 4)});
      T* X1_data = X1.mutable_data<T>();
      Y1_data = Y1.mutable_data<T>();
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int i = 0; i < M; i++) {
        memcpy(X1_data + i * KK, X + i * K, K * sizeof(X[0]));
      }
      lite::Tensor W1;
      T* W1_data = nullptr;
      if (!padding_weights) {
        W1.Resize({(K + 4) * (N + 4)});
        W1_data = W1.mutable_data<T>();
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
        for (int i = 0; i < K; i++) {
          memcpy(W1_data + i * NN, W + i * N, N * sizeof(W[0]));
        }
      }
      blas.GEMM(false,
                false,
                M,
                N,
                K,
                static_cast<T>(1.0),
                X1_data,
                KK,
                (padding_weights ? W : W1_data),
                NN,
                static_cast<T>(0.0),
                Y1_data,
                NN);
    } else {
      blas.MatMul(M, N, K, X, W, Y);
    }
    if (B == NULL) {
      if (N % 128 == 0 && K % 128 == 0) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
        for (int i = 0; i < M; i++) {
          memcpy(Y + i * N, Y1_data + i * (N + 4), N * sizeof(Y[0]));
        }
      }
      return;
    }
    if (relu) {
      auto compute =
          paddle::lite::jit::KernelFuncs<paddle::lite::jit::VAddReluTuple<T>,
                                         lite::fluid::CPUPlace>::Cache()
              .At(N);
      for (int i = 0; i < M; i++) {
        T* dst = Y + i * N;
        T* src = (N % 128 == 0 && K % 128 == 0) ? Y1_data + i * (N + 4) : dst;
        compute(B, src, dst, N);
      }
    } else {
      auto compute =
          paddle::lite::jit::KernelFuncs<paddle::lite::jit::VAddTuple<T>,
                                         lite::fluid::CPUPlace>::Cache()
              .At(N);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int i = 0; i < M; i++) {
        T* dst = Y + i * N;
        T* src = (N % 128 == 0 && K % 128 == 0) ? Y1_data + i * (N + 4) : dst;
        compute(B, src, dst, N);
      }
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
    int in_num_col_dims = param.in_num_col_dims;
    bool with_relu = (param.activation_type == "relu") ? true : false;

    auto w_dims = w->dims();
    bool padding_weights = param.padding_weights;

    std::vector<int64_t> output_dims;
    FCOutputSize(
        input->dims(), w_dims, output_dims, in_num_col_dims, padding_weights);
    output->Resize(output_dims);
    output->set_lod(input->lod());

    auto out_dims = output->dims();
    auto w_dims0 = padding_weights ? w_dims[0] - 4 : w_dims[0];
    auto w_dims1 = padding_weights ? w_dims[1] - 4 : w_dims[1];
    int M = out_dims.production() / w_dims1;

    const T* input_data = input->data<T>();
    const T* w_data = w->data<T>();
    T* output_data = output->mutable_data<T>();

    auto& context = ctx_->As<X86Context>();
    FCFunctor<lite::TargetType::kX86, T> fc;
    fc(context,
       M,
       w_dims1,
       w_dims0,
       input_data,
       w_data,
       output_data,
       bias ? bias->data<T>() : NULL,
       with_relu,
       padding_weights);
  }

  virtual ~FcCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
