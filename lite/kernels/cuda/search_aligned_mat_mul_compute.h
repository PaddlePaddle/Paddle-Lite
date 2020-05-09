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
#include <limits>
#include <memory>
#include "lite/backends/cuda/math/batched_gemm.h"
#include "lite/core/context.h"
#include "lite/core/kernel.h"
#include "lite/core/types.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class SearchAlignedMatMulCompute
    : public KernelLite<TARGET(kCUDA), PRECISION(kFloat)> {
 public:
  using param_t = operators::MatMulParam;

  void PrepareForRun() override {
    batched_gemm_impl_.reset(new lite::cuda::math::BatchedGemm<float, float>);
    last_seq_num_ = std::numeric_limits<int>::min();
  }

  void Run() override {
    auto& param = this->Param<param_t>();
    auto& cuda_ctx = ctx_->template As<CUDAContext>();
    auto x = param.X;
    auto y = param.Y;
    auto out = param.Out;
    bool x_transpose = param.transpose_X;
    bool y_transpose = param.transpose_Y;
    float alpha = param.alpha;
    const auto& x_dims = x->dims();
    const auto& y_dims = y->dims();
    const auto& x_lod = x->lod();
    const auto& y_lod = y->lod();
    const auto& x_lod_0 = x_lod[0];
    const auto& y_lod_0 = y_lod[0];
    int seq_num = x_lod_0.size() - 1;
    int x_inner_size = x_dims[1];
    int y_inner_size = y_dims[1];
    int x_batch_size = x_lod_0[1];
    int y_batch_size = y_lod_0[1];
    int M = x_transpose ? x_inner_size : x_batch_size;
    int N = y_transpose ? y_batch_size : y_inner_size;
    int X_K = x_transpose ? x_batch_size : x_inner_size;
    int Y_K = y_transpose ? y_inner_size : y_batch_size;
    CHECK_EQ(X_K, Y_K) << "K of Input(X) and Input(Y) is not equal";
    int K = X_K;

    auto x_data = x->data<float>();
    auto y_data = y->data<float>();
    auto out_data = out->mutable_data<float>(TARGET(kCUDA));
    auto x_stride = x_batch_size * x_inner_size;
    auto y_stride = y_batch_size * y_inner_size;
    auto out_stride = M * N;

    float* A_[seq_num * 3];
    for (int seq = 0; seq < seq_num; ++seq) {
      A_[seq] = const_cast<float*>(x_data) + seq * x_stride;
      A_[seq + seq_num] = const_cast<float*>(y_data) + seq * y_stride;
      A_[seq + seq_num * 2] = out_data + seq * out_stride;
    }

    if (seq_num != last_seq_num_) {
      CHECK(batched_gemm_impl_->init(
          x_transpose, y_transpose, seq_num, &cuda_ctx));
      last_seq_num_ = seq_num;
    }
    batched_gemm_impl_->run(
        alpha, 0.0f, const_cast<const float**>(A_), M, N, K, seq_num);
  }

  ~SearchAlignedMatMulCompute() { batched_gemm_impl_.reset(); }

 private:
  std::unique_ptr<lite::cuda::math::BatchedGemm<float, float>>
      batched_gemm_impl_;
  int last_seq_num_;
};

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
