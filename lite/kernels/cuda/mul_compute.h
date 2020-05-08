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
#include "lite/backends/cuda/blas.h"
#include "lite/core/context.h"
#include "lite/core/kernel.h"
#include "lite/core/types.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T>
void mul_compute(const lite::cuda::Blas<float>& blas,
                 const T* x,
                 int x_h,
                 int x_w,
                 const T* y,
                 int y_h,
                 int y_w,
                 T* out) {
  float alpha = 1.0;
  float beta = 0.0;
  /*
  blas.sgemm(CUBLAS_OP_N,
             CUBLAS_OP_N,
             x_h,
             y_w,
             x_w,
             &alpha,
             x,
             x_w,
             y,
             y_w,
             &beta,
             out,
             x_h);
  */
  blas.sgemm(CUBLAS_OP_N,
             CUBLAS_OP_N,
             y_w,
             x_h,
             y_h,
             &alpha,
             y,
             y_w,
             x,
             x_w,
             &beta,
             out,
             y_w);
}

class MulCompute : public KernelLite<TARGET(kCUDA), PRECISION(kFloat)> {
 public:
  using param_t = operators::MulParam;

  void Run() override {
    CHECK(ctx_) << "running context should be set first";
    auto& context = this->ctx_->template As<CUDAContext>();
    CHECK(context.cublas_fp32()) << "blas should init first";
    auto& blas = *context.cublas_fp32();

    auto& param = this->Param<param_t>();
    const auto* x_data = param.x->data<float>();
    const auto* y_data = param.y->data<float>();
    auto* out_data = param.output->mutable_data<float>(TARGET(kCUDA));

    int x_h = static_cast<int>(
        param.x->dims().Slice(0, param.x_num_col_dims).production());
    int x_w = static_cast<int>(
        param.x->dims()
            .Slice(param.x_num_col_dims, param.x->dims().size())
            .production());
    int y_h = static_cast<int>(
        param.y->dims().Slice(0, param.y_num_col_dims).production());
    int y_w = static_cast<int>(
        param.y->dims()
            .Slice(param.y_num_col_dims, param.y->dims().size())
            .production());
    CHECK_EQ(x_w, y_h) << "x_w must be equal with y_h";

    mul_compute<float>(blas, x_data, x_h, x_w, y_data, y_h, y_w, out_data);
  }

  virtual ~MulCompute() = default;
};

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
