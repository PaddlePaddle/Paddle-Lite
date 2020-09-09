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
#include <memory>
#include "lite/backends/cuda/math/gemm.h"
#include "lite/core/kernel.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T, PrecisionType PType>
class MulCompute : public KernelLite<TARGET(kCUDA), PType> {
 public:
  using param_t = operators::MulParam;

  void PrepareForRun() override {
    gemm_impl_.reset(new lite::cuda::math::Gemm<T, T>);
  }

  void Run() override {
    auto& context = this->ctx_->template As<CUDAContext>();
    auto& param = this->template Param<param_t>();
    const auto* x_data = param.x->template data<T>();
    const auto* y_data = param.y->template data<T>();
    auto* out_data = param.output->template mutable_data<T>(TARGET(kCUDA));

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

    CHECK(gemm_impl_->init(false, false, x_h, y_w, x_w, &context));
    gemm_impl_->run(1.0f, 0.0f, x_data, y_data, out_data, &context);
  }

  virtual ~MulCompute() = default;

 private:
  std::unique_ptr<lite::cuda::math::Gemm<T, T>> gemm_impl_{nullptr};
};

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
