/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/search_fc_compute.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T>
static __global__ void add_bias(int n,
                                int output_size,
                                const T* bias,
                                T* dout) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int bias_index = index % output_size;
  if (index < n) {
    dout[index] = dout[index] + bias[bias_index];
  }
}

template <typename T>
void SearchFcCompute<T>::PrepareForRun() {
  gemm_impl_.reset(new lite::cuda::math::Gemm<float, float>);
}

template <typename T>
void SearchFcCompute<T>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();
  const Tensor* x_tensor = param.X;
  param.Out->Resize({x_tensor->dims()[0], param.out_size});
  _M = x_tensor->dims().count(0, 1);
  _K = x_tensor->dims().count(1, x_tensor->numel());
  _N = param.out_size;
  const T* din = x_tensor->data<T>();
  Tensor* out_tensor = param.Out;
  T* dout = out_tensor->mutable_data<T>(TARGET(kCUDA));
  const Tensor* w_tensor = param.W;
  const T* weight = w_tensor->data<T>();
  const Tensor* b_tensor = param.b;
  const T* bias = b_tensor->data<T>();

  CHECK(gemm_impl_->init(false, true, _M, _N, _K, &ctx));
  gemm_impl_->run(1.0f, 0.0f, din, weight, dout, &ctx);

  int total_size = _M * _N;
  add_bias<T><<<CUDA_GET_BLOCKS(total_size), CUDA_NUM_THREADS, 0, stream>>>(
      total_size, _N, bias, dout);
}
}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(search_fc,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::SearchFcCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("b", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
