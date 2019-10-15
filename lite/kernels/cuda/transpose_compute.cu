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
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/transpose_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

void TransposeCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();

  const lite::Tensor* X = param.x;
  lite::Tensor* Out = param.output;
  std::vector<int> axes = param.axis;

  const float* in = X->data<float>();
  float* out = Out->mutable_data<float>(TARGET(kCUDA));

  int ndim = X->dims().size();
  std::vector<int64_t> dims = X->dims().data();

  // NCHW -> NHWC
  if (axes.size() == 4 && axes[0] == 0 && axes[1] == 2 && axes[2] == 3 &&
      axes[3] == 1) {
    lite::cuda::math::NCHW2NHWC(
        dims[0], dims[1], dims[2] * dims[3], in, out, &ctx);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
    return;
  }

  // NHWC -> NCHW
  if (axes.size() == 4 && axes[0] == 0 && axes[1] == 3 && axes[2] == 1 &&
      axes[3] == 2) {
    lite::cuda::math::NHWC2NCHW(
        dims[0], dims[3], dims[1] * dims[2], in, out, &ctx);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
    return;
  }

  lite::cuda::math::Transpose(dims, axes, in, out, &ctx);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(transpose,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::TransposeCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(transpose2,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::TransposeCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

// REGISTER_LITE_KERNEL(transpose2,
//                      kCUDA,
//                      kFloat,
//                      kNCHW,
//                      paddle::lite::kernels::cuda::TransposeCompute,
//                      def)
//     .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
//     .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
//     .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kCUDA))})
//     .Finalize();
