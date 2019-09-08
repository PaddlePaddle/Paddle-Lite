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
#include "lite/kernels/cuda/elementwise_add_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

__global__ void KeElementwiseAdd(const float* x_data,
                                 const float* y_data,
                                 float* out_data,
                                 const size_t total) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < total; tid += stride) {
    out_data[tid] = x_data[tid] + y_data[tid];
  }
}

void ElementwiseAddCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  const lite::Tensor* x = param.X;
  const lite::Tensor* y = param.Y;
  lite::Tensor* out = param.Out;

  CHECK(x->dims() == y->dims());

  const int n = x->dims()[0];
  const int c = x->dims()[1];
  const int h = x->dims()[2];
  const int w = x->dims()[3];

  auto* x_data = x->data<float>();
  auto* y_data = y->data<float>();
  auto out_data = out->mutable_data<float>(TARGET(kCUDA));

  int pixel_num = x->numel();
  int threads = 512;
  int blocks = (pixel_num + threads - 1) / threads;
  blocks = blocks > 8 ? 8 : blocks;

  KeElementwiseAdd<<<blocks, threads, 0, stream>>>(
      x_data, y_data, out_data, pixel_num);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(elementwise_add,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::ElementwiseAddCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
