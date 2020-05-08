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
#include "lite/kernels/cuda/lookup_table_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {
using Tensor = lite::Tensor;

template <int BlockDimX, int BlockDimY, int GridDimX, bool PaddingFlag>
__global__ void LookupTableKernel(float *output,
                                  const float *table,
                                  const int64_t *ids,
                                  const int64_t N,
                                  const int64_t K,
                                  const int64_t D,
                                  const int64_t padding_idx) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * GridDimX;

  while (idy < K) {
    int64_t id = ids[idy];
    float *out = output + idy * D;
    const float *tab = table + id * D;
    for (int i = idx; i < D; i += BlockDimX) {
      if (PaddingFlag) {
        if (id == padding_idx)
          out[i] = static_cast<float>(0);
        else
          out[i] = tab[i];
      } else {
        out[i] = tab[i];
      }
    }
    idy += BlockDimY * GridDimX;
  }
}

void LookupTableCompute::Run() {
  auto &param = this->Param<param_t>();
  auto &ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();
  const Tensor *w_t = param.W;
  const Tensor *ids_t = param.Ids;
  Tensor *out_t = param.Out;
  int64_t padding_idx = param.padding_idx;

  size_t N = w_t->dims()[0];
  size_t D = w_t->dims()[1];
  size_t K = ids_t->numel();

  auto *w = w_t->data<float>();
  auto *ids = ids_t->data<int64_t>();
  auto *out = out_t->mutable_data<float>(TARGET(kCUDA));

  dim3 threads(128, 8);
  dim3 grids(8, 1);

  if (padding_idx == -1) {
    LookupTableKernel<128, 8, 8, false><<<grids, threads, 0, stream>>>(
        out, w, ids, N, K, D, padding_idx);
  } else {
    LookupTableKernel<128, 8, 8, true><<<grids, threads, 0, stream>>>(
        out, w, ids, N, K, D, padding_idx);
  }

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(lookup_table,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::LookupTableCompute,
                     def)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFloat))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFloat))})
    .Finalize();
REGISTER_LITE_KERNEL(lookup_table_v2,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::LookupTableCompute,
                     def)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFloat))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFloat))})
    .Finalize();
