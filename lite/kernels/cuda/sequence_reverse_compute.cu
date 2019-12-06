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

#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/kernels/cuda/sequence_reverse_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T>
__host__ __device__ inline size_t UpperBound(const T* x,
                                             size_t num,
                                             const T& val) {
  // The following code is from
  // https://en.cppreference.com/w/cpp/algorithm/upper_bound
  auto* first = x;
  int64_t count = static_cast<int64_t>(num);
  while (count > 0) {
    auto step = (count >> 1);
    auto* it = first + step;
    if (val < *it) {
      count = step;
    } else {
      first = ++it;
      count -= (step + 1);
    }
  }
  return static_cast<size_t>(first - x);
}

template <typename T>
__global__ void SequenceReverseKernelGridIsOne(
    const T* x, T* y, const int64_t* lod, size_t lod_count, int64_t row_numel) {
  int64_t idx = static_cast<int64_t>(threadIdx.x);
  auto row_idx_x = idx / row_numel;
  auto lod_idx = UpperBound(lod, lod_count, row_idx_x);
  auto row_idx_y = lod[lod_idx - 1] + (lod[lod_idx] - 1 - row_idx_x);
  auto idx_y = row_idx_y * row_numel + idx % row_numel;
  y[idx_y] = x[idx];
}

template <typename T>
__global__ void SequenceReverseKernel(const T* x,
                                      T* y,
                                      const int64_t* lod,
                                      size_t lod_count,
                                      int64_t row_numel,
                                      size_t limit) {
  int64_t idx = static_cast<int64_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < limit) {
    auto row_idx_x = idx / row_numel;
    auto lod_idx = UpperBound(lod, lod_count, row_idx_x);
    auto row_idx_y = lod[lod_idx - 1] + (lod[lod_idx] - 1 - row_idx_x);
    auto idx_y = row_idx_y * row_numel + idx % row_numel;
    y[idx_y] = x[idx];
  }
}

template <typename T, PrecisionType Ptype>
void SequenceReverseCompute<T, Ptype>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();
  size_t limit = static_cast<size_t>(param.X->numel());
  int64_t row_numel = static_cast<int64_t>(limit / param.X->dims()[0]);
  const auto* x_data = param.X->template data<T>();
  auto y_data = param.Out->template mutable_data<T>(TARGET(kCUDA));
  CHECK_NE(x_data, y_data)
      << "SequenceReverse Op does not support in-place operation";
  const auto lod = param.X->lod()[param.X->lod().size() - 1];
  const size_t lod_count = lod.size();
  param.Out->set_lod(param.X->lod());

  lod_cuda.Resize({static_cast<int64_t>(lod.size())});
  int64_t* lod_data = lod_cuda.mutable_data<int64_t>(TARGET(kCUDA));
  TargetWrapperCuda::MemcpyAsync(lod_data,
                                 lod.data(),
                                 sizeof(int64_t) * lod.size(),
                                 IoDirection::HtoD,
                                 stream);
  constexpr int num_threads = 1024;
  int block_size = limit <= num_threads ? limit : num_threads;
  int grid_size = (limit + num_threads - 1) / num_threads;
  if (grid_size == 1) {
    SequenceReverseKernelGridIsOne<<<1, block_size, 0, stream>>>(
        x_data, y_data, lod_data, lod_count, row_numel);
  } else {
    SequenceReverseKernel<<<grid_size, block_size, 0, stream>>>(
        x_data, y_data, lod_data, lod_count, row_numel, limit);
  }
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::cuda::SequenceReverseCompute<float,
                                                            PRECISION(kFloat)>
    ReverseFp32;

typedef paddle::lite::kernels::cuda::SequenceReverseCompute<int64_t,
                                                            PRECISION(kInt64)>
    ReverseInt64;

REGISTER_LITE_KERNEL(sequence_reverse, kCUDA, kFloat, kNCHW, ReverseFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(sequence_reverse, kCUDA, kInt64, kNCHW, ReverseInt64, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt64))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt64))})
    .Finalize();
