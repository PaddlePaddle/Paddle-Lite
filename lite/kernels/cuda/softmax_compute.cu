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
#include <limits>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/softmax_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {
using Tensor = lite::Tensor;

extern __shared__ char tile[];
template <typename dtype>
__global__ void sharemem_softmax_kernel(int total_size,
                                        const dtype* in_data,
                                        dtype* out_data,
                                        int inner_num,
                                        int outer_num,
                                        int axis_size) {
  dtype* data = reinterpret_cast<dtype*>(tile) + threadIdx.x;
  //! compute thread index and real data index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_size) {
    int idx_inner = idx % inner_num;
    int idx_outer = (idx / inner_num) * axis_size;
    int blocksize = blockDim.x;
    int real_index = idx_outer * inner_num + idx_inner;
    int loop_idx = real_index;
//! read all data to sharemem in softmax channel
#pragma unroll
    for (int i = 0; i < axis_size; ++i) {
      data[i * blocksize] = in_data[loop_idx];
      loop_idx += inner_num;
    }
    //! get maximum value in softmax channel
    dtype max_data = data[0];
#pragma unroll
    for (int i = 1; i < axis_size; ++i) {
      dtype dt = data[i * blocksize];
      if (max_data < dt) {
        max_data = dt;
      }
    }
    //! subtract then summarize
    dtype sum = 0;
#pragma unroll
    for (int i = 0; i < axis_size; ++i) {
      dtype* dt = data + i * blocksize;
      *dt = expf(*dt - max_data);
      sum += *dt;
    }
    //! write back result
    loop_idx = real_index;
#pragma unroll
    for (int i = 0; i < axis_size; ++i) {
      out_data[loop_idx] = data[i * blocksize] / sum;
      loop_idx += inner_num;
    }
  }
}

//! general kernel for softmax
template <typename dtype>
__global__ void softmax_max_kernel(int total_size,
                                   const dtype* in_data,
                                   dtype* out_data,
                                   dtype min_data,
                                   int inner_num,
                                   int outer_num,
                                   int axis_size) {
  //! compute data index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_size) {
    int idx_inner = idx % inner_num;
    int idx_outer = (idx / inner_num) * axis_size;
    int real_index = idx_outer * inner_num + idx_inner;
    //! get maximum data across softmax axis
    dtype max_data = min_data;
    for (int i = 0; i < axis_size; ++i) {
      max_data =
          in_data[real_index] > max_data ? in_data[real_index] : max_data;
      real_index += inner_num;
    }
    out_data[idx] = max_data;
  }
}

template <typename dtype>
__global__ void softmax_sub_exp_sum_kernel(int total_size,
                                           const dtype* in_data,
                                           dtype* out_data,
                                           const dtype* max_data,
                                           dtype* sum_data,
                                           int inner_num,
                                           int outer_num,
                                           int axis_size) {
  //! compute data index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_size) {
    int idx_inner = idx % inner_num;
    int idx_outer = (idx / inner_num) * axis_size;

    dtype max_data_cur = max_data[idx];
    dtype sum_data_cur = 0;
    int real_index = idx_outer * inner_num + idx_inner;
    //! compute exp and summarize across the softmax axis
    for (int i = 0; i < axis_size; ++i) {
      dtype sub_data = in_data[real_index] - max_data_cur;
      sub_data = expf(sub_data);
      sum_data_cur += sub_data;
      out_data[real_index] = sub_data;
      real_index += inner_num;
    }
    sum_data[idx] = sum_data_cur;
  }
}

template <typename dtype>
__global__ void softmax_divid_output_kernel(int total_size,
                                            dtype* io_data,
                                            const dtype* sum_data,
                                            int inner_num,
                                            int outer_num,
                                            int axis_size) {
  //! compute data index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_size) {
    int idx_inner = idx % inner_num;
    int idx_outer = (idx / inner_num) * axis_size;
    dtype sum_data_cur = 1.f / sum_data[idx];
    int real_index = idx_outer * inner_num + idx_inner;
    //! compute final result
    for (int i = 0; i < axis_size; ++i) {
      io_data[real_index] = io_data[real_index] * sum_data_cur;
      real_index += inner_num;
    }
  }
}

void SoftmaxCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  auto x_dims = param.x->dims();
  auto x_rank = x_dims.size();
  int axis = param.axis;
  if (axis < 0) {
    axis += x_rank;
  }
  int outer_num = x_dims.Slice(0, axis).production();
  int inner_num = x_dims.Slice(axis + 1, x_rank).production();
  int total_threads = inner_num * outer_num;
  int axis_size = x_dims[axis];

  int device_id;
  const int threads = 512;
  const int blocks = (total_threads + threads - 1) / threads;
  cudaGetDevice(&device_id);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  size_t sharedmem_size = deviceProp.sharedMemPerBlock;
  int max_dimsize = sharedmem_size / sizeof(float) / threads;
  auto input_data = param.x->data<float>();
  auto output_data = param.output->mutable_data<float>(TARGET(kCUDA));
  TargetWrapperCuda::MemsetSync(
      output_data, 0, param.output->numel() * sizeof(float));
  if (axis_size <= max_dimsize) {
    int use_sharemem_size = axis_size * threads * sizeof(float);
    sharemem_softmax_kernel<<<blocks, threads, use_sharemem_size, stream>>>(
        total_threads,
        input_data,
        output_data,
        inner_num,
        outer_num,
        axis_size);
  } else {
    //! re_alloc device memory
    Tensor tmax_data;
    Tensor tsum_data;
    tmax_data.Resize({1, 1, 1, outer_num * inner_num});
    tsum_data.Resize({1, 1, 1, outer_num * inner_num});
    auto max_data = tmax_data.mutable_data<float>(TARGET(kCUDA));
    auto sum_data = tsum_data.mutable_data<float>(TARGET(kCUDA));
    //! firstly, get maximum data
    float min_data = std::numeric_limits<float>::lowest();
    softmax_max_kernel<float><<<blocks, threads, 0, stream>>>(total_threads,
                                                              input_data,
                                                              max_data,
                                                              min_data,
                                                              inner_num,
                                                              outer_num,
                                                              axis_size);
    //! then, compute exp and sum data
    softmax_sub_exp_sum_kernel<float><<<blocks, threads, 0, stream>>>(
        total_threads,
        input_data,
        output_data,
        max_data,
        sum_data,
        inner_num,
        outer_num,
        axis_size);
    //! last, compute divided output
    softmax_divid_output_kernel<float><<<blocks, threads, 0, stream>>>(
        total_threads, output_data, sum_data, inner_num, outer_num, axis_size);
  }
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(ERROR) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(softmax,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::SoftmaxCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("axis",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
REGISTER_LITE_KERNEL(search_seq_softmax,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::SoftmaxCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("Out_log", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
