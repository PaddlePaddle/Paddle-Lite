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

#include "lite/backends/cuda/cuda_utils.h"
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

template <>
__global__ void sharemem_softmax_kernel(int total_size,
                                        const half* in_data,
                                        half* out_data,
                                        int inner_num,
                                        int outer_num,
                                        int axis_size) {
  half* data = reinterpret_cast<half*>(tile) + threadIdx.x;
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
    half max_data = data[0];
#pragma unroll
    for (int i = 1; i < axis_size; ++i) {
      half dt = data[i * blocksize];
#if __CUDA_ARCH__ >= 530
      if (__hlt(max_data, dt)) {
#else
      if (__half2float(max_data) < __half2float(dt)) {
#endif
        max_data = dt;
      }
    }
    //! subtract then summarize
    half sum = 0;
#pragma unroll
    for (int i = 0; i < axis_size; ++i) {
      half* dt = data + i * blocksize;
#if __CUDA_ARCH__ >= 530
      *dt = hexp(__hsub(*dt, max_data));
      sum = __hadd(sum, *dt);
#else
      *dt = __float2half(expf(__half2float(*dt) - __half2float(max_data)));
      sum = __float2half(__half2float(sum) + __half2float(*dt));
#endif
    }
    //! write back result
    loop_idx = real_index;
#pragma unroll
    for (int i = 0; i < axis_size; ++i) {
#if __CUDA_ARCH__ >= 530
      out_data[loop_idx] = __hdiv(data[i * blocksize], sum);
#else
      out_data[loop_idx] =
          __float2half(__half2float(data[i * blocksize]) / __half2float(sum));
#endif
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

template <>
__global__ void softmax_max_kernel(int total_size,
                                   const half* in_data,
                                   half* out_data,
                                   half min_data,
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
    half max_data = min_data;
    for (int i = 0; i < axis_size; ++i) {
#if __CUDA_ARCH__ >= 530
      max_data =
          __hgt(in_data[real_index], max_data) ? in_data[real_index] : max_data;
#else
      float a = __half2float(in_data[real_index]);
      float b = __half2float(max_data);
      float res = a > b ? a : b;
      max_data = __float2half(res);
#endif
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

template <>
__global__ void softmax_sub_exp_sum_kernel(int total_size,
                                           const half* in_data,
                                           half* out_data,
                                           const half* max_data,
                                           half* sum_data,
                                           int inner_num,
                                           int outer_num,
                                           int axis_size) {
  //! compute data index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_size) {
    int idx_inner = idx % inner_num;
    int idx_outer = (idx / inner_num) * axis_size;

    half max_data_cur = max_data[idx];
    half sum_data_cur = 0;
    int real_index = idx_outer * inner_num + idx_inner;
    //! compute exp and summarize across the softmax axis
    for (int i = 0; i < axis_size; ++i) {
#if __CUDA_ARCH__ >= 530
      half sub_data = __hsub(in_data[real_index], max_data_cur);
      sub_data = hexp(sub_data);
      sum_data_cur = __hadd(sum_data_cur, sub_data);
#else
      half sub_data = __float2half(__half2float(in_data[real_index]) -
                                   __half2float(max_data_cur));
      sub_data = __float2half(expf(__half2float(sub_data)));
      sum_data_cur =
          __float2half(__half2float(sum_data_cur) + __half2float(sub_data));
#endif
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

template <>
__global__ void softmax_divid_output_kernel(int total_size,
                                            half* io_data,
                                            const half* sum_data,
                                            int inner_num,
                                            int outer_num,
                                            int axis_size) {
  //! compute data index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_size) {
    int idx_inner = idx % inner_num;
    int idx_outer = (idx / inner_num) * axis_size;
#if __CUDA_ARCH__ >= 530
    half sum_data_cur = __hdiv(__float2half(1.f), sum_data[idx]);
#else
    half sum_data_cur = __float2half(1.f / __half2float(sum_data[idx]));
#endif
    int real_index = idx_outer * inner_num + idx_inner;
    //! compute final result
    for (int i = 0; i < axis_size; ++i) {
#if __CUDA_ARCH__ >= 530
      io_data[real_index] = __hmul(io_data[real_index], sum_data_cur);
#else
      io_data[real_index] = __float2half(__half2float(io_data[real_index]) *
                                         __half2float(sum_data_cur));
#endif
      real_index += inner_num;
    }
  }
}

template <typename Dtype, PrecisionType Ptype>
void SoftmaxCompute<Dtype, Ptype>::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  sharedmem_size_ = deviceProp.sharedMemPerBlock;
  max_dimsize_ = sharedmem_size_ / sizeof(float) / CUDA_NUM_THREADS;
  if (param.use_cudnn) {
    cudnn_softmax_.Init(param, &ctx);
  }
}

template <typename Dtype, PrecisionType Ptype>
void SoftmaxCompute<Dtype, Ptype>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();
  if (param.use_cudnn) {
    cudnn_softmax_.Create(param, &ctx);
    cudnn_softmax_.Run(param);
  } else {
    auto x_dims = param.x->dims();
    auto x_rank = x_dims.size();
    int axis = param.axis;
    if (axis < 0) {
      axis += x_rank;
    }
    int outer_num = x_dims.Slice(0, axis).production();
    int inner_num = x_dims.Slice(axis + 1, x_rank).production();
    int total_threads = inner_num * outer_num;
    axis_size_ = x_dims[axis];

    const int threads = CUDA_NUM_THREADS;
    const int blocks = (total_threads + threads - 1) / threads;
    auto input_data = param.x->template data<Dtype>();
    auto output_data =
        param.output->template mutable_data<Dtype>(TARGET(kCUDA));
    if (axis_size_ <= max_dimsize_) {
      int use_sharemem_size = axis_size_ * threads * sizeof(Dtype);
      sharemem_softmax_kernel<
          Dtype><<<blocks, threads, use_sharemem_size, stream>>>(total_threads,
                                                                 input_data,
                                                                 output_data,
                                                                 inner_num,
                                                                 outer_num,
                                                                 axis_size_);
    } else {
      //! re_alloc device memory
      tmax_data_.Resize({1, 1, 1, outer_num * inner_num});
      tsum_data_.Resize({1, 1, 1, outer_num * inner_num});
      auto max_data = tmax_data_.mutable_data<Dtype>(TARGET(kCUDA));
      auto sum_data = tsum_data_.mutable_data<Dtype>(TARGET(kCUDA));
      //! firstly, get maximum data
      float min_data = std::numeric_limits<float>::lowest();
      softmax_max_kernel<Dtype><<<blocks, threads, 0, stream>>>(total_threads,
                                                                input_data,
                                                                max_data,
                                                                min_data,
                                                                inner_num,
                                                                outer_num,
                                                                axis_size_);
      //! then, compute exp and sum data
      softmax_sub_exp_sum_kernel<Dtype><<<blocks, threads, 0, stream>>>(
          total_threads,
          input_data,
          output_data,
          max_data,
          sum_data,
          inner_num,
          outer_num,
          axis_size_);
      //! last, compute divided output
      softmax_divid_output_kernel<Dtype><<<blocks, threads, 0, stream>>>(
          total_threads,
          output_data,
          sum_data,
          inner_num,
          outer_num,
          axis_size_);
    }
  }
  CUDA_POST_KERNEL_CHECK;
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using SoftmaxFp32 =
    paddle::lite::kernels::cuda::SoftmaxCompute<float, PRECISION(kFloat)>;
using SoftmaxFp16 =
    paddle::lite::kernels::cuda::SoftmaxCompute<half, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(softmax, kCUDA, kFloat, kNCHW, SoftmaxFp32, def)
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
REGISTER_LITE_KERNEL(softmax, kCUDA, kFP16, kNCHW, SoftmaxFp16, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("Out_log",
                {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .Finalize();
REGISTER_LITE_KERNEL(search_seq_softmax, kCUDA, kFloat, kNCHW, SoftmaxFp32, def)
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
REGISTER_LITE_KERNEL(search_seq_softmax, kCUDA, kFP16, kNCHW, SoftmaxFp16, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("Out_log",
                {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .Finalize();
