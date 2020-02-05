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
#include "lite/kernels/cuda/pool_compute.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {
using Tensor = lite::Tensor;
using DDim = lite::DDim;

#define MAX_VAL(a, b) (((a) > (b)) ? (a) : (b))
#define MIN_VAL(a, b) (((a) < (b)) ? (a) : (b))

__global__ void max_pool_kernel(const float* input,
                                float* output,
                                const int spatial_in,
                                const int spatial_out,
                                const int in_h,
                                const int in_w,
                                const int out_h,
                                const int out_w,
                                const int pad_h,
                                const int pad_w,
                                const int win_h,
                                const int win_w,
                                const int stride_h,
                                const int stride_w,
                                const int total_threads) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < total_threads) {
    const int nc_id = gid / spatial_out;
    const int w_id = gid % spatial_out % out_w;
    const int h_id = gid % spatial_out / out_w;
    const int w_s = w_id * stride_w - pad_w;
    const int iw_s = MAX_VAL(w_s, 0);
    const int iw_e = MIN_VAL(w_s + win_w, in_w);
    const int w_loop = iw_e - iw_s;
    const int h_s = h_id * stride_h - pad_h;
    const int ih_s = MAX_VAL(h_s, 0);
    const int ih_e = MIN_VAL(h_s + win_h, in_h);
    const int h_loop = ih_e - ih_s;
    const float* in_p = input + nc_id * spatial_in + ih_s * in_w + iw_s;
    float max_val = -FLT_MAX;
    for (int i = 0; i < h_loop; ++i) {
      for (int j = 0; j < w_loop; ++j) {
        max_val = MAX_VAL(max_val, *(in_p + j));
      }
      in_p += in_w;
    }
    max_val = max_val == -FLT_MAX ? 0.f : max_val;
    output[nc_id * spatial_out + h_id * out_w + w_id] = max_val;
  }
}

__global__ void adaptive_max_pool_kernel(const float* input,
                                         float* output,
                                         const int spatial_in,
                                         const int spatial_out,
                                         const int in_h,
                                         const int in_w,
                                         const int out_h,
                                         const int out_w,
                                         const int pad_h,
                                         const int pad_w,
                                         const int win_h,
                                         const int win_w,
                                         const int stride_h,
                                         const int stride_w,
                                         const int total_threads) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < total_threads) {
    const int nc_id = gid / spatial_out;
    const int w_id = gid % spatial_out % out_w;
    const int h_id = gid % spatial_out / out_w;
    const int iw_s = floor(static_cast<double>(w_id * in_w) / out_w);
    const int iw_e = ceil(static_cast<double>((w_id + 1) * in_w) / out_w);
    const int w_loop = iw_e - iw_s;
    const int ih_s = floor(static_cast<double>(h_id * in_h) / out_h);
    const int ih_e = ceil(static_cast<double>((h_id + 1) * in_h) / out_h);
    const int h_loop = ih_e - ih_s;
    const float* in_p = input + nc_id * spatial_in + ih_s * in_w + iw_s;
    float max_val = -FLT_MAX;
    for (int i = 0; i < h_loop; ++i) {
      for (int j = 0; j < w_loop; ++j) {
        max_val = MAX_VAL(max_val, *(in_p + j));
      }
      in_p += in_w;
    }
    output[nc_id * spatial_out + h_id * out_w + w_id] = max_val;
  }
}

__global__ void avg_pool_kernel(const float* input,
                                float* output,
                                const int spatial_in,
                                const int spatial_out,
                                const int in_h,
                                const int in_w,
                                const int out_h,
                                const int out_w,
                                const int pad_h,
                                const int pad_w,
                                const int win_h,
                                const int win_w,
                                const int stride_h,
                                const int stride_w,
                                bool exclusive,
                                const int total_threads) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < total_threads) {
    const int nc_id = gid / spatial_out;
    const int w_id = gid % spatial_out % out_w;
    const int h_id = gid % spatial_out / out_w;
    const int w_s = w_id * stride_w - pad_w;
    const int iw_s = MAX_VAL(w_s, 0);
    const int iw_e = MIN_VAL(w_s + win_w, in_w);
    const int w_loop = iw_e - iw_s;
    const int h_s = h_id * stride_h - pad_h;
    const int ih_s = MAX_VAL(h_s, 0);
    const int ih_e = MIN_VAL(h_s + win_h, in_h);
    const int h_loop = ih_e - ih_s;
    const float* in_p = input + nc_id * spatial_in + ih_s * in_w + iw_s;
    float sum_val = 0.f;
    for (int i = 0; i < h_loop; ++i) {
      for (int j = 0; j < w_loop; ++j) {
        sum_val += *(in_p + j);
      }
      in_p += in_w;
    }
    int pool_size = exclusive ? h_loop * w_loop : win_w * win_h;
    pool_size = pool_size == 0 ? 1 : pool_size;
    output[nc_id * spatial_out + h_id * out_w + w_id] = sum_val / pool_size;
  }
}

__global__ void adaptive_avg_pool_kernel(const float* input,
                                         float* output,
                                         const int spatial_in,
                                         const int spatial_out,
                                         const int in_h,
                                         const int in_w,
                                         const int out_h,
                                         const int out_w,
                                         const int pad_h,
                                         const int pad_w,
                                         const int win_h,
                                         const int win_w,
                                         const int stride_h,
                                         const int stride_w,
                                         const int total_threads) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < total_threads) {
    const int nc_id = gid / spatial_out;
    const int w_id = gid % spatial_out % out_w;
    const int h_id = gid % spatial_out / out_w;
    const int iw_s = floor(static_cast<double>(w_id * in_w) / out_w);
    const int iw_e = ceil(static_cast<double>((w_id + 1) * in_w) / out_w);
    const int w_loop = iw_e - iw_s;
    const int ih_s = floor(static_cast<double>(h_id * in_h) / out_h);
    const int ih_e = ceil(static_cast<double>((h_id + 1) * in_h) / out_h);
    const int h_loop = ih_e - ih_s;
    const float* in_p = input + nc_id * spatial_in + ih_s * in_w + iw_s;
    float sum_val = 0.f;
    for (int i = 0; i < h_loop; ++i) {
      for (int j = 0; j < w_loop; ++j) {
        sum_val += *(in_p + j);
      }
      in_p += in_w;
    }
    int pool_size = h_loop * w_loop;
    pool_size = pool_size == 0 ? 1 : pool_size;
    output[nc_id * spatial_out + h_id * out_w + w_id] = sum_val / pool_size;
  }
}

__global__ void global_max_pool_kernel(const float* input,
                                       float* output,
                                       const int in_h,
                                       const int in_w,
                                       const int total_threads) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < total_threads) {
    const int spatial_in = in_h * in_w;
    const float* in_p = input + gid * spatial_in;
    int i = 0;
    float max_val = -0.f;
    // unroll 8
    for (; i < spatial_in - 7; i += 8) {
      max_val = MAX_VAL(max_val, *(in_p + 0));
      max_val = MAX_VAL(max_val, *(in_p + 1));
      max_val = MAX_VAL(max_val, *(in_p + 2));
      max_val = MAX_VAL(max_val, *(in_p + 3));
      max_val = MAX_VAL(max_val, *(in_p + 4));
      max_val = MAX_VAL(max_val, *(in_p + 5));
      max_val = MAX_VAL(max_val, *(in_p + 6));
      max_val = MAX_VAL(max_val, *(in_p + 7));
      in_p += 8;
    }
    for (; i < spatial_in; i++) {
      max_val = MAX_VAL(max_val, *in_p);
      in_p++;
    }
    output[gid] = max_val;
  }
}

__global__ void global_avg_pool_kernel(const float* input,
                                       float* output,
                                       const int in_h,
                                       const int in_w,
                                       const int total_threads) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < total_threads) {
    const int spatial_in = in_h * in_w;
    const float* in_p = input + gid * spatial_in;
    int i = 0;
    float sum_val = 0.f;
    // unroll 8
    for (; i < spatial_in - 7; i += 8) {
      sum_val += *in_p++;
      sum_val += *in_p++;
      sum_val += *in_p++;
      sum_val += *in_p++;
      sum_val += *in_p++;
      sum_val += *in_p++;
      sum_val += *in_p++;
      sum_val += *in_p++;
    }
    for (; i < spatial_in; i++) {
      sum_val += *in_p++;
    }
    output[gid] = sum_val / spatial_in;
  }
}

void PoolCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  bool exclusive = param.exclusive;
  bool adaptive = param.adaptive;
  auto x_dims = param.x->dims();
  auto out_dims = param.output->dims();
  auto paddings = *param.paddings;
  const int in_h = x_dims[2];
  const int in_w = x_dims[3];
  const int out_h = out_dims[2];
  const int out_w = out_dims[3];
  const int spatial_in = in_h * in_w;
  const int spatial_out = out_h * out_w;
  const int win_h = param.ksize[0];
  const int win_w = param.ksize[1];
  const int stride_h = param.strides[0];
  const int stride_w = param.strides[1];
  const int pad_h = paddings[0];
  const int pad_w = paddings[2];
  const int total_threads = out_dims.production();
  const int threads = 512;
  const int blocks = (total_threads + threads - 1) / threads;
  auto input_data = param.x->data<float>();
  auto output_data = param.output->mutable_data<float>(TARGET(kCUDA));
  if (param.global_pooling) {
    if (param.pooling_type == "max") {
      global_max_pool_kernel<<<blocks, threads, 0, stream>>>(
          input_data, output_data, in_h, in_w, total_threads);
    } else {
      global_avg_pool_kernel<<<blocks, threads, 0, stream>>>(
          input_data, output_data, in_h, in_w, total_threads);
    }
  } else {
    if (!adaptive) {
      if (param.pooling_type == "max") {
        max_pool_kernel<<<blocks, threads, 0, stream>>>(input_data,
                                                        output_data,
                                                        spatial_in,
                                                        spatial_out,
                                                        in_h,
                                                        in_w,
                                                        out_h,
                                                        out_w,
                                                        pad_h,
                                                        pad_w,
                                                        win_h,
                                                        win_w,
                                                        stride_h,
                                                        stride_w,
                                                        total_threads);
      } else {
        avg_pool_kernel<<<blocks, threads, 0, stream>>>(input_data,
                                                        output_data,
                                                        spatial_in,
                                                        spatial_out,
                                                        in_h,
                                                        in_w,
                                                        out_h,
                                                        out_w,
                                                        pad_h,
                                                        pad_w,
                                                        win_h,
                                                        win_w,
                                                        stride_h,
                                                        stride_w,
                                                        exclusive,
                                                        total_threads);
      }
    } else {
      if (param.pooling_type == "max") {
        adaptive_max_pool_kernel<<<blocks, threads, 0, stream>>>(input_data,
                                                                 output_data,
                                                                 spatial_in,
                                                                 spatial_out,
                                                                 in_h,
                                                                 in_w,
                                                                 out_h,
                                                                 out_w,
                                                                 pad_h,
                                                                 pad_w,
                                                                 win_h,
                                                                 win_w,
                                                                 stride_h,
                                                                 stride_w,
                                                                 total_threads);
      } else {
        adaptive_avg_pool_kernel<<<blocks, threads, 0, stream>>>(input_data,
                                                                 output_data,
                                                                 spatial_in,
                                                                 spatial_out,
                                                                 in_h,
                                                                 in_w,
                                                                 out_h,
                                                                 out_w,
                                                                 pad_h,
                                                                 pad_w,
                                                                 win_h,
                                                                 win_w,
                                                                 stride_h,
                                                                 stride_w,
                                                                 total_threads);
      }
    }
  }
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(FATAL) << cudaGetErrorString(error);
}

inline int PoolOutputSize(
    int input_size, int filter_size, int padding, int stride, bool ceil_mode) {
  int output_size;
  if (!ceil_mode) {
    output_size = (input_size - filter_size + 2 * padding) / stride + 1;
  } else {
    output_size =
        (input_size - filter_size + 2 * padding + stride - 1) / stride + 1;
  }
  return output_size;
}

void PoolComputeNHWC::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  pool_impl_.reset(new lite::cuda::math::CudnnPool2DNHWC<PRECISION(kFloat)>);
  pool_impl_->init(param, &ctx);
}

void PoolComputeNHWC::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();
  const auto x_dims = param.x->dims();
  std::vector<int>& ksize = param.ksize;
  if (param.global_pooling) {
    ksize.resize(static_cast<size_t>(x_dims.size()) - 2);
    for (size_t i = 0; i < ksize.size(); ++i) {
      (*param.paddings)[i] = 0;
      ksize[i] = static_cast<int>(x_dims[i + 1]);
    }
  }

  std::vector<int64_t> output_shape({x_dims[0]});
  if (param.adaptive) {
    output_shape.insert(
        output_shape.end(), param.ksize.begin(), param.ksize.end());
  } else {
    for (size_t i = 0; i < param.ksize.size(); ++i) {
      output_shape.push_back(PoolOutputSize(x_dims[i + 1],
                                            param.ksize[i],
                                            (*param.paddings)[i],
                                            param.strides[i],
                                            param.ceil_mode));
    }
  }
  output_shape.push_back(x_dims[3]);
  param.output->Resize(lite::DDim(output_shape));

  pool_impl_->run(param);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(FATAL) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    pool2d, kCUDA, kFloat, kNCHW, paddle::lite::kernels::cuda::PoolCompute, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(pool2d,
                     kCUDA,
                     kFloat,
                     kNHWC,
                     paddle::lite::kernels::cuda::PoolComputeNHWC,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
