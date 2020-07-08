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

#include "lite/backends/cuda/math/type_trans.h"
#include "lite/backends/cuda/math/utils.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

__global__ void fp32_to_int8_nhwc_kernel(int num,
                                         const float* in,
                                         int8_t* out,
                                         const float* scale,
                                         int N,
                                         int C,
                                         int H,
                                         int W) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num) {
    int idx = tid % C;
#if __CUDA_ARCH__ >= 350
    out[tid] = from_float<int8_t>(__ldg(in + tid) * __ldg(scale + idx));
#else
    out[tid] = from_float<int8_t>(in[tid] * scale[idx]);
#endif
  }
}

__global__ void fp32_to_int8_nhwc4_kernel(int num,
                                          const float4* in,
                                          char4* out,
                                          const float4* scale,
                                          int N,
                                          int K,
                                          int H,
                                          int W) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num) {
    int scale_idx = tid % K;
    const float4 scale_ptr = scale[scale_idx];
    const float4 in_ptr = in[tid];
    char4 result_val;

    result_val.x = from_float<int8_t>(in_ptr.x * scale_ptr.x);
    result_val.y = from_float<int8_t>(in_ptr.y * scale_ptr.y);
    result_val.z = from_float<int8_t>(in_ptr.z * scale_ptr.z);
    result_val.w = from_float<int8_t>(in_ptr.w * scale_ptr.w);
    out[tid] = result_val;
  }
}

void fp32_to_int8_nhwc(int num,
                       const void* in,
                       void* out,
                       const void* scale,
                       int N,
                       int C,
                       int H,
                       int W,
                       cudaStream_t stream) {
  int thread = 256;
  if (C % 4 == 0) {
    int block = (num / 4 + thread - 1) / thread;
    fp32_to_int8_nhwc4_kernel<<<block, thread, 0, stream>>>(
        num / 4,
        static_cast<const float4*>(in),
        static_cast<char4*>(out),
        static_cast<const float4*>(scale),
        N,
        C / 4,
        H,
        W);
  } else {
    int block = (num + thread - 1) / thread;
    fp32_to_int8_nhwc_kernel<<<block, thread, 0, stream>>>(
        num,
        static_cast<const float*>(in),
        static_cast<int8_t*>(out),
        static_cast<const float*>(scale),
        N,
        C,
        H,
        W);
  }
}

__global__ void Fp32ToFp16Kernel(const int num,
                                 const float* input,
                                 half* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num) {
    output[index] = __float2half(input[index]);
  }
}

void fp32_to_fp16(int num, const float* din, half* dout, cudaStream_t stream) {
  int threads = 1024;
  int blocks = (num + threads - 1) / threads;
  Fp32ToFp16Kernel<<<blocks, threads, 0, stream>>>(num, din, dout);
  cudaError_t error = cudaGetLastError();
  CHECK(error == cudaSuccess) << cudaGetErrorString(error);
}

void fp32_to_fp16(int num, const float* din, half* dout) {
  int threads = 1024;
  int blocks = (num + threads - 1) / threads;
  Fp32ToFp16Kernel<<<blocks, threads>>>(num, din, dout);
  cudaError_t error = cudaGetLastError();
  CHECK(error == cudaSuccess) << cudaGetErrorString(error);
}

__global__ void Fp16ToFp32Kernel(const int num,
                                 const half* input,
                                 float* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num) {
    output[index] = __half2float(input[index]);
  }
}

void fp16_to_fp32(int num, const half* din, float* dout, cudaStream_t stream) {
  int threads = 1024;
  int blocks = (num + threads - 1) / threads;
  Fp16ToFp32Kernel<<<blocks, threads, 0, stream>>>(num, din, dout);
  cudaError_t error = cudaGetLastError();
  CHECK(error == cudaSuccess) << cudaGetErrorString(error);
}

void fp16_to_fp32(int num, const half* din, float* dout) {
  int threads = 1024;
  int blocks = (num + threads - 1) / threads;
  Fp16ToFp32Kernel<<<blocks, threads>>>(num, din, dout);
  cudaError_t error = cudaGetLastError();
  CHECK(error == cudaSuccess) << cudaGetErrorString(error);
}

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
