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

#include <iostream>
#include "lite/backends/cuda/math/activation.h"
#include "lite/backends/cuda/math/utils.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename T>
__global__ void relu_kernel(const int num,
                            const T alpha,
                            const T* input,
                            T* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num) {
#if __CUDA_ARCH__ >= 350
    output[index] = __ldg(input + index) >= 0 ? __ldg(input + index)
                                              : __ldg(input + index) * alpha;
#else
    output[index] = input[index] >= 0 ? input[index] : input[index] * alpha;
#endif
  }
}

__global__ void bias_relu_int8_nhwc4_kernel(int num,
                                            const float4* in,
                                            const float4* bias,
                                            float4* out,
                                            int N,
                                            int K,
                                            int H,
                                            int W,
                                            const float4* scale,
                                            float alpha) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num) {
    int bias_idx = tid % K;
    const float4 bias_ptr = bias[bias_idx];
    const float4 scale_ptr = scale[bias_idx];
    const float4 in_ptr = in[tid];

    float4 packed_val;
    packed_val.x = in_ptr.x * scale_ptr.x + bias_ptr.x;
    packed_val.x = fmaxf(packed_val.x * alpha, packed_val.x);
    packed_val.y = in_ptr.y * scale_ptr.y + bias_ptr.y;
    packed_val.y = fmaxf(packed_val.y * alpha, packed_val.y);
    packed_val.z = in_ptr.z * scale_ptr.z + bias_ptr.z;
    packed_val.z = fmaxf(packed_val.z * alpha, packed_val.z);
    packed_val.w = in_ptr.w * scale_ptr.w + bias_ptr.w;
    packed_val.w = fmaxf(packed_val.w * alpha, packed_val.w);
    out[tid] = packed_val;
  }
}

__global__ void bias_relu_int8_nhwc4_kernel(int num,
                                            const float4* in,
                                            const float4* bias,
                                            char4* out,
                                            int N,
                                            int K,
                                            int H,
                                            int W,
                                            const float4* scale,
                                            float alpha) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num) {
    int bias_idx = tid % K;
    const float4 bias_ptr = bias[bias_idx];
    const float4 scale_ptr = scale[bias_idx];
    const float4 in_ptr = in[tid];

    float4 packed_val;
    char4 result_val;
    packed_val.x = in_ptr.x * scale_ptr.x + bias_ptr.x;
    result_val.x =
        from_float<int8_t>(fmaxf(packed_val.x * alpha, packed_val.x));
    packed_val.y = in_ptr.y * scale_ptr.y + bias_ptr.y;
    result_val.y =
        from_float<int8_t>(fmaxf(packed_val.y * alpha, packed_val.y));
    packed_val.z = in_ptr.z * scale_ptr.z + bias_ptr.z;
    result_val.z =
        from_float<int8_t>(fmaxf(packed_val.z * alpha, packed_val.z));
    packed_val.w = in_ptr.w * scale_ptr.w + bias_ptr.w;
    result_val.w =
        from_float<int8_t>(fmaxf(packed_val.w * alpha, packed_val.w));

    out[tid] = result_val;
  }
}

__global__ void relu_int8_nhwc4_kernel(int num,
                                       const float4* in,
                                       float4* out,
                                       int N,
                                       int K,
                                       int H,
                                       int W,
                                       const float4* scale,
                                       float alpha) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num) {
    int scale_idx = tid % K;
    const float4 scale_ptr = scale[scale_idx];
    const float4 in_ptr = in[tid];

    float4 packed_val;
    packed_val.x = in_ptr.x * scale_ptr.x;
    packed_val.x = fmaxf(packed_val.x * alpha, packed_val.x);
    packed_val.y = in_ptr.y * scale_ptr.y;
    packed_val.y = fmaxf(packed_val.y * alpha, packed_val.y);
    packed_val.z = in_ptr.z * scale_ptr.z;
    packed_val.z = fmaxf(packed_val.z * alpha, packed_val.z);
    packed_val.w = in_ptr.w * scale_ptr.w;
    packed_val.w = fmaxf(packed_val.w * alpha, packed_val.w);
    out[tid] = packed_val;
  }
}

__global__ void relu_int8_nhwc4_kernel(int num,
                                       const float4* in,
                                       char4* out,
                                       int N,
                                       int K,
                                       int H,
                                       int W,
                                       const float4* scale,
                                       float alpha) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num) {
    int scale_idx = tid % K;
    const float4 scale_ptr = scale[scale_idx];
    const float4 in_ptr = in[tid];

    float4 packed_val;
    char4 result_val;
    packed_val.x = in_ptr.x * scale_ptr.x;
    result_val.x =
        from_float<int8_t>(fmaxf(packed_val.x * alpha, packed_val.x));
    packed_val.y = in_ptr.y * scale_ptr.y;
    result_val.y =
        from_float<int8_t>(fmaxf(packed_val.y * alpha, packed_val.y));
    packed_val.z = in_ptr.z * scale_ptr.z;
    result_val.z =
        from_float<int8_t>(fmaxf(packed_val.z * alpha, packed_val.z));
    packed_val.w = in_ptr.w * scale_ptr.w;
    result_val.w =
        from_float<int8_t>(fmaxf(packed_val.w * alpha, packed_val.w));

    out[tid] = result_val;
  }
}

template <>
void bias_relu_int8_nhwc4<float>(int num,
                                 const void* in,
                                 const void* bias,
                                 void* out,
                                 int N,
                                 int K,
                                 int H,
                                 int W,
                                 const void* scale,
                                 float alpha,
                                 cudaStream_t stream) {
  int thread = 256;
  int block = (num + thread - 1) / thread;
  bias_relu_int8_nhwc4_kernel<<<block, thread, 0, stream>>>(
      num,
      static_cast<const float4*>(in),
      static_cast<const float4*>(bias),
      static_cast<float4*>(out),
      N,
      K,
      H,
      W,
      static_cast<const float4*>(scale),
      alpha);
}

template <>
void bias_relu_int8_nhwc4<int8_t>(int num,
                                  const void* in,
                                  const void* bias,
                                  void* out,
                                  int N,
                                  int K,
                                  int H,
                                  int W,
                                  const void* scale,
                                  float alpha,
                                  cudaStream_t stream) {
  int thread = 256;
  int block = (num + thread - 1) / thread;
  bias_relu_int8_nhwc4_kernel<<<block, thread, 0, stream>>>(
      num,
      static_cast<const float4*>(in),
      static_cast<const float4*>(bias),
      static_cast<char4*>(out),
      N,
      K,
      H,
      W,
      static_cast<const float4*>(scale),
      alpha);
}

template <>
void relu_int8_nhwc4<float>(int num,
                            const void* in,
                            void* out,
                            int N,
                            int K,
                            int H,
                            int W,
                            const void* scale,
                            float alpha,
                            cudaStream_t stream) {
  int thread = 256;
  int block = (num + thread - 1) / thread;
  relu_int8_nhwc4_kernel<<<block, thread, 0, stream>>>(
      num,
      static_cast<const float4*>(in),
      static_cast<float4*>(out),
      N,
      K,
      H,
      W,
      static_cast<const float4*>(scale),
      alpha);
}

template <>
void relu_int8_nhwc4<int8_t>(int num,
                             const void* in,
                             void* out,
                             int N,
                             int K,
                             int H,
                             int W,
                             const void* scale,
                             float alpha,
                             cudaStream_t stream) {
  int thread = 256;
  int block = (num + thread - 1) / thread;
  relu_int8_nhwc4_kernel<<<block, thread, 0, stream>>>(
      num,
      static_cast<const float4*>(in),
      static_cast<char4*>(out),
      N,
      K,
      H,
      W,
      static_cast<const float4*>(scale),
      alpha);
}

template <typename T>
void relu(int num, const T* din, T* dout, float alpha, cudaStream_t stream) {
  int thread = 256;
  int block = (num + thread - 1) / thread;
  relu_kernel<<<block, thread, 0, stream>>>(num, alpha, din, dout);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) std::cout << cudaGetErrorString(error);
}
template void relu(int, const float*, float*, float, cudaStream_t);

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
