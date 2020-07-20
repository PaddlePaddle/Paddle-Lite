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
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/backends/cuda/math/activation.h"
#include "lite/backends/cuda/math/utils.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

ActivationType GetActiveType(const std::string& act) {
  if (act == "sigmoid") {
    return kSigmoid;
  } else if (act == "relu") {
    return kReLU;
  } else if (act == "tanh") {
    return kTanh;
  } else if (act == "identify") {
    return kIdentity;
  } else {
    LOG(FATAL) << "not supported activation: " << act;
  }
}

template <typename T>
__global__ void relu_kernel(const int num,
                            const float alpha,
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

template <>
__global__ void relu_kernel<half>(const int num,
                                  const float alpha,
                                  const half* input,
                                  half* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num) {
    const half kZero = __float2half(0.0f);
#if __CUDA_ARCH__ >= 530
    output[index] = __hgt(__ldg(input + index), kZero)
                        ? __ldg(input + index)
                        : __hmul(__ldg(input + index), __float2half(alpha));
#else
    output[index] = (__half2float(input[index]) > 0)
                        ? input[index]
                        : __float2half(__half2float(input[index]) * alpha);
#endif
  }
}

template <typename T>
__global__ void bias_relu_kernel(const int num,
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

template <typename Dtype>
__global__ void bias_relu_int8_nhwc_kernel(int num,
                                           const float* in,
                                           const float* bias,
                                           Dtype* out,
                                           int N,
                                           int C,
                                           int H,
                                           int W,
                                           const float* scale,
                                           float alpha) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num) {
    int idx = tid % C;
#if __CUDA_ARCH__ >= 350
    float temp = __ldg(in + tid) * __ldg(scale + idx) + __ldg(bias + idx);
    out[tid] =
        temp > 0 ? from_float<Dtype>(temp) : from_float<Dtype>(temp * alpha);
#else
    float temp = in[tid] * scale[idx] + bias[idx];
    out[tid] =
        temp > 0 ? from_float<Dtype>(temp) : from_float<Dtype>(temp * alpha);
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

template <typename Dtype>
__global__ void bias_int8_nhwc_kernel(int num,
                                      const float* in,
                                      const float* bias,
                                      Dtype* out,
                                      int N,
                                      int C,
                                      int H,
                                      int W,
                                      const float* scale) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num) {
    int idx = tid % C;
#if __CUDA_ARCH__ >= 350
    float temp = __ldg(in + tid) * __ldg(scale + idx) + __ldg(bias + idx);
    out[tid] = from_float<Dtype>(temp);
#else
    float temp = in[tid] * scale[idx] + bias[idx];
    out[tid] = from_float<Dtype>(temp);
#endif
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
void bias_relu_int8_nhwc<float>(int num,
                                const void* in,
                                const void* bias,
                                void* out,
                                int N,
                                int C,
                                int H,
                                int W,
                                const void* scale,
                                float alpha,
                                cudaStream_t stream) {
  int thread = 256;
  if (C % 4 == 0) {
    int block = (num / 4 + thread - 1) / thread;
    bias_relu_int8_nhwc4_kernel<<<block, thread, 0, stream>>>(
        num / 4,
        static_cast<const float4*>(in),
        static_cast<const float4*>(bias),
        static_cast<float4*>(out),
        N,
        C / 4,
        H,
        W,
        static_cast<const float4*>(scale),
        alpha);
  } else {
    int block = (num + thread - 1) / thread;
    bias_relu_int8_nhwc_kernel<<<block, thread, 0, stream>>>(
        num,
        static_cast<const float*>(in),
        static_cast<const float*>(bias),
        static_cast<float*>(out),
        N,
        C,
        H,
        W,
        static_cast<const float*>(scale),
        alpha);
  }
}

template <>
void bias_relu_int8_nhwc<int8_t>(int num,
                                 const void* in,
                                 const void* bias,
                                 void* out,
                                 int N,
                                 int C,
                                 int H,
                                 int W,
                                 const void* scale,
                                 float alpha,
                                 cudaStream_t stream) {
  int thread = 256;
  if (C % 4 == 0) {
    int block = (num / 4 + thread - 1) / thread;
    bias_relu_int8_nhwc4_kernel<<<block, thread, 0, stream>>>(
        num / 4,
        static_cast<const float4*>(in),
        static_cast<const float4*>(bias),
        static_cast<char4*>(out),
        N,
        C / 4,
        H,
        W,
        static_cast<const float4*>(scale),
        alpha);
  } else {
    int block = (num + thread - 1) / thread;
    bias_relu_int8_nhwc_kernel<<<block, thread, 0, stream>>>(
        num,
        static_cast<const float*>(in),
        static_cast<const float*>(bias),
        static_cast<int8_t*>(out),
        N,
        C,
        H,
        W,
        static_cast<const float*>(scale),
        alpha);
  }
}

template <typename out_type>
void bias_int8_nhwc(int num,
                    const void* in,
                    const void* bias,
                    void* out,
                    int N,
                    int C,
                    int H,
                    int W,
                    const void* scale,
                    cudaStream_t stream) {
  int thread = 256;
  int block = (num + thread - 1) / thread;
  bias_int8_nhwc_kernel<<<block, thread, 0, stream>>>(
      num,
      static_cast<const float*>(in),
      static_cast<const float*>(bias),
      static_cast<out_type*>(out),
      N,
      C,
      H,
      W,
      static_cast<const float*>(scale));
}

template void bias_int8_nhwc<float>(int,
                                    const void*,
                                    const void* bias,
                                    void*,
                                    int,
                                    int,
                                    int,
                                    int,
                                    const void*,
                                    cudaStream_t);
template void bias_int8_nhwc<int8_t>(int,
                                     const void*,
                                     const void* bias,
                                     void*,
                                     int,
                                     int,
                                     int,
                                     int,
                                     const void*,
                                     cudaStream_t);

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

template <>
void relu<half>(
    int num, const half* din, half* dout, float alpha, cudaStream_t stream) {
  if (num == 0) {
    return;
  }
  int thread = 256;
  int block = (num + thread - 1) / thread;
  relu_kernel<half><<<block, thread, 0, stream>>>(num, alpha, din, dout);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) std::cout << cudaGetErrorString(error);
}

template <typename T>
void bias_relu(int num,
               const T* din,
               const float* bias,
               T* dout,
               float alpha,
               cudaStream_t stream) {
  int thread = 256;
  int block = (num + thread - 1) / thread;
  relu_kernel<<<block, thread, 0, stream>>>(num, alpha, din, dout);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) std::cout << cudaGetErrorString(error);
}
template void relu(int, const float*, float*, float, cudaStream_t);
template void relu(int, const half*, half*, float, cudaStream_t);
template void bias_relu(
    int, const float*, const float* bias, float*, float, cudaStream_t);

// ------------- sigmoid -------------

template <typename T>
__global__ void sigmoid_kernel(const int num, const T* in, T* out) {
  CUDA_KERNEL_LOOP(i, num) {
#if __CUDA_ARCH__ >= 350
    out[i] = static_cast<T>(1.0f) /
             (static_cast<T>(1.0f) + expf(-1 * __ldg(in + i)));
#else
    out[i] = static_cast<T>(1.0f) / (static_cast<T>(1.0f) + expf(-in[i]));
#endif
  }
}

template <>
__global__ void sigmoid_kernel(const int num, const half* in, half* out) {
  CUDA_KERNEL_LOOP(i, num) {
    half tmp = __float2half(1.0f);
#if __CUDA_ARCH__ >= 530
    out[i] = __hdiv(
        tmp, __hadd(tmp, hexp(__hmul(__float2half(-1.0f), __ldg(in + i)))));
#else
    out[i] = __float2half(1.0f / (1.0f + expf(-1 * __half2float(in[i]))));
#endif
  }
}

template <>
__global__ void sigmoid_kernel(const int num, const half2* in, half2* out) {
  CUDA_KERNEL_LOOP(i, num) {
    half2 tmp = __floats2half2_rn(1.0f, 1.0f);
#if __CUDA_ARCH__ >= 530
    out[i] = __h2div(tmp,
                     __hadd2(tmp,
                             h2exp(__hmul2(__floats2half2_rn(-1.0f, -1.0f),
                                           __ldg(in + i)))));
#else
    out[i].x = __float2half(1.0f / (1.0f + expf(-1 * __half2float(in[i].x))));
    out[i].y = __float2half(1.0f / (1.0f + expf(-1 * __half2float(in[i].y))));
#endif
  }
}

template <typename T>
void sigmoid(const int num, const T* din, T* dout, cudaStream_t stream) {
  sigmoid_kernel<T><<<CUDA_GET_BLOCKS(num), CUDA_NUM_THREADS, 0, stream>>>(
      num, din, dout);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void sigmoid(const int num, const half* din, half* dout, cudaStream_t stream) {
  if (num % 2 == 0) {
    const half2* din2 = reinterpret_cast<const half2*>(din);
    half2* dout2 = reinterpret_cast<half2*>(dout);
    sigmoid_kernel<
        half2><<<CUDA_GET_BLOCKS(num / 2), CUDA_NUM_THREADS, 0, stream>>>(
        num / 2, din2, dout2);
  } else {
    sigmoid_kernel<half><<<CUDA_GET_BLOCKS(num), CUDA_NUM_THREADS, 0, stream>>>(
        num, din, dout);
  }
  CUDA_POST_KERNEL_CHECK;
}

template void sigmoid(const int num,
                      const float* din,
                      float* dout,
                      cudaStream_t stream);

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
