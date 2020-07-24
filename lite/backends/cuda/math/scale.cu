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

#include "iostream"
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/backends/cuda/math/scale.h"
#include "lite/backends/cuda/math/utils.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename T>
__global__ void scale_kernel(int count,
                             const T* in_data,
                             T* out_data,
                             const T* scale_data,
                             const T* bias_data,
                             const int scale_dim,
                             const int inner_dim) {
  CUDA_KERNEL_LOOP(tid, count) {
    int scale_id = (tid / inner_dim) % scale_dim;
    T scale = scale_data[scale_id];
    if (bias_data == nullptr) {
      out_data[tid] = scale * in_data[tid];
    } else {
      out_data[tid] = scale * in_data[tid] + bias_data[scale_id];
    }
  }
}

template <typename T>
__global__ void scale_kernel(
    int count, const T* in_data, T* out_data, const T scale, const T bias) {
  CUDA_KERNEL_LOOP(tid, count) { out_data[tid] = scale * in_data[tid] + bias; }
}

__global__ void fp32_scale_nhwc4_kernel(int num,
                                        const float4* in,
                                        float4* out,
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
    float4 packed_val;

    packed_val.x = in_ptr.x * scale_ptr.x;
    packed_val.y = in_ptr.y * scale_ptr.y;
    packed_val.z = in_ptr.z * scale_ptr.z;
    packed_val.w = in_ptr.w * scale_ptr.w;
    out[tid] = packed_val;
  }
}

__global__ void fp32_scale_nhwc_kernel(int num,
                                       const float* in,
                                       float* out,
                                       const float* scale,
                                       int N,
                                       int C,
                                       int H,
                                       int W) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num) {
    int idx = tid % C;
#if __CUDA_ARCH__ >= 350
    out[tid] = __ldg(in + tid) * __ldg(scale + idx);
#else
    out[tid] = in[tid] * scale[idx];
#endif
  }
}

void fp32_scale_nhwc(int num,
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
    fp32_scale_nhwc4_kernel<<<block, thread, 0, stream>>>(
        num / 4,
        static_cast<const float4*>(in),
        static_cast<float4*>(out),
        static_cast<const float4*>(scale),
        N,
        C / 4,
        H,
        W);
  } else {
    int block = (num + thread - 1) / thread;
    fp32_scale_nhwc_kernel<<<block, thread, 0, stream>>>(
        num,
        static_cast<const float*>(in),
        static_cast<float*>(out),
        static_cast<const float*>(scale),
        N,
        C,
        H,
        W);
  }

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) std::cout << cudaGetErrorString(error);
}

template <typename T>
void scale(int num, const T* in, T* out, T scale, T bias, cudaStream_t stream) {
  int thread = 256;
  int block = (num + thread - 1) / thread;
  scale_kernel<<<block, thread, 0, stream>>>(num, in, out, scale, bias);
  CUDA_POST_KERNEL_CHECK;
}

template <typename T>
void scale(int num, const T* in, T* out, T scale, T bias) {
  int thread = 256;
  int block = (num + thread - 1) / thread;
  scale_kernel<<<block, thread>>>(num, in, out, scale, bias);
  CUDA_POST_KERNEL_CHECK;
}

template void scale(int num, const float*, float*, float, float, cudaStream_t);
template void scale(int num, const float*, float*, float, float);

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
