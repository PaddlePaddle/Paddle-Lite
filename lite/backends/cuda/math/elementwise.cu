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

#include "lite/backends/cuda/math/elementwise.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename Dtype>
__global__ void elementwise_kernel(const size_t total,
                                   const Dtype* x_data,
                                   const Dtype* y_data,
                                   Dtype* out_data,
                                   int pre,
                                   int n,
                                   int post,
                                   BinaryOperation type) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < total) {
    int idx = tid / post % n;
#if __CUDA_ARCH__ >= 350
    out_data[tid] = binary_calc(__ldg(x_data + tid), __ldg(y_data + idx), type);
#else
    out_data[tid] = binary_calc(x_data[tid], y_data[idx], type);
#endif
  }
}

template <typename Dtype>
__global__ void elementwise_relu_kernel(const size_t total,
                                        const Dtype* x_data,
                                        const Dtype* y_data,
                                        Dtype* out_data,
                                        int pre,
                                        int n,
                                        int post,
                                        BinaryOperation type) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < total) {
    int idx = tid / post % n;
    Dtype temp;
#if __CUDA_ARCH__ >= 350
    temp = binary_calc(__ldg(x_data + tid), __ldg(y_data + idx), type);

#else
    temp = binary_calc(x_data[tid], y_data[idx], type);
#endif
    out_data[tid] = temp > 0 ? temp : 0;
  }
}

template <typename Dtype>
__global__ void elementwise_abs_kernel(const size_t total,
                                       const Dtype* x_data,
                                       const Dtype* y_data,
                                       Dtype* out_data,
                                       int pre,
                                       int n,
                                       int post,
                                       BinaryOperation type) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < total) {
    int idx = tid / post % n;
    Dtype temp;
#if __CUDA_ARCH__ >= 350
    temp = binary_calc(__ldg(x_data + tid), __ldg(y_data + idx), type);

#else
    temp = binary_calc(x_data[tid], y_data[idx], type);
#endif
    out_data[tid] = temp > 0 ? temp : -temp;
  }
}

template <typename Dtype>
__global__ void elementwise_tanh_kernel(const size_t total,
                                        const Dtype* x_data,
                                        const Dtype* y_data,
                                        Dtype* out_data,
                                        int pre,
                                        int n,
                                        int post,
                                        BinaryOperation type) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < total) {
    int idx = tid / post % n;
    Dtype temp;
#if __CUDA_ARCH__ >= 350
    temp = binary_calc(__ldg(x_data + tid), __ldg(y_data + idx), type);

#else
    temp = binary_calc(x_data[tid], y_data[idx], type);
#endif
    out_data[tid] = tanh(temp);
  }
}

template <typename Dtype>
__global__ void elementwise_add_kernel(const size_t total,
                                       const Dtype* x_data,
                                       const Dtype* y_data,
                                       Dtype* out_data) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < total) {
#if __CUDA_ARCH__ >= 350
    out_data[tid] = __ldg(x_data + tid) + __ldg(y_data + tid);
#else
    out_data[tid] = x_data[tid] + y_data[tid];
#endif
  }
}

__global__ void elementwise_add_int8_kernel(const size_t total,
                                            const float* x_data,
                                            const float* y_data,
                                            const float alpha,
                                            int8_t* out_data) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < total) {
    float temp_d;
#if __CUDA_ARCH__ >= 350
    temp_d = __ldg(x_data + tid) + __ldg(y_data + tid);
#else
    temp_d = x_data[tid] + y_data[tid];
#endif
    out_data[tid] = from_float<int8_t>(temp_d * alpha);
  }
}

__global__ void elementwise_add_nhwc4_int8_kernel(const size_t total,
                                                  const float4* x_data,
                                                  const float4* y_data,
                                                  const float alpha,
                                                  char4* out_data) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < total) {
    const float4 x_d = x_data[tid];
    const float4 y_d = y_data[tid];

    float4 packed_val;
    char4 result_val;
    packed_val.x = (x_d.x + y_d.x) * alpha;
    result_val.x = from_float<int8_t>(packed_val.x);
    packed_val.y = (x_d.y + y_d.y) * alpha;
    result_val.y = from_float<int8_t>(packed_val.y);
    packed_val.z = (x_d.z + y_d.z) * alpha;
    result_val.z = from_float<int8_t>(packed_val.z);
    packed_val.w = (x_d.w + y_d.w) * alpha;
    result_val.w = from_float<int8_t>(packed_val.w);
    out_data[tid] = result_val;
  }
}

template <typename Dtype>
void elementwise(const Dtype* x_data,
                 const Dtype* y_data,
                 Dtype* out_data,
                 int pre,
                 int n,
                 int post,
                 BinaryOperation type,
                 cudaStream_t stream) {
  int num = pre * n * post;
  int thread = 256;
  int block = (num + thread - 1) / thread;
  elementwise_kernel<<<block, thread, 0, stream>>>(
      num, x_data, y_data, out_data, pre, n, post, type);
}

template <typename Dtype>
void elementwise_act(const Dtype* x_data,
                     const Dtype* y_data,
                     Dtype* out_data,
                     int pre,
                     int n,
                     int post,
                     std::string act,
                     BinaryOperation type,
                     cudaStream_t stream) {
  int num = pre * n * post;
  int thread = 256;
  int block = (num + thread - 1) / thread;
  if (act == "relu") {
    elementwise_relu_kernel<<<block, thread, 0, stream>>>(
        num, x_data, y_data, out_data, pre, n, post, type);
  } else if (act == "tanh") {
    elementwise_tanh_kernel<<<block, thread, 0, stream>>>(
        num, x_data, y_data, out_data, pre, n, post, type);
  } else if (act == "abs") {
    elementwise_abs_kernel<<<block, thread, 0, stream>>>(
        num, x_data, y_data, out_data, pre, n, post, type);
  } else {
    LOG(FATAL) << "not supported activate type: " << act;
  }
}

template void elementwise(const float*,
                          const float*,
                          float*,
                          int,
                          int,
                          int,
                          BinaryOperation,
                          cudaStream_t);

template void elementwise_act(const float* x_data,
                              const float* y_data,
                              float* out_data,
                              int pre,
                              int n,
                              int post,
                              std::string act,
                              BinaryOperation type,
                              cudaStream_t stream);

template <typename Dtype>
void elementwise_add(int num,
                     const Dtype* x_data,
                     const Dtype* y_data,
                     Dtype* out_data,
                     cudaStream_t stream) {
  int thread = 256;
  int block = (num + thread - 1) / thread;
  elementwise_add_kernel<<<block, thread, 0, stream>>>(
      num, x_data, y_data, out_data);
}

template void elementwise_add(
    int, const float*, const float*, float*, cudaStream_t);

// input type is float32
// output type is int8
void elementwise_add_int8(int num,
                          const float* x_data,
                          const float* y_data,
                          const float alpha,
                          int8_t* out_data,
                          cudaStream_t stream) {
  int thread = 256;
  int block = (num + thread - 1) / thread;
  // elementwise_add_int8_kernel<<<block, thread, 0, stream>>>(
  elementwise_add_int8_kernel<<<block, thread>>>(
      num, x_data, y_data, alpha, out_data);
}

void elementwise_add_nhwc4_int8(int num,
                                const void* x_data,
                                const void* y_data,
                                const float alpha,
                                void* out_data,
                                cudaStream_t stream) {
  int thread = 512;
  int block = (num + thread - 1) / thread;
  // elementwise_add_nhwc4_int8_kernel<<<block, thread, 0, stream>>>(
  elementwise_add_nhwc4_int8_kernel<<<block, thread>>>(
      num,
      static_cast<const float4*>(x_data),
      static_cast<const float4*>(y_data),
      alpha,
      static_cast<char4*>(out_data));
}

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
