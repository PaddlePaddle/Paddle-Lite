// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/nvidia_tensorrt/converter/plugin/hard_swish.h"

namespace nnadapter {
namespace nvidia_tensorrt {

template <typename T, unsigned TPB>
__global__ void HardSwishKernel(
    const T* input, T* output, int num, T alpha, T beta) {
  const int idx = blockIdx.x * TPB + threadIdx.x;
  if (idx < num) {
    output[idx] = input[idx] * alpha + beta;
    output[idx] =
        output[idx] < static_cast<T>(1) ? output[idx] : static_cast<T>(1);
    output[idx] =
        output[idx] > static_cast<T>(0) ? output[idx] : static_cast<T>(0);
    output[idx] *= input[idx];
  }
}

template <typename T>
cudaError_t HardSwish(
    const T* input, T* output, int num, T alpha, T beta, cudaStream_t stream) {
  const int block_size = 256;
  const int grid_size = (num + block_size - 1) / block_size;
  HardSwishKernel<T, block_size><<<grid_size, block_size, 0, stream>>>(
      input, output, num, alpha, beta);
  return cudaGetLastError();
}

template cudaError_t HardSwish(const float* input,
                               float* output,
                               int num,
                               float alpha,
                               float beta,
                               cudaStream_t stream);

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
