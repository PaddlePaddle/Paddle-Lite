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

#include "driver/nvidia_tensorrt/converter/plugin/cast.h"

namespace nnadapter {
namespace nvidia_tensorrt {

template <typename Tin, typename Tout, unsigned TPB>
__global__ void CastKernel(const Tin* input, Tout* output, int num) {
  const int idx = blockIdx.x * TPB + threadIdx.x;
  if (idx < num) {
    output[idx] = static_cast<Tout>(input[idx]);
  }
}

template <typename Tin, typename Tout>
cudaError_t Cast(const Tin* input, Tout* output, int num, cudaStream_t stream) {
  const int block_size = 256;
  const int grid_size = (num + block_size - 1) / block_size;
  CastKernel<Tin, Tout, block_size><<<grid_size, block_size, 0, stream>>>(
      input, output, num);
  return cudaGetLastError();
}

template cudaError_t Cast(const float* input,
                          int32_t* output,
                          int num,
                          cudaStream_t stream);
template cudaError_t Cast(const int32_t* input,
                          float* output,
                          int num,
                          cudaStream_t stream);

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
