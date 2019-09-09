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
#include "lite/backends/cuda/math/scale.h"
#include "lite/backends/cuda/math/utils.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

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

void fp32_scale_nhwc4(int num,
                      const void* in,
                      void* out,
                      const void* scale,
                      int N,
                      int K,
                      int H,
                      int W,
                      cudaStream_t stream) {
  int thread = 256;
  int block = (num + thread - 1) / thread;
  fp32_scale_nhwc4_kernel<<<block, thread, 0, stream>>>(
      num,
      static_cast<const float4*>(in),
      static_cast<float4*>(out),
      static_cast<const float4*>(scale),
      N,
      K,
      H,
      W);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) std::cout << cudaGetErrorString(error);
}

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
