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

__global__ void fp32_scale_nhwc4_kernel(int num,
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

void fp32_to_int8_nhwc4(int num,
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
      static_cast<char4*>(out),
      static_cast<const float4*>(scale),
      N,
      K,
      H,
      W);
}

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
