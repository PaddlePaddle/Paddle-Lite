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

#include "driver/nvidia_tensorrt/converter/plugin/swish.h"

namespace nnadapter {
namespace nvidia_tensorrt {

template <typename T>
__device__ inline T MathExp(T a);

template <>
__device__ inline float MathExp<float>(float a) {
  return expf(a);
}

template <typename T>
__global__ void SwishKernel(int num, const T* input, T* output, T beta) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num) {
    output[index] =
        input[index] / (static_cast<T>(1.0) + MathExp<T>(-beta * input[index]));
  }
}

int SwishPlugin::enqueue(int batch_size,
#if TENSORRT_VERSION_GE(8, 0, 0, 0)
                         void const* const* inputs,
                         void* const* outputs,
#else
                         const void* const* inputs,
                         void** outputs,
#endif
                         void* workspace,
                         cudaStream_t stream) TRT_NOEXCEPT {
  auto input_dims = input_dims_[0];
  int num = batch_size;
  for (int i = 0; i < input_dims.nbDims; i++) {
    num *= input_dims.d[i];
  }
  int threads = 1024;
  int blocks = (num + threads - 1) / threads;
  const float* input = static_cast<const float*>(inputs[0]);
  float* output = static_cast<float*>(outputs[0]);
  SwishKernel<float><<<blocks, threads, 0, stream>>>(num, input, output, beta_);
  return 0;
}

REGISTER_NNADAPTER_TENSORRT_PLUGIN(SwishPlugin,
                                   SwishPluginCreator,
                                   "swish_plugin");

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
