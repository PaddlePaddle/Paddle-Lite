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

#include "driver/nvidia_tensorrt/plugin/relu.h"

namespace nnadapter {
namespace nvidia_tensorrt {

ReluPluginDynamic::ReluPluginDynamic() {}

ReluPluginDynamic::ReluPluginDynamic(const void* serial_data,
                                     size_t serial_length) {}

nvinfer1::IPluginV2DynamicExt* ReluPluginDynamic::clone() const noexcept {
  return new ReluPluginDynamic;
}

template <typename T, unsigned TPB>
__global__ void relu_kernel(int n, const T* input, T* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;
  if (idx < n) {
    output[idx] = input[idx] > 0 ? input[idx] : 0.f;
  }
}

int32_t ReluPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  auto input_dims = input_desc[0].dims;
  int num = 1;
  for (int i = 0; i < input_dims.nbDims; i++) {
    num *= input_dims.d[i];
  }
  const int block_size = 256;
  const int grid_size = (num + block_size - 1) / block_size;
  const float* input = static_cast<const float*>(inputs[0]);
  float* output = static_cast<float*>(outputs[0]);
  relu_kernel<float, block_size><<<grid_size, block_size, 0, stream>>>(
      num, input, output);
  return 0;
}

nvinfer1::IPluginV2* ReluPluginDynamicCreator::deserializePlugin(
    const char* name, void const* serial_data, size_t serial_length) noexcept {
  return new ReluPluginDynamic(serial_data, serial_length);
}

REGISTER_NNADAPTER_TENSORRT_PLUGIN(ReluPluginDynamic,
                                   ReluPluginDynamicCreator,
                                   "relu_plugin_dynamic");

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
