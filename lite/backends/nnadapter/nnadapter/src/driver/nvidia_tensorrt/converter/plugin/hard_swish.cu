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

HardSwishPluginDynamic::HardSwishPluginDynamic() {}

HardSwishPluginDynamic::HardSwishPluginDynamic(float alpha, float beta)
    : alpha_(alpha), beta_(beta) {}

HardSwishPluginDynamic::HardSwishPluginDynamic(const void* serial_data,
                                               size_t serial_length) {
  Deserialize(&serial_data, &serial_length, &alpha_);
  Deserialize(&serial_data, &serial_length, &beta_);
}

nvinfer1::IPluginV2DynamicExt* HardSwishPluginDynamic::clone() const noexcept {
  return new HardSwishPluginDynamic(alpha_, beta_);
}

template <typename T, unsigned TPB>
__global__ void hard_swish_kernel(
    int n, const T* input, T* output, T alpha, T beta) {
  const int idx = blockIdx.x * TPB + threadIdx.x;
  if (idx < n) {
    output[idx] = input[idx] * alpha + beta;
    output[idx] =
        output[idx] < static_cast<T>(1) ? output[idx] : static_cast<T>(1);
    output[idx] =
        output[idx] > static_cast<T>(0) ? output[idx] : static_cast<T>(0);
    output[idx] *= input[idx];
  }
}

int32_t HardSwishPluginDynamic::enqueue(
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
  hard_swish_kernel<float, block_size><<<grid_size, block_size, 0, stream>>>(
      num, input, output, alpha_, beta_);
  return 0;
}

size_t HardSwishPluginDynamic::getSerializationSize() const noexcept {
  return SerializedSize(alpha_) + SerializedSize(beta_);
}

void HardSwishPluginDynamic::serialize(void* buffer) const noexcept {
  Serialize(&buffer, alpha_);
  Serialize(&buffer, beta_);
}

REGISTER_NNADAPTER_TENSORRT_PLUGIN(HardSwishPluginDynamic,
                                   HardSwishPluginDynamicCreator,
                                   "hard_swish_plugin_dynamic");

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
