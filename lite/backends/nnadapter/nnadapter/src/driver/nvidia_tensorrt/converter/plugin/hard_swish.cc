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

HardSwishPlugin::HardSwishPlugin() {}

HardSwishPlugin::HardSwishPlugin(float alpha, float beta)
    : alpha_(alpha), beta_(beta) {}

HardSwishPlugin::HardSwishPlugin(const void* serial_data,
                                 size_t serial_length) {
  Deserialize(&serial_data, &serial_length, &alpha_);
  Deserialize(&serial_data, &serial_length, &beta_);
}

HardSwishPlugin::~HardSwishPlugin() TRT_NOEXCEPT {}

int HardSwishPlugin::enqueue(int batch_size,
#if TENSORRT_VERSION_GE(8, 0, 0, 0)
                             void const* const* inputs,
                             void* const* outputs,
#else
                             const void* const* inputs,
                             void** outputs,
#endif
                             void* workspace,
                             cudaStream_t stream) TRT_NOEXCEPT {
  int num = batch_size;
  for (int i = 0; i < input_dims_[0].nbDims; i++) {
    num *= input_dims_[0].d[i];
  }
  const float* input = static_cast<const float*>(inputs[0]);
  float* output = static_cast<float*>(outputs[0]);
  NNADAPTER_CHECK_EQ(HardSwish(input, output, num, alpha_, beta_, stream),
                     cudaSuccess);
  return 0;
}

size_t HardSwishPlugin::getSerializationSize() const TRT_NOEXCEPT {
  return SerializedSize(alpha_) + SerializedSize(beta_);
}

void HardSwishPlugin::serialize(void* buffer) const TRT_NOEXCEPT {
  Serialize(&buffer, alpha_);
  Serialize(&buffer, beta_);
}

nvinfer1::IPluginV2* HardSwishPlugin::clone() const TRT_NOEXCEPT {
  return new HardSwishPlugin(alpha_, beta_);
}

HardSwishPluginDynamic::HardSwishPluginDynamic() {}

HardSwishPluginDynamic::HardSwishPluginDynamic(float alpha, float beta)
    : alpha_(alpha), beta_(beta) {}

HardSwishPluginDynamic::HardSwishPluginDynamic(const void* serial_data,
                                               size_t serial_length) {
  Deserialize(&serial_data, &serial_length, &alpha_);
  Deserialize(&serial_data, &serial_length, &beta_);
}

HardSwishPluginDynamic::~HardSwishPluginDynamic() TRT_NOEXCEPT {}

nvinfer1::IPluginV2DynamicExt* HardSwishPluginDynamic::clone() const
    TRT_NOEXCEPT {
  return new HardSwishPluginDynamic(alpha_, beta_);
}

int32_t HardSwishPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  auto input_dims = input_desc[0].dims;
  int num = 1;
  for (int i = 0; i < input_dims.nbDims; i++) {
    num *= input_dims.d[i];
  }
  const float* input = static_cast<const float*>(inputs[0]);
  float* output = static_cast<float*>(outputs[0]);
  NNADAPTER_CHECK_EQ(HardSwish(input, output, num, alpha_, beta_, stream),
                     cudaSuccess);
  return 0;
}

size_t HardSwishPluginDynamic::getSerializationSize() const TRT_NOEXCEPT {
  return SerializedSize(alpha_) + SerializedSize(beta_);
}

void HardSwishPluginDynamic::serialize(void* buffer) const TRT_NOEXCEPT {
  Serialize(&buffer, alpha_);
  Serialize(&buffer, beta_);
}

REGISTER_NNADAPTER_TENSORRT_PLUGIN(HardSwishPlugin,
                                   HardSwishPluginCreator,
                                   "hard_swish_plugin");
REGISTER_NNADAPTER_TENSORRT_PLUGIN(HardSwishPluginDynamic,
                                   HardSwishPluginDynamicCreator,
                                   "hard_swish_plugin_dynamic");

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
