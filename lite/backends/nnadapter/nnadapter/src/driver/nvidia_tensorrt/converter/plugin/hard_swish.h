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

#pragma once
#include "driver/nvidia_tensorrt/converter/plugin/plugin.h"

namespace nnadapter {
namespace nvidia_tensorrt {

class HardSwishPlugin : public Plugin {
 public:
  HardSwishPlugin();
  HardSwishPlugin(float alpha, float beta);
  HardSwishPlugin(const void* serial_data, size_t serial_length);
  ~HardSwishPlugin();
  const char* getPluginType() const noexcept;
  int enqueue(int batch_size,
              const void* const* inputs,
              void** outputs,
              void* workspace,
              cudaStream_t stream) noexcept;
  size_t getSerializationSize() const noexcept;
  void serialize(void* buffer) const noexcept;
  nvinfer1::IPluginV2* clone() const noexcept;

 private:
  float alpha_;
  float beta_;
};

class HardSwishPluginDynamic : public PluginDynamic {
 public:
  HardSwishPluginDynamic();
  HardSwishPluginDynamic(float alpha, float beta);
  HardSwishPluginDynamic(const void* serial_data, size_t serial_length);
  ~HardSwishPluginDynamic();
  nvinfer1::IPluginV2DynamicExt* clone() const noexcept;
  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs,
                  void* const* outputs,
                  void* workspace,
                  cudaStream_t stream) noexcept;
  const char* getPluginType() const noexcept;
  size_t getSerializationSize() const noexcept;
  void serialize(void* buffer) const noexcept;

 private:
  float alpha_;
  float beta_;
};

class HardSwishPluginCreator : public PluginCreator {
 public:
  const char* getPluginName() const noexcept;
  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         void const* serial_data,
                                         size_t serial_length) noexcept;
};

class HardSwishPluginDynamicCreator : public PluginCreator {
 public:
  const char* getPluginName() const noexcept;
  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         void const* serial_data,
                                         size_t serial_length) noexcept;
};

template <typename T>
cudaError_t HardSwish(
    const T* input, T* output, int num, T alpha, T beta, cudaStream_t stream);

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
