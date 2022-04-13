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

class SwishPlugin : public Plugin {
 public:
  SwishPlugin() = default;

  explicit SwishPlugin(const float beta) : beta_(beta) {}

  SwishPlugin(const void* serial_data, size_t serial_length) {
    Deserialize(&serial_data, &serial_length, &beta_);
  }

  nvinfer1::IPluginV2* clone() const noexcept { return new SwishPlugin(beta_); }

  int enqueue(int batch_size,
              const void* const* inputs,
              void** outputs,
              void* workspace,
              cudaStream_t stream) noexcept;

  const char* getPluginType() const noexcept;

  size_t getSerializationSize() const noexcept { return SerializedSize(beta_); }

  void serialize(void* buffer) const noexcept { Serialize(&buffer, beta_); };

 private:
  float beta_{1.0f};
};

class SwishPluginCreator : public PluginCreator {
 public:
  const char* getPluginName() const noexcept;
  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         void const* serial_data,
                                         size_t serial_length) noexcept;
};

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
