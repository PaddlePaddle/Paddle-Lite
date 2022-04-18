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
#include "driver/nvidia_tensorrt/utility.h"
#include "utility/debug.h"

namespace nnadapter {
namespace nvidia_tensorrt {

class CastPlugin : public Plugin {
 public:
  CastPlugin();
  CastPlugin(nvinfer1::DataType intype, nvinfer1::DataType outype);
  CastPlugin(const void* serial_data, size_t serial_length);
  ~CastPlugin();
  const char* getPluginType() const noexcept;
  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const noexcept;
  int enqueue(int batch_size,
              const void* const* inputs,
              void** outputs,
              void* workspace,
              cudaStream_t stream) noexcept;
  size_t getSerializationSize() const noexcept;
  void serialize(void* buffer) const noexcept;
  nvinfer1::IPluginV2* clone() const noexcept;

 private:
  nvinfer1::DataType intype_;
  nvinfer1::DataType outtype_;
};

class CastPluginDynamic : public PluginDynamic {
 public:
  CastPluginDynamic(nvinfer1::DataType intype, nvinfer1::DataType outype);
  CastPluginDynamic(const void* serial_data, size_t serial_length);
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
  bool supportsFormatCombination(int32_t pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int32_t nb_inputs,
                                 int32_t nb_outputs) noexcept;

 private:
  nvinfer1::DataType intype_;
  nvinfer1::DataType outtype_;
};

class CastPluginDynamicCreator : public PluginCreator {
 public:
  const char* getPluginName() const noexcept;
  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         void const* serial_data,
                                         size_t serial_length) noexcept;
};

class CastPluginCreator : public PluginCreator {
 public:
  const char* getPluginName() const noexcept;
  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         void const* serial_data,
                                         size_t serial_length) noexcept;
};

template <typename Tin, typename Tout>
cudaError_t Cast(const Tin* input, Tout* output, int num, cudaStream_t stream);

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
