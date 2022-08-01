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
  ~CastPlugin() TRT_NOEXCEPT;
  const char* getPluginType() const TRT_NOEXCEPT;
  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const TRT_NOEXCEPT;
  int enqueue(int batch_size,
#if TENSORRT_VERSION_GE(8, 0, 0, 0)
              void const* const* inputs,
              void* const* outputs,
#else
              const void* const* inputs,
              void** outputs,
#endif
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT;
  size_t getSerializationSize() const TRT_NOEXCEPT;
  void serialize(void* buffer) const TRT_NOEXCEPT;
  nvinfer1::IPluginV2* clone() const TRT_NOEXCEPT;

 private:
  nvinfer1::DataType intype_;
  nvinfer1::DataType outtype_;
};

class CastPluginDynamic : public PluginDynamic {
 public:
  CastPluginDynamic(nvinfer1::DataType intype, nvinfer1::DataType outype);
  CastPluginDynamic(const void* serial_data, size_t serial_length);
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT;
  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  void const* const* inputs,
                  void* const* outputs,
                  void* workspace,
                  cudaStream_t stream) TRT_NOEXCEPT;
  const char* getPluginType() const TRT_NOEXCEPT;
  size_t getSerializationSize() const TRT_NOEXCEPT;
  void serialize(void* buffer) const TRT_NOEXCEPT;
  bool supportsFormatCombination(int32_t pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int32_t nb_inputs,
                                 int32_t nb_outputs) TRT_NOEXCEPT;

 private:
  nvinfer1::DataType intype_;
  nvinfer1::DataType outtype_;
};

class CastPluginDynamicCreator : public PluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT;
  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         void const* serial_data,
                                         size_t serial_length) TRT_NOEXCEPT;
};

class CastPluginCreator : public PluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT;
  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         void const* serial_data,
                                         size_t serial_length) TRT_NOEXCEPT;
};

template <typename Tin, typename Tout>
cudaError_t Cast(const Tin* input, Tout* output, int num, cudaStream_t stream);

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
