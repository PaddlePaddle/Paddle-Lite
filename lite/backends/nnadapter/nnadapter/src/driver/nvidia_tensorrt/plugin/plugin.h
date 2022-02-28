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
#include <string>
#include "driver/nvidia_tensorrt/utility.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

class PluginDynamic : public nvinfer1::IPluginV2DynamicExt {
 public:
  PluginDynamic();
  PluginDynamic(const void* serial_data, size_t serial_length);
  ~PluginDynamic();
  // Override funcs in IPluginV2DynamicExt
  virtual nvinfer1::IPluginV2DynamicExt* clone() const noexcept = 0;
  nvinfer1::DimsExprs getOutputDimensions(
      int32_t output_index,
      const nvinfer1::DimsExprs* inputs,
      int32_t nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) noexcept;  // NOLINT
  bool supportsFormatCombination(int32_t pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int32_t nb_inputs,
                                 int32_t nb_outputs) noexcept;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int32_t nb_inputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int32_t nb_outputs) noexcept;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int32_t nb_inputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int32_t nb_outputs) const noexcept;
  virtual int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                          const nvinfer1::PluginTensorDesc* output_desc,
                          const void* const* inputs,
                          void* const* outputs,
                          void* workspace,
                          cudaStream_t stream) noexcept = 0;
  // Override funcs in IpluginV2Ext
  nvinfer1::DataType getOutputDataType(int32_t index,
                                       const nvinfer1::DataType* input_types,
                                       int32_t nb_inputs) const noexcept;
  // Override funcs in IPluginV2
  virtual const char* getPluginType() const noexcept = 0;
  const char* getPluginVersion() const noexcept;
  int32_t getNbOutputs() const noexcept;
  int32_t initialize() noexcept;
  void terminate() noexcept;
  size_t getSerializationSize() const noexcept;
  void serialize(void* buffer) const noexcept;
  void destroy() noexcept;
  void setPluginNamespace(const char* plugin_namespace) noexcept;
  const char* getPluginNamespace() const noexcept;

 private:
  std::string namespace_;
};

class PluginCreator : public nvinfer1::IPluginCreator {
 public:
  PluginCreator() = default;
  virtual ~PluginCreator() = default;
  // Override funcs in IPluginCreator
  virtual const char* getPluginName() const noexcept = 0;
  const char* getPluginVersion() const noexcept;
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept;
  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept;
  virtual nvinfer1::IPluginV2* deserializePlugin(
      const char* name,
      void const* serial_data,
      size_t serial_length) noexcept = 0;
  void setPluginNamespace(const char* plugin_namespace) noexcept;
  const char* getPluginNamespace() const noexcept;

 private:
  std::string namespace_;
  nvinfer1::PluginFieldCollection field_collection_{0, nullptr};
};

#define REGISTER_NNADAPTER_TENSORRT_PLUGIN(plugin_, plugin_creater_, name_) \
  const char* plugin_::getPluginType() const noexcept { return name_; }     \
  const char* plugin_creater_::getPluginName() const noexcept {             \
    return name_;                                                           \
  }                                                                         \
  nvinfer1::IPluginV2* deserializePlugin(const char* name,                  \
                                         void const* serial_data,           \
                                         size_t serial_length) noexcept {   \
    return new plugin_(serial_data, serial_length);                         \
  }                                                                         \
  REGISTER_TENSORRT_PLUGIN(plugin_creater_);

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
