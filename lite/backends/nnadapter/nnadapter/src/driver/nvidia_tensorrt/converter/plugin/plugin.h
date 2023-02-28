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
#include <vector>
#include "driver/nvidia_tensorrt/utility.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

class Plugin : public nvinfer1::IPluginV2 {
 public:
  Plugin();
  Plugin(const void* serial_data, size_t serial_length);
  ~Plugin() TRT_NOEXCEPT;
  // Override funcs in IPluginV2
  virtual const char* getPluginType() const TRT_NOEXCEPT = 0;
  const char* getPluginVersion() const TRT_NOEXCEPT;
  int32_t getNbOutputs() const TRT_NOEXCEPT;
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims* inputs,
                                             int nb_input_dims) TRT_NOEXCEPT;
  virtual bool supportsFormat(nvinfer1::DataType type,
                              nvinfer1::PluginFormat format) const TRT_NOEXCEPT;
  virtual void configureWithFormat(const nvinfer1::Dims* input_dims,
                                   int nb_inputs,
                                   const nvinfer1::Dims* output_dims,
                                   int nb_outputs,
                                   nvinfer1::DataType type,
                                   nvinfer1::PluginFormat format,
                                   int max_batch_size) TRT_NOEXCEPT;
  virtual int initialize() TRT_NOEXCEPT;
  virtual void terminate() TRT_NOEXCEPT;
  virtual size_t getWorkspaceSize(int max_batch_size) const TRT_NOEXCEPT;
  virtual int enqueue(int batch_size,
#if TENSORRT_VERSION_GE(8, 0, 0, 0)
                      void const* const* inputs,
                      void* const* outputs,
#else
                      const void* const* inputs,
                      void** outputs,
#endif
                      void* workspace,
                      cudaStream_t stream) TRT_NOEXCEPT = 0;
  virtual size_t getSerializationSize() const TRT_NOEXCEPT;
  virtual void serialize(void* buffer) const TRT_NOEXCEPT;
  virtual void destroy() TRT_NOEXCEPT;
  virtual IPluginV2* clone() const TRT_NOEXCEPT = 0;
  virtual void setPluginNamespace(const char* plugin_namespace) TRT_NOEXCEPT;
  virtual const char* getPluginNamespace() const TRT_NOEXCEPT;

 protected:
  std::vector<nvinfer1::Dims> input_dims_;
  nvinfer1::DataType data_type_;
  nvinfer1::PluginFormat data_format_;

 private:
  std::string namespace_;
};

class PluginDynamic : public nvinfer1::IPluginV2DynamicExt {
 public:
  PluginDynamic();
  PluginDynamic(const void* serial_data, size_t serial_length);
  ~PluginDynamic() TRT_NOEXCEPT;
  // Override funcs in IPluginV2DynamicExt
  virtual nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT = 0;
  nvinfer1::DimsExprs getOutputDimensions(
      int32_t output_index,
      const nvinfer1::DimsExprs* inputs,
      int32_t nb_inputs,
      nvinfer1::IExprBuilder& expr_builder)  // NOLINT
      TRT_NOEXCEPT;
  bool supportsFormatCombination(int32_t pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int32_t nb_inputs,
                                 int32_t nb_outputs) TRT_NOEXCEPT;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int32_t nb_inputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int32_t nb_outputs) TRT_NOEXCEPT;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int32_t nb_inputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int32_t nb_outputs) const TRT_NOEXCEPT;
  virtual int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                          const nvinfer1::PluginTensorDesc* output_desc,
                          void const* const* inputs,
                          void* const* outputs,
                          void* workspace,
                          cudaStream_t stream) TRT_NOEXCEPT = 0;
  // Override funcs in IpluginV2Ext
  nvinfer1::DataType getOutputDataType(int32_t index,
                                       const nvinfer1::DataType* input_types,
                                       int32_t nb_inputs) const TRT_NOEXCEPT;
  // Override funcs in IPluginV2
  virtual const char* getPluginType() const TRT_NOEXCEPT = 0;
  const char* getPluginVersion() const TRT_NOEXCEPT;
  int32_t getNbOutputs() const TRT_NOEXCEPT;
  int32_t initialize() TRT_NOEXCEPT;
  void terminate() TRT_NOEXCEPT;
  size_t getSerializationSize() const TRT_NOEXCEPT;
  void serialize(void* buffer) const TRT_NOEXCEPT;
  void destroy() TRT_NOEXCEPT;
  void setPluginNamespace(const char* plugin_namespace) TRT_NOEXCEPT;
  const char* getPluginNamespace() const TRT_NOEXCEPT;

 private:
  std::string namespace_;
};

class PluginCreator : public nvinfer1::IPluginCreator {
 public:
  PluginCreator() = default;
  virtual ~PluginCreator() = default;
  // Override funcs in IPluginCreator
  virtual const char* getPluginName() const TRT_NOEXCEPT = 0;
  const char* getPluginVersion() const TRT_NOEXCEPT;
  const nvinfer1::PluginFieldCollection* getFieldNames() TRT_NOEXCEPT;
  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) TRT_NOEXCEPT;
  virtual nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                                 void const* serial_data,
                                                 size_t serial_length)
      TRT_NOEXCEPT = 0;
  void setPluginNamespace(const char* plugin_namespace) TRT_NOEXCEPT;
  const char* getPluginNamespace() const TRT_NOEXCEPT;

 private:
  std::string namespace_;
  nvinfer1::PluginFieldCollection field_collection_{0, nullptr};
};

#define REGISTER_NNADAPTER_TENSORRT_PLUGIN(plugin_, plugin_creater_, name_) \
  const char* plugin_::getPluginType() const TRT_NOEXCEPT { return name_; } \
  const char* plugin_creater_::getPluginName() const TRT_NOEXCEPT {         \
    return name_;                                                           \
  }                                                                         \
  nvinfer1::IPluginV2* plugin_creater_::deserializePlugin(                  \
      const char* name, void const* serial_data, size_t serial_length)      \
      TRT_NOEXCEPT {                                                        \
    return new plugin_(serial_data, serial_length);                         \
  }                                                                         \
  REGISTER_TENSORRT_PLUGIN(plugin_creater_);

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
