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

#include "driver/nvidia_tensorrt/converter/plugin/plugin.h"

namespace nnadapter {
namespace nvidia_tensorrt {

Plugin::Plugin() {}

Plugin::Plugin(const void* serial_data, size_t serial_length) {}

Plugin::~Plugin() TRT_NOEXCEPT {}

const char* Plugin::getPluginVersion() const TRT_NOEXCEPT { return "1"; }

int32_t Plugin::getNbOutputs() const TRT_NOEXCEPT { return 1; }

nvinfer1::Dims Plugin::getOutputDimensions(int index,
                                           const nvinfer1::Dims* inputs,
                                           int nb_input_dims) TRT_NOEXCEPT {
  if (nb_input_dims > 0) {
    return inputs[0];
  } else {
    nvinfer1::Dims dims;
    dims.nbDims = -1;
    return dims;
  }
}

bool Plugin::supportsFormat(nvinfer1::DataType type,
                            nvinfer1::PluginFormat format) const TRT_NOEXCEPT {
  return (type == nvinfer1::DataType::kFLOAT) &&
         (format == nvinfer1::PluginFormat::kLINEAR);
}

void Plugin::configureWithFormat(const nvinfer1::Dims* input_dims,
                                 int nb_inputs,
                                 const nvinfer1::Dims* output_dims,
                                 int nb_outputs,
                                 nvinfer1::DataType type,
                                 nvinfer1::PluginFormat format,
                                 int max_batch_size) TRT_NOEXCEPT {
  input_dims_.assign(input_dims, input_dims + nb_inputs);
  data_type_ = type;
  data_format_ = format;
}

int Plugin::initialize() TRT_NOEXCEPT { return 0; }

void Plugin::terminate() TRT_NOEXCEPT {}

size_t Plugin::getWorkspaceSize(int max_batch_size) const TRT_NOEXCEPT {
  return 0;
}

size_t Plugin::getSerializationSize() const TRT_NOEXCEPT { return 0; }

void Plugin::serialize(void* buffer) const TRT_NOEXCEPT {}

void Plugin::destroy() TRT_NOEXCEPT {}

void Plugin::setPluginNamespace(const char* plugin_namespace) TRT_NOEXCEPT {
  namespace_ = plugin_namespace;
}

const char* Plugin::getPluginNamespace() const TRT_NOEXCEPT {
  return namespace_.c_str();
}

PluginDynamic::PluginDynamic() {}

PluginDynamic::PluginDynamic(const void* serial_data, size_t serial_length) {}

PluginDynamic::~PluginDynamic() TRT_NOEXCEPT {}

nvinfer1::DimsExprs PluginDynamic::getOutputDimensions(
    int32_t output_index,
    const nvinfer1::DimsExprs* inputs,
    int32_t nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  NNADAPTER_CHECK_EQ(output_index, 0);
  NNADAPTER_CHECK(inputs);
  NNADAPTER_CHECK_GE(nb_inputs, 1);
  return inputs[0];
}

bool PluginDynamic::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int32_t nb_inputs,
    int32_t nb_outputs) TRT_NOEXCEPT {
  NNADAPTER_CHECK_LT(pos, nb_inputs + nb_outputs);
  NNADAPTER_CHECK(in_out);
  return in_out[pos].type == nvinfer1::DataType::kFLOAT &&
         in_out[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void PluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int32_t nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int32_t nb_outputs) TRT_NOEXCEPT {}

size_t PluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nb_inputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nb_outputs) const TRT_NOEXCEPT {
  return 0;
}

nvinfer1::DataType PluginDynamic::getOutputDataType(
    int32_t index,
    const nvinfer1::DataType* input_types,
    int32_t nb_inputs) const TRT_NOEXCEPT {
  return nb_inputs > 0 ? input_types[0] : nvinfer1::DataType::kFLOAT;
}

const char* PluginDynamic::getPluginVersion() const TRT_NOEXCEPT { return "1"; }

int32_t PluginDynamic::getNbOutputs() const TRT_NOEXCEPT { return 1; }

int32_t PluginDynamic::initialize() TRT_NOEXCEPT { return 0; }

void PluginDynamic::terminate() TRT_NOEXCEPT {}

size_t PluginDynamic::getSerializationSize() const TRT_NOEXCEPT { return 0; }

void PluginDynamic::serialize(void* buffer) const TRT_NOEXCEPT {}

void PluginDynamic::destroy() TRT_NOEXCEPT { delete this; }

void PluginDynamic::setPluginNamespace(const char* plugin_namespace)
    TRT_NOEXCEPT {
  namespace_ = plugin_namespace;
}

const char* PluginDynamic::getPluginNamespace() const TRT_NOEXCEPT {
  return namespace_.c_str();
}

const char* PluginCreator::getPluginVersion() const TRT_NOEXCEPT { return "1"; }

const nvinfer1::PluginFieldCollection* PluginCreator::getFieldNames()
    TRT_NOEXCEPT {
  return &field_collection_;
}

nvinfer1::IPluginV2* PluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) TRT_NOEXCEPT {
  return nullptr;
}

void PluginCreator::setPluginNamespace(const char* plugin_namespace)
    TRT_NOEXCEPT {
  namespace_ = plugin_namespace;
}

const char* PluginCreator::getPluginNamespace() const TRT_NOEXCEPT {
  return namespace_.c_str();
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
