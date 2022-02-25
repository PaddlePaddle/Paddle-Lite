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

#include "driver/nvidia_tensorrt/plugin/plugin.h"

namespace nnadapter {
namespace nvidia_tensorrt {

PluginDynamic::PluginDynamic() {}

PluginDynamic::PluginDynamic(const void* serial_data, size_t serial_length) {}

PluginDynamic::~PluginDynamic() {}

nvinfer1::DimsExprs PluginDynamic::getOutputDimensions(
    int32_t output_index,
    const nvinfer1::DimsExprs* inputs,
    int32_t nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) noexcept {
  NNADAPTER_CHECK_EQ(output_index, 0);
  NNADAPTER_CHECK(inputs);
  NNADAPTER_CHECK_GE(nb_inputs, 1);
  return inputs[0];
}

bool PluginDynamic::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int32_t nb_inputs,
    int32_t nb_outputs) noexcept {
  NNADAPTER_CHECK_LT(pos, nb_inputs + nb_outputs);
  NNADAPTER_CHECK(in_out);
  return in_out[pos].type == nvinfer1::DataType::kFLOAT &&
         in_out[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void PluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int32_t nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int32_t nb_outputs) noexcept {}

size_t PluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nb_inputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nb_outputs) const noexcept {
  return 0;
}

nvinfer1::DataType PluginDynamic::getOutputDataType(
    int32_t index,
    const nvinfer1::DataType* input_types,
    int32_t nb_inputs) const noexcept {
  NNADAPTER_CHECK_EQ(index, 0);
  NNADAPTER_CHECK(input_types);
  NNADAPTER_CHECK_GE(nb_inputs, 1);
  return input_types[0];
}

const char* PluginDynamic::getPluginVersion() const noexcept { return "1"; }

int32_t PluginDynamic::getNbOutputs() const noexcept { return 1; }

int32_t PluginDynamic::initialize() noexcept { return 0; }

void PluginDynamic::terminate() noexcept {}

size_t PluginDynamic::getSerializationSize() const noexcept { return 0; }

void PluginDynamic::serialize(void* buffer) const noexcept {}

void PluginDynamic::destroy() noexcept { delete this; }

void PluginDynamic::setPluginNamespace(const char* plugin_namespace) noexcept {
  namespace_ = plugin_namespace;
}

const char* PluginDynamic::getPluginNamespace() const noexcept {
  return namespace_.data();
}

const char* PluginCreator::getPluginVersion() const noexcept { return "1"; }

const nvinfer1::PluginFieldCollection* PluginCreator::getFieldNames() noexcept {
  return &field_collection_;
}

nvinfer1::IPluginV2* PluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
  return nullptr;
}

void PluginCreator::setPluginNamespace(const char* plugin_namespace) noexcept {
  namespace_ = plugin_namespace;
}

const char* PluginCreator::getPluginNamespace() const noexcept {
  return namespace_.data();
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
