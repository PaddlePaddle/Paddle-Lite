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

#include "driver/qualcomm_qnn/engine.h"
#include <utility>
#include "driver/qualcomm_qnn/optimizer/convert_datalayout_nchw_to_nhwc.h"
#include "driver/qualcomm_qnn/optimizer/restrict_input_output_quant_params.h"
#include "driver/qualcomm_qnn/optimizer/unpack_op_fusion.h"
#include "optimizer/convert_quantization_symm_to_asymm.h"
#include "optimizer/fuse_matmul_add_into_fully_connected.h"

namespace nnadapter {
namespace qualcomm_qnn {

static core::Argument* FindArgumentByIndex(core::Argument* arguments,
                                           int index,
                                           uint32_t count) {
  for (uint32_t i = 0; i < count; i++) {
    if (arguments[i].index == index) {
      return &arguments[i];
    }
  }
  return static_cast<core::Argument*>(nullptr);
}

Context::Context(void* device, const char* properties) : device_(device) {
  // Extract the runtime parameters from the context properties
  NNADAPTER_VLOG(1) << "properties: " << std::string(properties);
  auto key_values = GetKeyValues(properties);
  // Runtime lib
  runtime_lib_ = key_values.count(QUALCOMM_QNN_RUNTIME_LIB)
                     ? key_values[QUALCOMM_QNN_RUNTIME_LIB]
                     : GetStringFromEnv(QUALCOMM_QNN_RUNTIME_LIB);
  NNADAPTER_CHECK(!runtime_lib_.empty());
}

Program::Program(Context* context) : context_(context) {
  lib_backend_handle_ =
      dlopen(context_->RuntimeLib().c_str(), RTLD_NOW | RTLD_LOCAL);
  NNADAPTER_CHECK(lib_backend_handle_);
  qnn_interface_ = GetQnnInterface(lib_backend_handle_);
  // Init backend
  QNN_CHECK(qnn_interface_.backendInitialize(&qnn_backend_configs_));
  // Create context
  QNN_CHECK(qnn_interface_.contextCreate(&qnn_context_configs_, &qnn_context_));
}

Program::~Program() {
  Clear();
  QNN_CHECK(qnn_interface_.contextFree(qnn_context_, nullptr));
  QNN_CHECK(qnn_interface_.backendTerminate());
  dlclose(lib_backend_handle_);
}

void Program::Clear() {
  tensors_.clear();
  input_types_.clear();
  output_types_.clear();
  input_tensors_.clear();
  output_tensors_.clear();
}

int Program::Build(core::Model* model, core::Cache* cache) {
  Clear();
  if (cache->buffer.empty()) {
    NNADAPTER_CHECK_EQ(BuildFromModel(model), NNADAPTER_NO_ERROR);
  } else {
    NNADAPTER_CHECK_EQ(BuildFromCache(cache), NNADAPTER_NO_ERROR);
  }
  return NNADAPTER_NO_ERROR;
}

int Program::BuildFromModel(core::Model* model) {
  // Optimzie model
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  FuseMatMulAddIntoFullyConnected(model);
  UnpackOpFusion(model);
  ConvertQuantizationSymmToAsymm(model);
  RestrictInputOutputQuantParams(model);
  ConvertDataLayoutNCHWToNHWC(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  // Create graph
  std::string graph_name("subgraph_" +
                         std::to_string(reinterpret_cast<uint64_t>(model)));
  QNN_CHECK(qnn_interface_.graphCreate(qnn_context_,
                                       graph_name.c_str(),
                                       &qnn_graph_config_,
                                       &qnn_graph_handle_));
  Converter converter(qnn_interface_, &qnn_graph_handle_, &tensors_);
  NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);
  QNN_CHECK(qnn_interface_.graphFinalize(qnn_graph_handle_, nullptr, nullptr));
  for (auto input_operand : model->input_operands) {
    input_types_.push_back(input_operand->type);
    auto& dims = input_operand->type.dimensions;
    std::vector<uint32_t> qnn_dims;
    for (uint32_t i = 0; i < dims.count; i++) {
      qnn_dims.push_back(dims.data[i]);
    }
    input_dims_.push_back(qnn_dims);
    auto qnn_tensor = tensors_.at(input_operand).back();
    qnn_tensor.maxDimensions = input_dims_.back().data();
    qnn_tensor.currentDimensions = input_dims_.back().data();
    input_tensors_.push_back(qnn_tensor);
  }
  for (auto output_operand : model->output_operands) {
    output_types_.push_back(output_operand->type);
    auto& dims = output_operand->type.dimensions;
    std::vector<uint32_t> qnn_dims;
    for (uint32_t i = 0; i < dims.count; i++) {
      qnn_dims.push_back(dims.data[i]);
    }
    output_dims_.push_back(qnn_dims);
    auto qnn_tensor = tensors_.at(output_operand).back();
    qnn_tensor.maxDimensions = output_dims_.back().data();
    qnn_tensor.currentDimensions = output_dims_.back().data();
    output_tensors_.push_back(qnn_tensor);
  }
  return NNADAPTER_NO_ERROR;
}

int Program::BuildFromCache(core::Cache* cache) {
  NNADAPTER_LOG(FATAL) << "Build from cache is unimpleted.";
  return NNADAPTER_DEVICE_INTERNAL_ERROR;
}

int Program::CheckInputsAndOutputs(uint32_t input_count,
                                   core::Argument* input_arguments,
                                   uint32_t output_count,
                                   core::Argument* output_arguments) {
  return NNADAPTER_NO_ERROR;
}

int Program::Execute(uint32_t input_count,
                     core::Argument* input_arguments,
                     uint32_t output_count,
                     core::Argument* output_arguments) {
  int ret = CheckInputsAndOutputs(
      input_count, input_arguments, output_count, output_arguments);
  if (ret != NNADAPTER_NO_ERROR) return ret;
  // Prepare input
  for (uint32_t i = 0; i < input_count; i++) {
    auto arg = FindArgumentByIndex(input_arguments, i, input_count);
    NNADAPTER_CHECK(arg) << "Input argument " << i << " does not exist!";
    auto& type = input_types_.at(i);
    auto buffer = arg->access(arg->memory, &type, nullptr);
    auto length = GetOperandTypeBufferLength(type);
    if (IsUInt8AsymmPerLayerQuantType(type.precision)) {
      Symm2AsymmData(reinterpret_cast<const int8_t*>(buffer),
                     length,
                     type.asymm_per_layer_params.zero_point,
                     reinterpret_cast<uint8_t*>(buffer));
    }
    input_tensors_.at(i).clientBuf.data = buffer;
    input_tensors_.at(i).clientBuf.dataSize = length;
  }
  // Prepare output
  std::vector<std::pair<void*, size_t>> output_buffers(output_count);
  for (uint32_t i = 0; i < output_count; i++) {
    auto arg = FindArgumentByIndex(output_arguments, i, output_count);
    NNADAPTER_CHECK(arg) << "Output argument " << i << " does not exist!";
    auto& type = output_types_.at(i);
    auto buffer = arg->access(arg->memory, &type, nullptr);
    auto length = GetOperandTypeBufferLength(type);
    output_tensors_.at(i).clientBuf.data = buffer;
    output_tensors_.at(i).clientBuf.dataSize = length;
    output_buffers[i].first = buffer;
    output_buffers[i].second = length;
  }
  // Execute graph
  QNN_CHECK(qnn_interface_.graphExecute(qnn_graph_handle_,
                                        input_tensors_.data(),
                                        input_tensors_.size(),
                                        output_tensors_.data(),
                                        output_tensors_.size(),
                                        nullptr,
                                        nullptr));
  for (uint32_t i = 0; i < output_count; i++) {
    auto& type = output_types_[i];
    auto buffer = output_buffers[i].first;
    auto length = output_buffers[i].second;
    if (IsUInt8AsymmPerLayerQuantType(type.precision)) {
      Asymm2SymmData(reinterpret_cast<const uint8_t*>(buffer),
                     length,
                     type.asymm_per_layer_params.zero_point,
                     reinterpret_cast<int8_t*>(buffer));
    }
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace qualcomm_qnn
}  // namespace nnadapter
