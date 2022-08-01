// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/mediatek_apu/engine.h"
#include <algorithm>
#include <utility>
#include "driver/mediatek_apu/converter/converter.h"
#include "driver/mediatek_apu/optimizer/resolve_operation_liminations.h"
#include "driver/mediatek_apu/optimizer/restrict_input_output_quant_params.h"
#include "optimizer/convert_datalayout_nchw_to_nhwc.h"
#include "optimizer/convert_quantization_symm_to_asymm.h"
#include "optimizer/fuse_conv2d_activation_into_conv2d.h"
#include "optimizer/fuse_conv2d_add_into_conv2d.h"
#include "optimizer/fuse_conv2d_batch_norm_into_conv2d.h"
#include "optimizer/fuse_matmul_add_into_fully_connected.h"
#include "optimizer/fuse_reshape_transpose_reshape_into_channel_shuffle.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace mediatek_apu {

Context::Context(void* device, const char* properties) : device_(device) {
  // TODO(hong19860320) create the raw context from NeuronAdapter
}

Context::~Context() {}

Program::~Program() { Clear(); }

void Program::Clear() {
  if (execution_) {
    NeuronExecution_free_invoke(execution_);
    execution_ = nullptr;
  }
  if (compilation_) {
    NeuronCompilation_free_invoke(compilation_);
    compilation_ = nullptr;
  }
  if (model_) {
    NeuronModel_free_invoke(model_);
    model_ = nullptr;
  }
  operand_indexes_.clear();
  operand_buffers_.clear();
  input_types_.clear();
  output_types_.clear();
  dump_graph_path_ = "";
  dump_graph_buffer_ = nullptr;
}

int Program::Build(core::Model* model, core::Cache* cache) {
  Clear();
  if (model && cache->dir && cache->token) {
    dump_graph_path_ = string_format("%s/%s.dat", cache->dir, cache->token);
  }
  dump_graph_buffer_ = &cache->buffer;
  return cache->buffer.empty() ? BuildFromModel(model) : BuildFromCache(cache);
}

int Program::BuildFromModel(core::Model* model) {
  // Convert the data layout and quantization parameters of the operands in the
  // NNAdapter model
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  FuseConv2DBatchNormIntoConv2D(model);
  FuseConv2DAddIntoConv2D(model);
  FuseConv2DActivationIntoConv2D(model);
  FuseMatMulAddIntoFullyConnected(model);
  FuseReshapeTransposeReshapeIntoChannelShuffle(model);
  ConvertQuantizationSymmToAsymm(model);
  RestrictInputOutputQuantParams(model);
  ConvertDataLayoutNCHWToNHWC(model);
  ResolveOperationLiminations(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  // Convert the NNAdapter model to Neuron model
  operand_indexes_.clear();
  uint32_t version;
  Neuron_getVersion_invoke(&version);
  NNADAPTER_VLOG(3) << "Neuron Adapter version: " << version;
  int result = NeuronModel_create_invoke(&model_);
  if (result != NEURON_NO_ERROR) {
    NNADAPTER_LOG(FATAL) << "Failed to create a Neuron Model(" << result
                         << ")!";
    return result;
  }
  Converter converter(model_, &operand_indexes_);
  NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);
  // Indentify the inputs and outputs
  auto input_count = model->input_operands.size();
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  std::vector<uint32_t> input_operand_indexes(input_count);
  if (input_count > 0) {
    input_operand_indexes.resize(input_count);
    input_types_.resize(input_count);
    for (size_t i = 0; i < input_count; i++) {
      auto operand = model->input_operands[i];
      NNADAPTER_CHECK(operand_indexes_.find(operand) != operand_indexes_.end())
          << "No Neuron operand found for input operand @0x" << std::hex
          << reinterpret_cast<int64_t>(operand);
      auto index = operand_indexes_[operand].front();
      NNADAPTER_CHECK_NE(index, INVALID_INDEX);
      NNADAPTER_VLOG(5) << "Found a Neuron operand " << index
                        << " for input operand @0x" << std::hex
                        << reinterpret_cast<int64_t>(operand);
      input_operand_indexes[i] = index;
      input_types_[i] = operand->type;
    }
  }
  auto output_count = model->output_operands.size();
  NNADAPTER_VLOG(3) << "Model output count: " << output_count;
  std::vector<uint32_t> output_operand_indexes(output_count);
  output_types_.resize(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto operand = model->output_operands[i];
    NNADAPTER_CHECK(operand_indexes_.find(operand) != operand_indexes_.end())
        << "No Neuron operand found for output operand @0x" << std::hex
        << reinterpret_cast<int64_t>(operand);
    auto index = operand_indexes_[operand].back();
    NNADAPTER_CHECK_NE(index, INVALID_INDEX);
    NNADAPTER_VLOG(5) << "Found a Neuron operand " << index
                      << " for output operand @0x" << std::hex
                      << reinterpret_cast<int64_t>(operand);
    output_operand_indexes[i] = index;
    output_types_[i] = operand->type;
  }
  result =
      NeuronModel_identifyInputsAndOutputs_invoke(model_,
                                                  input_operand_indexes.size(),
                                                  &input_operand_indexes[0],
                                                  output_operand_indexes.size(),
                                                  &output_operand_indexes[0]);
  if (result != NEURON_NO_ERROR) {
    NeuronModel_free_invoke(model_);
    NNADAPTER_LOG(FATAL) << "Failed to identify the inputs and outputs("
                         << result << ")!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  result = NeuronModel_finish_invoke(model_);
  if (result != NEURON_NO_ERROR) {
    NeuronModel_free_invoke(model_);
    NNADAPTER_LOG(FATAL) << "Failed to finish the Neuron model(" << result
                         << ")!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  // Build model
  result = NeuronCompilation_create_invoke(model_, &compilation_);
  if (result != NEURON_NO_ERROR) {
    NeuronModel_free_invoke(model_);
    NNADAPTER_LOG(FATAL) << "Failed to create a Neuron Compilation(" << result
                         << ")!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  result = NeuronCompilation_finish_invoke(compilation_);
  if (result != NEURON_NO_ERROR) {
    NeuronModel_free_invoke(model_);
    NeuronCompilation_free_invoke(compilation_);
    NNADAPTER_LOG(FATAL) << "Failed to compile the Neuron Model(" << result
                         << ")!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  if (!dump_graph_path_.empty()) {
    size_t dump_graph_size = 0;
    result = NeuronCompilation_getCompiledNetworkSize_invoke(compilation_,
                                                             &dump_graph_size);
    if (result == NEURON_NO_ERROR && dump_graph_size > 0) {
      dump_graph_buffer_->resize(dump_graph_size);
      result = NeuronCompilation_storeCompiledNetwork_invoke(
          compilation_, dump_graph_buffer_->data(), dump_graph_size);
      if (result == NEURON_NO_ERROR) {
        NNADAPTER_LOG(INFO)
            << "Serialize the Neuron compiled network into buffer success.";
      } else {
        NNADAPTER_LOG(WARNING)
            << "Failed to serialize the Neuron compiled network into buffer!";
      }
    } else {
      NNADAPTER_LOG(WARNING)
          << "Failed to query the size of the Neuron compiled network!";
    }
    dump_graph_path_ = "";
  }
  // Create an execution for inference
  result = NeuronExecution_create_invoke(compilation_, &execution_);
  if (result != NEURON_NO_ERROR) {
    NeuronModel_free_invoke(model_);
    NeuronCompilation_free_invoke(compilation_);
    NNADAPTER_LOG(FATAL) << "Failed to create a Neuron Execution for inference("
                         << result << ")!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  NNADAPTER_VLOG(3) << "Build success.";
  return NNADAPTER_NO_ERROR;
}

int Program::BuildFromCache(core::Cache* cache) {
  uint32_t version;
  Neuron_getVersion_invoke(&version);
  NNADAPTER_VLOG(3) << "Neuron Adapter version: " << version;
  int result = NeuronModel_restoreFromCompiledNetwork_invoke(
      &model_, &compilation_, cache->buffer.data(), cache->buffer.size());
  if (result != NEURON_NO_ERROR) {
    NNADAPTER_LOG(FATAL)
        << "Failed to restore the Neuron compiled network from buffer!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  auto input_count = cache->input_types.size();
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  input_types_ = cache->input_types;
  auto output_count = cache->output_types.size();
  NNADAPTER_VLOG(3) << "Model output count: " << output_count;
  NNADAPTER_CHECK_GT(output_count, 0);
  output_types_ = cache->output_types;
  // Create an execution for inference
  result = NeuronExecution_create_invoke(compilation_, &execution_);
  if (result != NEURON_NO_ERROR) {
    NeuronModel_free_invoke(model_);
    NeuronCompilation_free_invoke(compilation_);
    NNADAPTER_LOG(FATAL) << "Failed to create a Neuron Execution for inference("
                         << result << ")!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  NNADAPTER_VLOG(3) << "Build success.";
  return NNADAPTER_NO_ERROR;
}

int Program::CheckInputsAndOutputs(uint32_t input_count,
                                   core::Argument* input_arguments,
                                   uint32_t output_count,
                                   core::Argument* output_arguments) {
  // Check inputs
  for (uint32_t i = 0; i < input_count; i++) {
    // Get actual type
    auto& arg = input_arguments[i];
    NNAdapterOperandType type;
    arg.access(arg.memory, &type, nullptr);
    // Check dimensions count
    uint32_t count = type.dimensions.count;
    int32_t* data = type.dimensions.data;
    auto& src_dimensions = input_types_[i].dimensions;
    int32_t* src_data = src_dimensions.data;
    if (count != src_dimensions.count) {
      return NNADAPTER_INVALID_DIMENSIONS;
    }
    // Check dimensions data
    for (uint32_t j = 0; j < count; j++) {
      if (data[j] != src_data[j]) {
        return NNADAPTER_INVALID_DIMENSIONS;
      }
    }
  }
  return NNADAPTER_NO_ERROR;
}

int Program::Execute(uint32_t input_count,
                     core::Argument* input_arguments,
                     uint32_t output_count,
                     core::Argument* output_arguments) {
  int ret = CheckInputsAndOutputs(
      input_count, input_arguments, output_count, output_arguments);
  if (ret != NNADAPTER_NO_ERROR) return ret;
  // Set inputs and outputs and transform the data with zero point
  for (uint32_t i = 0; i < input_count; i++) {
    auto& arg = input_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, input_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = input_types_[arg.index];
    auto buffer = arg.access(arg.memory, &type, nullptr);
    NNADAPTER_CHECK(buffer);
    auto length = GetOperandTypeBufferLength(type);
    if (IsUInt8AsymmPerLayerQuantType(type.precision)) {
      Symm2AsymmData(reinterpret_cast<const int8_t*>(buffer),
                     length,
                     type.asymm_per_layer_params.zero_point,
                     reinterpret_cast<uint8_t*>(buffer));
    }
    NNADAPTER_CHECK_EQ(NeuronExecution_setInput_invoke(
                           execution_, arg.index, NULL, buffer, length),
                       NEURON_NO_ERROR);
  }
  std::vector<std::pair<void*, size_t>> output_buffers(output_count);
  for (uint32_t i = 0; i < output_count; i++) {
    auto& arg = output_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, output_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = &output_types_[arg.index];
    // TODO(hong19860320) Get the dimensions of the outputs from imgdnn
    // according to the dynamic dimensions of the inputs, fill them to 'type'
    // and call the 'access' function to re-allocate the host output memory
    auto buffer = arg.access(arg.memory, type, nullptr);
    NNADAPTER_CHECK(buffer);
    auto length = GetOperandTypeBufferLength(*type);
    NNADAPTER_CHECK_EQ(NeuronExecution_setOutput_invoke(
                           execution_, arg.index, NULL, buffer, length),
                       NEURON_NO_ERROR);
    output_buffers[arg.index].first = buffer;
    output_buffers[arg.index].second = length;
  }
  auto start_time = GetCurrentUS();
  NNADAPTER_CHECK_EQ(NeuronExecution_compute_invoke(execution_),
                     NEURON_NO_ERROR);
  NNADAPTER_VLOG(3) << "Process cost " << GetCurrentUS() - start_time << " us";
  for (uint32_t i = 0; i < output_count; i++) {
    auto type = &output_types_[i];
    auto buffer = output_buffers[i].first;
    auto length = output_buffers[i].second;
    if (IsUInt8AsymmPerLayerQuantType(type->precision)) {
      Asymm2SymmData(reinterpret_cast<const uint8_t*>(buffer),
                     length,
                     type->asymm_per_layer_params.zero_point,
                     reinterpret_cast<int8_t*>(buffer));
    }
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace mediatek_apu
}  // namespace nnadapter
