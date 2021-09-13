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

#include "driver/mediatek_apu/converter.h"
#include <algorithm>
#include <utility>
#include "driver/mediatek_apu/optimizer/propagate_quant_params.h"
#include "driver/mediatek_apu/optimizer/resolve_op_liminations.h"
#include "driver/mediatek_apu/optimizer/update_bias_quant_params_and_values.h"
#include "optimizer/nchw2nhwc.h"
#include "optimizer/symm2asymm.h"
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
  operand_index_ = 0;
  operand_buffers_.clear();
  input_types_.clear();
  output_types_.clear();
  dump_graph_path_ = "";
  dump_graph_buffer_ = nullptr;
}

int Program::Build(hal::Model* model, hal::Cache* cache) {
  Clear();
  if (model && cache->dir && cache->token) {
    dump_graph_path_ = string_format("%s/%s.dat", cache->dir, cache->token);
  }
  dump_graph_buffer_ = &cache->buffer;
  return cache->buffer.empty() ? BuildFromModel(model) : BuildFromCache(cache);
}

int Program::BuildFromModel(hal::Model* model) {
  // Convert the data layout and quantization parameters of the operands in the
  // NNAdapter model
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  ConvertQuantizationSymmToAsymm(model);
  PropagateQuantParams(model);
  UpdateBiasQuantParamsAndValues(model);
  ConvertDataLayoutNCHWToNHWC(model);
  ResolveOpLiminations(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  // Convert the NNAdapter model to Neuron model
  operand_indexes_.clear();
  operand_index_ = 0;
  uint32_t version;
  Neuron_getVersion_invoke(&version);
  NNADAPTER_VLOG(3) << "Neuron Adapter version: " << version;
  int result = NeuronModel_create_invoke(&model_);
  if (result != NEURON_NO_ERROR) {
    NNADAPTER_LOG(FATAL) << "Failed to create a Neuron Model(" << result
                         << ")!";
    return result;
  }
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
      case NNADAPTER_CONV_2D:
        ConvertConv2D(operation);
        break;
      case NNADAPTER_CONCAT:
        ConvertConcat(operation);
        break;
      case NNADAPTER_FULLY_CONNECTED:
        ConvertFullyConnected(operation);
        break;
      case NNADAPTER_AVERAGE_POOL_2D:
      case NNADAPTER_MAX_POOL_2D:
        ConvertPool2D(operation);
        break;
      case NNADAPTER_ADD:
      case NNADAPTER_SUB:
      case NNADAPTER_MUL:
      case NNADAPTER_DIV:
        ConvertElementwise(operation);
        break;
      case NNADAPTER_SOFTMAX:
        ConvertSoftmax(operation);
        break;
      case NNADAPTER_SIGMOID:
      case NNADAPTER_RELU:
      case NNADAPTER_RELU6:
      case NNADAPTER_TANH:
        ConvertActivation(operation);
        break;
      case NNADAPTER_FLATTEN:
        ConvertFlatten(operation);
        break;
      case NNADAPTER_RESHAPE:
        ConvertReshape(operation);
        break;
      case NNADAPTER_UNSQUEEZE:
        ConvertUnsqueeze(operation);
        break;
      case NNADAPTER_TRANSPOSE:
        ConvertTranspose(operation);
        break;
      default:
        NNADAPTER_LOG(FATAL) << "Unsupported operation("
                             << OperationTypeToString(operation->type)
                             << ") is found.";
        break;
    }
  }
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
      auto index = operand_indexes_[operand].back();
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

int Program::BuildFromCache(hal::Cache* cache) {
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

int Program::Execute(uint32_t input_count,
                     hal::Argument* input_arguments,
                     uint32_t output_count,
                     hal::Argument* output_arguments) {
  // Set inputs and outputs and transform the data with zero point
  for (uint32_t i = 0; i < input_count; i++) {
    auto& arg = input_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, input_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = &input_types_[arg.index];
    auto buffer = arg.access(arg.memory, type);
    NNADAPTER_CHECK(buffer);
    auto length = GetOperandTypeBufferLength(*type);
    if (IsUInt8AsymmPerLayerQuantType(type->precision)) {
      Symm2AsymmData(reinterpret_cast<const int8_t*>(buffer),
                     length,
                     type->asymm_per_layer_params.zero_point,
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
    auto buffer = arg.access(arg.memory, type);
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

uint32_t Program::GetMappedIndex(hal::Operand* operand) {
  auto it = operand_indexes_.find(operand);
  if (it != operand_indexes_.end()) {
    return it->second.back();
  }
  return INVALID_INDEX;
}

uint32_t Program::UpdateIndexMap(hal::Operand* operand, uint32_t index) {
  auto it = operand_indexes_.find(operand);
  if (it == operand_indexes_.end()) {
    auto result = operand_indexes_.insert(
        std::make_pair(operand, std::vector<uint32_t>()));
    NNADAPTER_CHECK(result.second);
    it = result.first;
  }
  it->second.push_back(index);
  return index;
}

uint32_t Program::AddOperand(int32_t* dimensions,
                             uint32_t dimension_count,
                             int precision,
                             float* quant_scales,
                             int32_t* zero_point,
                             uint32_t quant_scale_count,
                             uint32_t quant_channel_dim,
                             void* buffer) {
  NeuronOperandType type;
  memset(&type, 0, sizeof(NeuronOperandType));
  type.type = precision;
  std::vector<uint32_t> converted_dimensions;
  if (dimensions && dimension_count > 0) {
    converted_dimensions =
        ConvertToNeuronDimensions(dimensions, dimension_count);
    type.dimensions = &converted_dimensions[0];
  }
  type.dimensionCount = converted_dimensions.size();
  NeuronSymmPerChannelQuantParams symm_per_channel_quant_params;
  memset(&symm_per_channel_quant_params,
         0,
         sizeof(NeuronSymmPerChannelQuantParams));
  if (quant_scales && quant_scale_count > 0) {
    // Quant type
    if (quant_scale_count > 1) {
      // Symmetric per-channel quantization
      NNADAPTER_CHECK(!zero_point &&
                          precision == NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL ||
                      precision == NEURON_TENSOR_INT32);
      symm_per_channel_quant_params.scales = quant_scales;
      symm_per_channel_quant_params.scaleCount = quant_scale_count;
      symm_per_channel_quant_params.channelDim = quant_channel_dim;
    } else {
      type.scale = quant_scales[0];
      if (zero_point) {
        // Asymmetric per-layer quantization
        NNADAPTER_CHECK(precision == NEURON_TENSOR_QUANT8_ASYMM);
        type.zeroPoint = zero_point[0];
      } else {
        // Symmetric per-layer quantization
        NNADAPTER_CHECK(precision == NEURON_TENSOR_INT32);
        // zeroPoint = 0
      }
    }
  } else {
    // Basic type, without any quantization parameters
  }
  NNADAPTER_CHECK_EQ(NeuronModel_addOperand_invoke(model_, &type),
                     NEURON_NO_ERROR);
  auto index = operand_index_++;
  if (buffer) {
    // Constant operand
    auto length = NeuronOperandDataTypeLength(precision) *
                  ProductionOfDimensions(dimensions, dimension_count);
    NNADAPTER_CHECK_EQ(
        NeuronModel_setOperandValue_invoke(model_, index, buffer, length),
        NEURON_NO_ERROR);
  } else {
    // Variable/Input/Output operand
  }
  if (quant_scales && quant_scale_count > 1) {
    // Symmetric per-channel quantization
    NNADAPTER_CHECK_EQ(NeuronModel_setOperandSymmPerChannelQuantParams_invoke(
                           model_, index, &symm_per_channel_quant_params),
                       NEURON_NO_ERROR);
  }
  return index;
}

int Program::AddOperation(NeuronOperationType type,
                          std::vector<uint32_t>* input_indexes,
                          std::vector<uint32_t>* output_indexes) {
  return NeuronModel_addOperation_invoke(model_,
                                         type,
                                         input_indexes->size(),
                                         &((*input_indexes)[0]),
                                         output_indexes->size(),
                                         &((*output_indexes)[0]));
}

uint32_t Program::AddBool8ConstantOperand(bool value) {
  int8_t int8_value = value ? static_cast<int8_t>(1) : static_cast<int8_t>(0);
  return AddOperand(
      nullptr, 0, NEURON_BOOL, nullptr, nullptr, 0, 0, &int8_value);
}

uint32_t Program::AddInt32ConstantOperand(int32_t value) {
  return AddOperand(nullptr, 0, NEURON_INT32, nullptr, nullptr, 0, 0, &value);
}

uint32_t Program::AddFloat32ConstantOperand(float value) {
  return AddOperand(nullptr, 0, NEURON_FLOAT32, nullptr, nullptr, 0, 0, &value);
}

uint32_t Program::AddInt32ConstantOperand(int32_t* values,
                                          uint32_t num_values) {
  std::vector<int32_t> dimensions({static_cast<int32_t>(num_values)});
  return AddOperand(&dimensions[0],
                    dimensions.size(),
                    NEURON_TENSOR_INT32,
                    nullptr,
                    nullptr,
                    0,
                    0,
                    values);
}

uint32_t Program::AddFloat32ConstantOperand(float* values,
                                            uint32_t num_values) {
  std::vector<int32_t> dimensions({static_cast<int32_t>(num_values)});
  return AddOperand(&dimensions[0],
                    dimensions.size(),
                    NEURON_TENSOR_FLOAT32,
                    nullptr,
                    nullptr,
                    0,
                    0,
                    values);
}

uint32_t Program::AddInt32ConstantOperand(int32_t* values,
                                          int32_t* dimensions,
                                          uint32_t dimension_count) {
  return AddOperand(dimensions,
                    dimension_count,
                    NEURON_TENSOR_INT32,
                    nullptr,
                    nullptr,
                    0,
                    0,
                    values);
}

uint32_t Program::AddFloat32ConstantOperand(float* values,
                                            int32_t* dimensions,
                                            uint32_t dimension_count) {
  return AddOperand(dimensions,
                    dimension_count,
                    NEURON_TENSOR_FLOAT32,
                    nullptr,
                    nullptr,
                    0,
                    0,
                    values);
}

uint32_t Program::AddQuant8ConstantOperand(int8_t* values,
                                           int32_t* dimensions,
                                           uint32_t dimension_count,
                                           float* quant_scales,
                                           uint32_t quant_scale_count,
                                           uint32_t quant_channel_dim) {
  return AddOperand(dimensions,
                    dimension_count,
                    NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL,
                    quant_scales,
                    nullptr,
                    quant_scale_count,
                    quant_channel_dim,
                    values);
}

uint32_t Program::AddQuant8ConstantOperand(uint8_t* values,
                                           int32_t* dimensions,
                                           uint32_t dimension_count,
                                           float quant_scale,
                                           int32_t zero_point) {
  return AddOperand(dimensions,
                    dimension_count,
                    NEURON_TENSOR_QUANT8_ASYMM,
                    &quant_scale,
                    &zero_point,
                    1,
                    0,
                    values);
}

uint32_t Program::AddQuant32ConstantOperand(int32_t* values,
                                            int32_t* dimensions,
                                            uint32_t dimension_count,
                                            float quant_scale) {
  return AddOperand(dimensions,
                    dimension_count,
                    NEURON_TENSOR_INT32,
                    &quant_scale,
                    nullptr,
                    1,
                    0,
                    values);
}

uint32_t Program::AddQuant8VariableOperand(int32_t* dimensions,
                                           uint32_t dimension_count,
                                           float quant_scale,
                                           int32_t zero_point) {
  return AddOperand(dimensions,
                    dimension_count,
                    NEURON_TENSOR_QUANT8_ASYMM,
                    &quant_scale,
                    &zero_point,
                    1,
                    0);
}

uint32_t Program::ConvertOperand(hal::Operand* operand,
                                 std::vector<int32_t> dimensions) {
  auto& type = operand->type;
  auto buffer = operand->buffer;
  if (dimensions.empty()) {
    for (uint32_t i = 0; i < type.dimension_count; i++) {
      dimensions.push_back(type.dimensions[i]);
    }
  }
  auto is_constant_copy = type.lifetime == NNADAPTER_CONSTANT_COPY;
  auto is_constant_reference = type.lifetime == NNADAPTER_CONSTANT_REFERENCE;
  auto is_constant = is_constant_copy || is_constant_reference;
  uint32_t index = INVALID_INDEX;
  switch (type.precision) {
    case NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER: {
      if (is_constant) {
        index =
            AddQuant8ConstantOperand(reinterpret_cast<uint8_t*>(buffer),
                                     &dimensions[0],
                                     dimensions.size(),
                                     type.asymm_per_layer_params.scale,
                                     type.asymm_per_layer_params.zero_point);
      } else {
        index =
            AddQuant8VariableOperand(&dimensions[0],
                                     dimensions.size(),
                                     type.asymm_per_layer_params.scale,
                                     type.asymm_per_layer_params.zero_point);
      }
    } break;
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL: {
      NNADAPTER_CHECK(is_constant);
      index =
          AddQuant8ConstantOperand(reinterpret_cast<int8_t*>(buffer),
                                   &dimensions[0],
                                   dimensions.size(),
                                   type.symm_per_channel_params.scales,
                                   type.symm_per_channel_params.scale_count,
                                   type.symm_per_channel_params.channel_dim);
    } break;
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER:
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL: {
      // Only for bias
      NNADAPTER_CHECK(is_constant);
      if (type.precision == NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER) {
        index = AddQuant32ConstantOperand(reinterpret_cast<int32_t*>(buffer),
                                          &dimensions[0],
                                          dimensions.size(),
                                          type.symm_per_layer_params.scale);
      } else {
        index = AddInt32ConstantOperand(reinterpret_cast<int32_t*>(buffer),
                                        &dimensions[0],
                                        dimensions.size());
      }
    } break;
    default:
      NNADAPTER_LOG(FATAL) << "Missing the processing "
                           << OperationTypeToString(type.precision)
                           << " for the conversion of Neuron operands.";
      break;
  }
  NNADAPTER_CHECK_NE(index, INVALID_INDEX);
  UpdateIndexMap(operand, index);
  return index;
}

}  // namespace mediatek_apu
}  // namespace nnadapter
