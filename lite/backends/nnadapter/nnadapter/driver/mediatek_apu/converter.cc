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
#include "driver/mediatek_apu/optimizer/quantization_parameter_consistency_constraint.h"
#include "optimizer/nchw2nhwc.h"
#include "optimizer/symm2asymm.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace mediatek_apu {

Context::Context() {
  // TODO(hong19860320) create the raw context from NeuronAdapter
}

Context::~Context() {}

Program::~Program() {
  if (!execution_) {
    NeuronExecution_free_invoke(execution_);
  }
  if (!compilation_) {
    NeuronCompilation_free_invoke(compilation_);
  }
  if (!model_) {
    NeuronModel_free_invoke(model_);
  }
}

int Program::Build(hal::Model* model, hal::Cache* cache) {
  // Convert the data layout and quantization parameters of the operands in the
  // NNAdapter model
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  ConvertModelFromSymmToAsymmQuantization(model);
  ApplyQuantizationParametersConsistencyConstraint(model);
  ConvertModelFromNCHWToNHWCDataLayout(model);
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
      case NNADAPTER_RESHAPE:
        ConvertReshape(operation);
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
  std::vector<uint32_t> input_operand_indexes(input_count);
  input_zero_points_.resize(input_count);
  for (size_t i = 0; i < input_count; i++) {
    auto operand = model->input_operands[i];
    NNADAPTER_CHECK_EQ(operand->type.precision,
                       NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER);
    input_zero_points_[i] = operand->type.asymm_per_layer_params.zero_point;
    NNADAPTER_CHECK(operand_indexes_.find(operand) != operand_indexes_.end())
        << "No Neuron operand found for input operand @0x" << std::hex
        << reinterpret_cast<int64_t>(operand);
    auto index = operand_indexes_[operand];
    NNADAPTER_VLOG(5) << "Found a Neuron operand " << index
                      << " for input operand @0x" << std::hex
                      << reinterpret_cast<int64_t>(operand);
    input_operand_indexes[i] = index;
  }
  auto output_count = model->output_operands.size();
  std::vector<uint32_t> output_operand_indexes(output_count);
  output_zero_points_.resize(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto operand = model->output_operands[i];
    NNADAPTER_CHECK_EQ(operand->type.precision,
                       NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER);
    output_zero_points_[i] = operand->type.asymm_per_layer_params.zero_point;
    NNADAPTER_CHECK(operand_indexes_.find(operand) != operand_indexes_.end())
        << "No Neuron operand found for output operand @0x" << std::hex
        << reinterpret_cast<int64_t>(operand);
    auto index = operand_indexes_[operand];
    NNADAPTER_VLOG(5) << "Found a Neuron operand " << index
                      << " for output operand @0x" << std::hex
                      << reinterpret_cast<int64_t>(operand);
    output_operand_indexes[i] = index;
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
    return result;
  }
  result = NeuronModel_finish_invoke(model_);
  if (result != NEURON_NO_ERROR) {
    NeuronModel_free_invoke(model_);
    NNADAPTER_LOG(FATAL) << "Failed to finish the Neuron model(" << result
                         << ")!";
    return result;
  }
  // Build model
  result = NeuronCompilation_create_invoke(model_, &compilation_);
  if (result != NEURON_NO_ERROR) {
    NeuronModel_free_invoke(model_);
    NNADAPTER_LOG(FATAL) << "Failed to create a Neuron Compilation(" << result
                         << ")!";
    return result;
  }
  result = NeuronCompilation_finish_invoke(compilation_);
  if (result != NEURON_NO_ERROR) {
    NeuronModel_free_invoke(model_);
    NeuronCompilation_free_invoke(compilation_);
    NNADAPTER_LOG(FATAL) << "Failed to compile the Neuron Model(" << result
                         << ")!";
    return result;
  }
  // Create an execution for inference
  result = NeuronExecution_create_invoke(compilation_, &execution_);
  if (result != NEURON_NO_ERROR) {
    NeuronModel_free_invoke(model_);
    NeuronCompilation_free_invoke(compilation_);
    NNADAPTER_LOG(FATAL) << "Failed to create a Neuron Execution for inference("
                         << result << ")!";
    return result;
  }
  return NNADAPTER_NO_ERROR;
}

int Program::Execute(uint32_t input_count,
                     hal::Argument* input_arguments,
                     uint32_t output_count,
                     hal::Argument* output_arguments) {
  // Set inputs and outputs and transform the data with zero point
  for (uint32_t i = 0; i < input_count; i++) {
    auto& argument = input_arguments[i];
    auto buffer = reinterpret_cast<uint8_t*>(argument.buffer);
    auto zero_point = input_zero_points_[argument.index];
    for (int j = 0; j < argument.length; j++) {
      buffer[j] =
          static_cast<uint8_t>(static_cast<int16_t>(buffer[j]) + zero_point);
    }
    NNADAPTER_CHECK_EQ(
        NeuronExecution_setInput_invoke(
            execution_, argument.index, NULL, argument.buffer, argument.length),
        NEURON_NO_ERROR);
  }
  for (uint32_t i = 0; i < output_count; i++) {
    auto& argument = output_arguments[i];
    NNADAPTER_CHECK_EQ(
        NeuronExecution_setOutput_invoke(
            execution_, argument.index, NULL, argument.buffer, argument.length),
        NEURON_NO_ERROR);
  }
  NNADAPTER_CHECK_EQ(NeuronExecution_compute_invoke(execution_),
                     NEURON_NO_ERROR);
  for (uint32_t i = 0; i < output_count; i++) {
    auto& argument = output_arguments[i];
    auto buffer = reinterpret_cast<uint8_t*>(argument.buffer);
    auto zero_point = output_zero_points_[argument.index];
    for (int j = 0; j < argument.length; j++) {
      buffer[j] =
          static_cast<int8_t>(static_cast<int16_t>(buffer[j]) - zero_point);
    }
  }
  return NNADAPTER_NO_ERROR;
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
    converted_dimensions = ConvertDimensions(dimensions, dimension_count);
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
    auto length = PrecisionLength(precision) *
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

uint32_t Program::ConvertOperand(hal::Operand* operand) {
  if (operand_indexes_.find(operand) != operand_indexes_.end()) {
    return operand_indexes_.at(operand);
  }
  auto& type = operand->type;
  auto buffer = operand->buffer;
  auto is_constant_copy = type.lifetime == NNADAPTER_CONSTANT_COPY;
  auto is_constant_reference = type.lifetime == NNADAPTER_CONSTANT_REFERENCE;
  auto is_constant = is_constant_copy || is_constant_reference;
  uint32_t index = 0xFFFFFFFF;
  switch (type.precision) {
    case NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER: {
      if (is_constant) {
        index =
            AddQuant8ConstantOperand(reinterpret_cast<uint8_t*>(buffer),
                                     type.dimensions,
                                     type.dimension_count,
                                     type.asymm_per_layer_params.scale,
                                     type.asymm_per_layer_params.zero_point);
      } else {
        index =
            AddQuant8VariableOperand(type.dimensions,
                                     type.dimension_count,
                                     type.asymm_per_layer_params.scale,
                                     type.asymm_per_layer_params.zero_point);
        // Only mapping the temporary operand
        operand_indexes_[operand] = index;
      }
    } break;
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL: {
      NNADAPTER_CHECK(is_constant);
      index =
          AddQuant8ConstantOperand(reinterpret_cast<int8_t*>(buffer),
                                   type.dimensions,
                                   type.dimension_count,
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
                                          type.dimensions,
                                          type.dimension_count,
                                          type.symm_per_layer_params.scale);
      } else {
        index = AddInt32ConstantOperand(reinterpret_cast<int32_t*>(buffer),
                                        type.dimensions,
                                        type.dimension_count);
      }
    } break;
    default:
      NNADAPTER_LOG(FATAL) << "Missing the processing "
                           << OperationTypeToString(type.precision)
                           << " for the conversion of Neuron operands.";
      break;
  }
  NNADAPTER_CHECK_NE(index, 0xFFFFFFFF);
  return index;
}

}  // namespace mediatek_apu
}  // namespace nnadapter
