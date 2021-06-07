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

#include "converter.h"  // NOLINT
#include <memory>
#include <vector>
#include "../../nnadapter_common.h"     // NOLINT
#include "../../nnadapter_optimizer.h"  // NOLINT
#include "optimizer.h"                  // NOLINT
#include "utility.h"                    // NOLINT

namespace nnadapter {
namespace driver {
namespace mediatek_apu {

Context::Context() {
  // TODO(hong19860320) create the raw context from rknpu ddk driver
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

int Program::Build(driver::Model* model, driver::Cache* cache) {
  // Convert the data layout and quantization parameters of the operands in the
  // NNAdapter model
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << driver::Visualize(model);
  ConvertModelFromSymmToAsymmQuantization(model);
  ApplyConstraintsToQuantizationParameters(model);
  ConvertModelFromNCHWToNHWCDataLayout(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl
                    << driver::Visualize(model);
  // Convert the NNAdapter model to Neuron model
  operand_indexes_.clear();
  operand_index_ = 0;
  uint32_t version;
  Neuron_getVersion_invoke(&version);
  NNADAPTER_VLOG(3) << "Neuron Adapter version: " << version;
  int result = NeuronModel_create_invoke(&model_);
  if (result != NEURON_NO_ERROR) {
    NNADAPTER_LOG(WARNING) << "Failed to create a Neuron Model(" << result
                           << ")!";
    return result;
  }
  std::vector<Operation*> operations =
      driver::SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
      case NNADAPTER_CONV_2D:
        ConvertConv2D(operation);
        break;
      case NNADAPTER_FULLY_CONNECTED:
        ConvertFullyConnected(operation);
        break;
      case NNADAPTER_AVERAGE_POOL_2D:
      case NNADAPTER_MAX_POOL_2D:
        ConvertAverageAndMaxPool2D(operation);
        break;
      case NNADAPTER_ADD:
      case NNADAPTER_SUB:
      case NNADAPTER_MUL:
      case NNADAPTER_DIV:
        ConvertElementwiseBinaryOperations(operation);
        break;
      case NNADAPTER_SOFTMAX:
        ConvertSoftmax(operation);
        break;
      case NNADAPTER_SIGMOID:
      case NNADAPTER_RELU:
      case NNADAPTER_RELU6:
      case NNADAPTER_TANH:
        ConvertActivationUnaryOperations(operation);
        break;
      case NNADAPTER_TRANSPOSE:
        ConvertTranspose(operation);
        break;
      default:
        NNADAPTER_LOG(ERROR) << "Unsupported operation("
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
    NNADAPTER_LOG(WARNING) << "Failed to identify the inputs and outputs("
                           << result << ")!";
    return result;
  }
  result = NeuronModel_finish_invoke(model_);
  if (result != NEURON_NO_ERROR) {
    NeuronModel_free_invoke(model_);
    NNADAPTER_LOG(WARNING) << "Failed to finish the Neuron model(" << result
                           << ")!";
    return result;
  }
  // Build model
  result = NeuronCompilation_create_invoke(model_, &compilation_);
  if (result != NEURON_NO_ERROR) {
    NeuronModel_free_invoke(model_);
    NNADAPTER_LOG(WARNING) << "Failed to create a Neuron Compilation(" << result
                           << ")!";
    return result;
  }
  result = NeuronCompilation_finish_invoke(compilation_);
  if (result != NEURON_NO_ERROR) {
    NeuronModel_free_invoke(model_);
    NeuronCompilation_free_invoke(compilation_);
    NNADAPTER_LOG(WARNING) << "Failed to compile the Neuron Model(" << result
                           << ")!";
    return result;
  }
  // Create an execution for inference
  result = NeuronExecution_create_invoke(compilation_, &execution_);
  if (result != NEURON_NO_ERROR) {
    NeuronModel_free_invoke(model_);
    NeuronCompilation_free_invoke(compilation_);
    NNADAPTER_LOG(WARNING)
        << "Failed to create a Neuron Execution for inference(" << result
        << ")!";
    return result;
  }
  return NNADAPTER_NO_ERROR;
}

int Program::Execute(uint32_t input_count,
                     driver::Argument* input_arguments,
                     uint32_t output_count,
                     driver::Argument* output_arguments) {
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

uint32_t Program::ConvertOperand(driver::Operand* operand) {
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
      NNADAPTER_LOG(ERROR) << "Missing the processing "
                           << OperationTypeToString(type.precision)
                           << " for the conversion of Neuron operands.";
      break;
  }
  NNADAPTER_CHECK_NE(index, 0xFFFFFFFF);
  return index;
}

int Program::ConvertConv2D(driver::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 13);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input"
                    << " : " << OperandToString(input_operand);
  auto input_channel_size = input_operand->type.dimensions[3];
  // Filter
  auto filter_operand = input_operands[1];
  NNADAPTER_VLOG(5) << "filter: " << OperandToString(filter_operand);
  NNADAPTER_CHECK(filter_operand && filter_operand->buffer);
  auto filter_height = filter_operand->type.dimensions[1];
  auto filter_width = filter_operand->type.dimensions[2];
  // Bias
  auto bias_operand = input_operands[2];
  NNADAPTER_VLOG(5) << "bias: " << OperandToString(bias_operand);
  NNADAPTER_CHECK(bias_operand && bias_operand->buffer);
  // Paddings
  auto padding_width_left =
      *reinterpret_cast<int32_t*>(input_operands[3]->buffer);
  auto padding_width_right =
      *reinterpret_cast<int32_t*>(input_operands[4]->buffer);
  auto padding_height_top =
      *reinterpret_cast<int32_t*>(input_operands[5]->buffer);
  auto padding_height_bottom =
      *reinterpret_cast<int32_t*>(input_operands[6]->buffer);
  NNADAPTER_VLOG(5) << "paddings=[" << padding_width_left << ","
                    << padding_width_right << "," << padding_height_top << ","
                    << padding_height_bottom << "]";
  // Strides
  auto stride_width = *reinterpret_cast<int32_t*>(input_operands[7]->buffer);
  auto stride_height = *reinterpret_cast<int32_t*>(input_operands[8]->buffer);
  NNADAPTER_VLOG(5) << "strides=[" << stride_width << "," << stride_height
                    << "]";
  // Group
  auto group = *reinterpret_cast<int32_t*>(input_operands[9]->buffer);
  NNADAPTER_VLOG(5) << "group=" << group;
  // Fuse code
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[10]->buffer);
  NNADAPTER_VLOG(5) << "fuse_code=" << fuse_code;
  // Dilations
  auto dilation_width = *reinterpret_cast<int32_t*>(input_operands[11]->buffer);
  auto dilation_height =
      *reinterpret_cast<int32_t*>(input_operands[12]->buffer);
  NNADAPTER_CHECK_EQ(dilation_width, 1);
  NNADAPTER_CHECK_EQ(dilation_height, 1);
  NNADAPTER_VLOG(5) << "dilations=[" << dilation_width << "," << dilation_height
                    << "]";
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  // Check depthwise mode
  bool is_depthwise_mode = input_channel_size == group;
  NNADAPTER_VLOG(5) << "depthwise mode(" << is_depthwise_mode << ").";

  // Convert to Neuron operands and operations
  auto input_index = ConvertOperand(input_operand);
  bool is_per_channel = filter_operand->type.precision ==
                        NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL;
  NNADAPTER_CHECK(filter_operand->type.precision ==
                      NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER ||
                  is_per_channel);
  NNADAPTER_VLOG(5) << "is_per_channel:" << is_per_channel;
  auto filter_index = ConvertOperand(filter_operand);
  NNADAPTER_VLOG(5) << "filter_index:" << filter_index;
  NNADAPTER_VLOG(5) << "bias_buffer:" << std::hex << bias_operand->buffer;
  auto bias_index = ConvertOperand(bias_operand);
  NNADAPTER_VLOG(5) << "bias_index:" << bias_index;
  auto padding_width_left_index = AddInt32ConstantOperand(padding_width_left);
  auto padding_width_right_index = AddInt32ConstantOperand(padding_width_right);
  auto padding_height_top_index = AddInt32ConstantOperand(padding_height_top);
  auto padding_height_bottom_index =
      AddInt32ConstantOperand(padding_height_bottom);
  auto stride_width_index = AddInt32ConstantOperand(stride_width);
  auto stride_height_index = AddInt32ConstantOperand(stride_height);
  auto fuse_code_index = AddInt32ConstantOperand(ConvertFuseCode(fuse_code));
  auto output_index = ConvertOperand(output_operand);
  std::vector<uint32_t> input_indexes = {input_index,
                                         filter_index,
                                         bias_index,
                                         padding_width_left_index,
                                         padding_width_right_index,
                                         padding_height_top_index,
                                         padding_height_bottom_index,
                                         stride_width_index,
                                         stride_height_index};
  std::vector<uint32_t> output_indexes = {output_index};
  if (is_depthwise_mode) {
    int32_t multiplier =
        filter_operand->type.dimensions[3] / input_channel_size;
    auto multiplier_index = AddInt32ConstantOperand(multiplier);
    input_indexes.push_back(multiplier_index);
    input_indexes.push_back(fuse_code_index);
    NNADAPTER_CHECK_EQ(
        AddOperation(NEURON_DEPTHWISE_CONV_2D, &input_indexes, &output_indexes),
        NEURON_NO_ERROR);
  } else {
    input_indexes.push_back(fuse_code_index);
    NNADAPTER_CHECK_EQ(
        AddOperation(NEURON_CONV_2D, &input_indexes, &output_indexes),
        NEURON_NO_ERROR);
  }
  return NNADAPTER_NO_ERROR;
}

int Program::ConvertFullyConnected(driver::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 4);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  // Weight
  auto weight_operand = input_operands[1];
  NNADAPTER_VLOG(5) << "weight: " << OperandToString(weight_operand);
  NNADAPTER_CHECK_EQ(weight_operand->type.dimension_count, 2);
  auto num_units = weight_operand->type.dimensions[0];
  auto input_size = weight_operand->type.dimensions[1];
  NNADAPTER_LOG(INFO) << "weight dims[" << num_units << "," << input_size
                      << "]";
  // Bias
  auto bias_operand = input_operands[2];
  NNADAPTER_VLOG(5) << "bias: " << OperandToString(bias_operand);
  NNADAPTER_CHECK_EQ(bias_operand->type.dimension_count, 1);
  NNADAPTER_CHECK_EQ(num_units, bias_operand->type.dimensions[0]);
  NNADAPTER_LOG(INFO) << "bias dims[" << bias_operand->type.dimensions[0]
                      << "]";
  // Fuse code
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[3]->buffer);
  NNADAPTER_VLOG(5) << "fuse_code=" << fuse_code;
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to Neuron operands and operations
  auto input_index = ConvertOperand(input_operand);
  auto weight_index = ConvertOperand(weight_operand);
  auto bias_index = ConvertOperand(bias_operand);
  auto fuse_code_index = AddInt32ConstantOperand(ConvertFuseCode(fuse_code));
  auto output_index = ConvertOperand(output_operand);
  std::vector<uint32_t> input_indexes = {
      input_index, weight_index, bias_index, fuse_code_index};
  std::vector<uint32_t> output_indexes = {output_index};
  NNADAPTER_CHECK_EQ(
      AddOperation(NEURON_FULLY_CONNECTED, &input_indexes, &output_indexes),
      NEURON_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

int Program::ConvertAverageAndMaxPool2D(driver::Operation* operation) {
  int result = 0;
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 12);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  // Paddings
  auto padding_width_left =
      *reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  auto padding_width_right =
      *reinterpret_cast<int32_t*>(input_operands[2]->buffer);
  auto padding_height_top =
      *reinterpret_cast<int32_t*>(input_operands[3]->buffer);
  auto padding_height_bottom =
      *reinterpret_cast<int32_t*>(input_operands[4]->buffer);
  NNADAPTER_VLOG(5) << "paddings=[" << padding_width_left << ","
                    << padding_width_right << "," << padding_height_top << ","
                    << padding_height_bottom << "]";
  // Strides
  auto stride_width = *reinterpret_cast<int32_t*>(input_operands[5]->buffer);
  auto stride_height = *reinterpret_cast<int32_t*>(input_operands[6]->buffer);
  NNADAPTER_VLOG(5) << "strides=[" << stride_width << "," << stride_height
                    << "]";
  // Filter
  auto filter_width = *reinterpret_cast<int32_t*>(input_operands[7]->buffer);
  auto filter_height = *reinterpret_cast<int32_t*>(input_operands[8]->buffer);
  NNADAPTER_VLOG(5) << "filter=[" << filter_width << "," << filter_height
                    << "]";
  bool global_pooling = filter_width == input_operand->type.dimensions[3] &&
                        filter_height == input_operand->type.dimensions[2];
  // Fuse code
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[9]->buffer);
  NNADAPTER_VLOG(5) << "fuse_code=" << fuse_code;
  // Ceil mode
  bool ceil_mode = *reinterpret_cast<int8_t*>(input_operands[10]->buffer);
  NNADAPTER_VLOG(5) << "ceil_mode=" << ceil_mode;
  NNADAPTER_CHECK_EQ(ceil_mode, false)
      << "Neuron Aadapter doesn't suppport ceil_mode=true";
  // Count include pad
  bool count_include_pad =
      *reinterpret_cast<int8_t*>(input_operands[11]->buffer);
  NNADAPTER_VLOG(5) << "count_include_pad=" << count_include_pad;
  NNADAPTER_CHECK_EQ(count_include_pad, false)
      << "Neuron Aadapter doesn't suppport count_include_pad=true";
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to Neuron operands and operations
  auto input_index = ConvertOperand(input_operand);
  auto padding_width_left_index = AddInt32ConstantOperand(padding_width_left);
  auto padding_width_right_index = AddInt32ConstantOperand(padding_width_right);
  auto padding_height_top_index = AddInt32ConstantOperand(padding_height_top);
  auto padding_height_bottom_index =
      AddInt32ConstantOperand(padding_height_bottom);
  auto stride_width_index = AddInt32ConstantOperand(stride_width);
  auto stride_height_index = AddInt32ConstantOperand(stride_height);
  auto filter_width_index = AddInt32ConstantOperand(filter_width);
  auto filter_height_index = AddInt32ConstantOperand(filter_height);
  auto fuse_code_index = AddInt32ConstantOperand(ConvertFuseCode(fuse_code));
  auto output_index = ConvertOperand(output_operand);
  std::vector<uint32_t> input_indexes = {input_index,
                                         padding_width_left_index,
                                         padding_width_right_index,
                                         padding_height_top_index,
                                         padding_height_bottom_index,
                                         stride_width_index,
                                         stride_height_index,
                                         filter_width_index,
                                         filter_height_index,
                                         fuse_code_index};
  std::vector<uint32_t> output_indexes = {output_index};
  NeuronOperationType op_type;
  if (operation->type == NNADAPTER_AVERAGE_POOL_2D) {
    op_type = NEURON_AVERAGE_POOL_2D;
  } else if (operation->type == NNADAPTER_MAX_POOL_2D) {
    op_type = NEURON_MAX_POOL_2D;
  } else {
    NNADAPTER_LOG(ERROR) << "Unsupported pooling operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  NNADAPTER_CHECK_EQ(AddOperation(op_type, &input_indexes, &output_indexes),
                     NEURON_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

int Program::ConvertElementwiseBinaryOperations(driver::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 3);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input0
  auto input0_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input0: " << OperandToString(input0_operand);
  // Input1
  auto input1_operand = input_operands[1];
  NNADAPTER_VLOG(5) << "input1: " << OperandToString(input1_operand);
  // Fuse code
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[2]->buffer);
  NNADAPTER_VLOG(5) << "fuse_code=" << fuse_code;
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to Neuron operands and operations
  auto input0_index = ConvertOperand(input0_operand);
  auto input1_index = ConvertOperand(input1_operand);
  auto fuse_code_index = AddInt32ConstantOperand(ConvertFuseCode(fuse_code));
  auto output_index = ConvertOperand(output_operand);
  NeuronOperationType op_type;
  if (operation->type == NNADAPTER_ADD) {
    op_type = NEURON_ADD;
  } else if (operation->type == NNADAPTER_SUB) {
    op_type = NEURON_SUB;
  } else if (operation->type == NNADAPTER_MUL) {
    op_type = NEURON_MUL;
  } else if (operation->type == NNADAPTER_DIV) {
    op_type = NEURON_DIV;
  } else {
    NNADAPTER_LOG(ERROR) << "Unsupported element-wise binary operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  std::vector<uint32_t> input_indexes = {
      input0_index, input1_index, fuse_code_index};
  std::vector<uint32_t> output_indexes = {output_index};
  NNADAPTER_CHECK_EQ(AddOperation(op_type, &input_indexes, &output_indexes),
                     NEURON_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

int Program::ConvertSoftmax(driver::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  // Axis
  auto axis = *reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  if (axis < 0) {
    axis += input_operand->type.dimension_count;
  }
  NNADAPTER_VLOG(5) << "axis=" << axis;
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to Neuron operands and operations
  auto input_index = ConvertOperand(input_operand);
  auto beta_index = AddFloat32ConstantOperand(0.5f);
  auto axis_index = AddInt32ConstantOperand(axis);
  auto output_index = ConvertOperand(output_operand);
  std::vector<uint32_t> input_indexes = {input_index, beta_index, axis_index};
  std::vector<uint32_t> output_indexes = {output_index};
  NNADAPTER_CHECK_EQ(
      AddOperation(NEURON_SOFTMAX, &input_indexes, &output_indexes),
      NEURON_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

int Program::ConvertActivationUnaryOperations(driver::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 1);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to Neuron operands and operations
  auto input_index = ConvertOperand(input_operand);
  auto output_index = ConvertOperand(output_operand);
  NeuronOperationType op_type;
  if (operation->type == NNADAPTER_SIGMOID) {
    op_type = NEURON_LOGISTIC;
  } else if (operation->type == NNADAPTER_RELU) {
    op_type = NEURON_RELU;
  } else if (operation->type == NNADAPTER_RELU6) {
    op_type = NEURON_RELU6;
  } else if (operation->type == NNADAPTER_TANH) {
    op_type = NEURON_TANH;
  } else {
    NNADAPTER_LOG(ERROR) << "Unsupported activation unary operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  std::vector<uint32_t> input_indexes = {input_index};
  std::vector<uint32_t> output_indexes = {output_index};
  NNADAPTER_CHECK_EQ(AddOperation(op_type, &input_indexes, &output_indexes),
                     NEURON_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

int Program::ConvertTranspose(driver::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  // Perm
  auto perm_operand = input_operands[1];
  auto perm_count = perm_operand->length / sizeof(int32_t);
  auto perm_data = reinterpret_cast<int32_t*>(perm_operand->buffer);
  for (uint32_t i = 0; i < perm_count; i++) {
    NNADAPTER_VLOG(5) << "perm[" << i << "]=" << perm_data[i];
  }
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to Neuron operands and operations
  auto input_index = ConvertOperand(input_operand);
  auto perm_index = AddInt32ConstantOperand(perm_data, perm_count);
  auto output_index = ConvertOperand(output_operand);
  NNADAPTER_LOG(INFO) << "yrans output_idx:" << output_index;
  std::vector<uint32_t> input_indexes = {input_index, perm_index};
  std::vector<uint32_t> output_indexes = {output_index};
  NNADAPTER_CHECK_EQ(
      AddOperation(NEURON_TRANSPOSE, &input_indexes, &output_indexes),
      NEURON_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

}  // namespace mediatek_apu
}  // namespace driver
}  // namespace nnadapter
