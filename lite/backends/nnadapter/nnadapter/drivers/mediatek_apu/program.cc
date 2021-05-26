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

#include "program.h"  // NOLINT
#include <memory>
#include <vector>
#include "../../nnadapter_common.h"     // NOLINT
#include "../../nnadapter_optimizer.h"  // NOLINT
#include "utility.h"                    // NOLINT

namespace nnadapter {
namespace driver {
namespace mediatek_apu {

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
  ConvertQuantizationFromSymmToAsymm(model);
  ConvertDataLayoutFromNCHWToNHWC(model);
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

uint32_t Program::AddScalarInt32ConstantOperand(int32_t value) {
  return AddScalarConstantOperand<int32_t>(value, NEURON_INT32);
}

uint32_t Program::AddScalarFloat32ConstantOperand(float value) {
  return AddScalarConstantOperand<float>(value, NEURON_FLOAT32);
}

uint32_t Program::AddVectorInt32ConstantOperand(const int32_t* values,
                                                uint32_t num_values) {
  return AddVectorConstantOperand<int32_t>(
      values, num_values, NEURON_TENSOR_INT32);
}

uint32_t Program::AddVectorInt32ConstantOperand(const int32_t* values,
                                                uint32_t num_values,
                                                float scale,
                                                int32_t zero_point) {
  return AddVectorConstantOperand<int32_t>(
      values, num_values, NEURON_TENSOR_INT32, scale, zero_point);
}

uint32_t Program::AddVectorFloat32ConstantOperand(const float* values,
                                                  uint32_t num_values) {
  return AddVectorConstantOperand<float>(
      values, num_values, NEURON_TENSOR_FLOAT32);
}

uint32_t Program::ConvertOperand(driver::Operand* operand) {
  if (operand_indexes_.find(operand) != operand_indexes_.end()) {
    return operand_indexes_.at(operand);
  }
  auto is_constant_copy = operand->type.lifetime == NNADAPTER_CONSTANT_COPY;
  auto is_constant_reference =
      operand->type.lifetime == NNADAPTER_CONSTANT_REFERENCE;
  auto is_constant = is_constant_copy || is_constant_reference;
  uint32_t operand_index = 0;
  NeuronOperandType operand_type;
  memset(&operand_type, 0, sizeof(NeuronOperandType));
  auto operand_dimensions = ConvertDimensions(operand->type.dimensions,
                                              operand->type.dimension_count);
  operand_type.dimensions = &operand_dimensions[0];
  operand_type.dimensionCount = operand_dimensions.size();
  NeuronSymmPerChannelQuantParams operand_symm_per_channel_quant_params;
  memset(&operand_symm_per_channel_quant_params,
         0,
         sizeof(NeuronSymmPerChannelQuantParams));
  switch (operand->type.precision) {
    case NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER: {
      operand_type.type = NEURON_TENSOR_QUANT8_ASYMM;
      operand_type.scale = operand->type.asymm_per_layer_params.scale;
      operand_type.zeroPoint = operand->type.asymm_per_layer_params.zero_point;
      NeuronModel_addOperand_invoke(model_, &operand_type);
      operand_index = operand_index_++;
      if (is_constant) {
        NNADAPTER_CHECK_EQ(
            NeuronModel_setOperandValue_invoke(
                model_, operand_index, operand->buffer, operand->length),
            NNADAPTER_NO_ERROR);
      } else {
        operand_indexes_[operand] = operand_index;
      }
    } break;
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL: {
      NNADAPTER_CHECK(is_constant);
      operand_type.type = NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL;
      NeuronModel_addOperand_invoke(model_, &operand_type);
      operand_index = operand_index_++;
      NNADAPTER_CHECK_EQ(
          NeuronModel_setOperandValue_invoke(
              model_, operand_index, operand->buffer, operand->length),
          NNADAPTER_NO_ERROR);
      operand_symm_per_channel_quant_params.scales =
          operand->type.symm_per_channel_params.scales;
      operand_symm_per_channel_quant_params.scaleCount =
          operand->type.symm_per_channel_params.scale_count;
      operand_symm_per_channel_quant_params.channelDim =
          operand->type.symm_per_channel_params.channel_dim;
      NNADAPTER_CHECK_EQ(
          NeuronModel_setOperandSymmPerChannelQuantParams_invoke(
              model_, operand_index, &operand_symm_per_channel_quant_params),
          NNADAPTER_NO_ERROR);
    } break;
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER:
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL: {
      // Only for bias
      NNADAPTER_CHECK(is_constant);
      operand_type.type = NEURON_TENSOR_INT32;
      if (operand->type.precision ==
          NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER) {
        operand_type.scale = operand->type.symm_per_layer_params.scale;
      }
      NeuronModel_addOperand_invoke(model_, &operand_type);
      operand_index = operand_index_++;
      NNADAPTER_CHECK_EQ(
          NeuronModel_setOperandValue_invoke(
              model_, operand_index, operand->buffer, operand->length),
          NNADAPTER_NO_ERROR);
    } break;
    default:
      NNADAPTER_LOG(WARNING) << "Skip processing "
                             << OperationTypeToString(operand->type.precision)
                             << " for the conversion of Neuron operands.";
      break;
  }
  return operand_index;
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
  NNADAPTER_VLOG(5) << "input: " << OperandTypeToString(&input_operand->type);
  auto input_channel_size = input_operand->type.dimensions[3];
  // Filter
  auto filter_operand = input_operands[1];
  NNADAPTER_VLOG(5) << "filter: " << OperandTypeToString(&filter_operand->type);
  auto filter_height = filter_operand->type.dimensions[1];
  auto filter_width = filter_operand->type.dimensions[2];
  // Bias
  auto bias_operand = input_operands[2];
  NNADAPTER_VLOG(5) << "bias: " << OperandTypeToString(&bias_operand->type);
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
  NNADAPTER_VLOG(5) << "output: " << OperandTypeToString(&output_operand->type);
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
  auto bias_index = ConvertOperand(bias_operand);
  auto padding_width_left_index =
      AddScalarInt32ConstantOperand(padding_width_left);
  auto padding_width_right_index =
      AddScalarInt32ConstantOperand(padding_width_right);
  auto padding_height_top_index =
      AddScalarInt32ConstantOperand(padding_height_top);
  auto padding_height_bottom_index =
      AddScalarInt32ConstantOperand(padding_height_bottom);
  auto stride_width_index = AddScalarInt32ConstantOperand(stride_width);
  auto stride_height_index = AddScalarInt32ConstantOperand(stride_height);
  auto fuse_code_index =
      AddScalarInt32ConstantOperand(ConvertFuseCode(fuse_code));
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
    auto multiplier_index = AddScalarInt32ConstantOperand(multiplier);
    input_indexes.push_back(multiplier_index);
    input_indexes.push_back(fuse_code_index);
    NNADAPTER_CHECK_EQ(NeuronModel_addOperation_invoke(model_,
                                                       NEURON_DEPTHWISE_CONV_2D,
                                                       input_indexes.size(),
                                                       &input_indexes[0],
                                                       output_indexes.size(),
                                                       &output_indexes[0]),
                       NEURON_NO_ERROR);
  } else {
    input_indexes.push_back(fuse_code_index);
    NNADAPTER_CHECK_EQ(NeuronModel_addOperation_invoke(model_,
                                                       NEURON_CONV_2D,
                                                       input_indexes.size(),
                                                       &input_indexes[0],
                                                       output_indexes.size(),
                                                       &output_indexes[0]),
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
  NNADAPTER_VLOG(5) << "input: " << OperandTypeToString(&input_operand->type);
  // Weight
  auto weight_operand = input_operands[1];
  NNADAPTER_VLOG(5) << "weight: " << OperandTypeToString(&weight_operand->type);
  NNADAPTER_CHECK_EQ(weight_operand->type.dimension_count, 2);
  auto num_units = weight_operand->type.dimensions[0];
  auto input_size = weight_operand->type.dimensions[1];
  NNADAPTER_LOG(INFO) << "weight dims[" << num_units << "," << input_size
                      << "]";
  // Bias
  auto bias_operand = input_operands[2];
  NNADAPTER_VLOG(5) << "bias: " << OperandTypeToString(&bias_operand->type);
  NNADAPTER_CHECK_EQ(bias_operand->type.dimension_count, 1);
  NNADAPTER_CHECK_EQ(num_units, bias_operand->type.dimensions[0]);
  NNADAPTER_LOG(INFO) << "bias dims[" << bias_operand->type.dimensions[0]
                      << "]";
  // Fuse code
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[3]->buffer);
  NNADAPTER_VLOG(5) << "fuse_code=" << fuse_code;
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandTypeToString(&output_operand->type);

  // Convert to Neuron operands and operations
  auto input_index = ConvertOperand(input_operand);
  auto weight_index = ConvertOperand(weight_operand);
  auto bias_index = ConvertOperand(bias_operand);
  auto fuse_code_index =
      AddScalarInt32ConstantOperand(ConvertFuseCode(fuse_code));
  auto output_index = ConvertOperand(output_operand);
  std::vector<uint32_t> input_indexes = {
      input_index, weight_index, bias_index, fuse_code_index};
  std::vector<uint32_t> output_indexes = {output_index};
  NNADAPTER_CHECK_EQ(NeuronModel_addOperation_invoke(model_,
                                                     NEURON_FULLY_CONNECTED,
                                                     input_indexes.size(),
                                                     &input_indexes[0],
                                                     output_indexes.size(),
                                                     &output_indexes[0]),
                     NNADAPTER_NO_ERROR);
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
  NNADAPTER_VLOG(5) << "input: " << OperandTypeToString(&input_operand->type);
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
  NNADAPTER_VLOG(5) << "output: " << OperandTypeToString(&output_operand->type);

  // Convert to Neuron operands and operations
  auto input_index = ConvertOperand(input_operand);
  auto padding_width_left_index =
      AddScalarInt32ConstantOperand(padding_width_left);
  auto padding_width_right_index =
      AddScalarInt32ConstantOperand(padding_width_right);
  auto padding_height_top_index =
      AddScalarInt32ConstantOperand(padding_height_top);
  auto padding_height_bottom_index =
      AddScalarInt32ConstantOperand(padding_height_bottom);
  auto stride_width_index = AddScalarInt32ConstantOperand(stride_width);
  auto stride_height_index = AddScalarInt32ConstantOperand(stride_height);
  auto filter_width_index = AddScalarInt32ConstantOperand(filter_width);
  auto filter_height_index = AddScalarInt32ConstantOperand(filter_height);
  auto fuse_code_index =
      AddScalarInt32ConstantOperand(ConvertFuseCode(fuse_code));
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
  NNADAPTER_CHECK_EQ(NeuronModel_addOperation_invoke(model_,
                                                     op_type,
                                                     input_indexes.size(),
                                                     &input_indexes[0],
                                                     output_indexes.size(),
                                                     &output_indexes[0]),
                     NNADAPTER_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

int Program::ConvertElementwiseBinaryOperations(driver::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input0
  auto input0_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input0: " << OperandTypeToString(&input0_operand->type);
  // Input1
  auto input1_operand = input_operands[1];
  NNADAPTER_VLOG(5) << "input1: " << OperandTypeToString(&input1_operand->type);
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandTypeToString(&output_operand->type);

  // Convert to Neuron operands and operations
  auto input0_index = ConvertOperand(input0_operand);
  auto input1_index = ConvertOperand(input1_operand);
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
  std::vector<uint32_t> input_indexes = {input0_index, input1_index};
  std::vector<uint32_t> output_indexes = {output_index};
  NNADAPTER_CHECK_EQ(NeuronModel_addOperation_invoke(model_,
                                                     op_type,
                                                     input_indexes.size(),
                                                     &input_indexes[0],
                                                     output_indexes.size(),
                                                     &output_indexes[0]),
                     NNADAPTER_NO_ERROR);
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
  NNADAPTER_VLOG(5) << "input: " << OperandTypeToString(&input_operand->type);
  // Axis
  auto axis = *reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  if (axis < 0) {
    axis += input_operand->type.dimension_count;
  }
  NNADAPTER_VLOG(5) << "axis=" << axis;
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandTypeToString(&output_operand->type);

  // Convert to Neuron operands and operations
  auto input_index = ConvertOperand(input_operand);
  auto beta_index = AddScalarFloat32ConstantOperand(1.0f);
  auto axis_index = AddScalarInt32ConstantOperand(axis);
  auto output_index = ConvertOperand(output_operand);
  std::vector<uint32_t> input_indexes = {input_index, beta_index, axis_index};
  std::vector<uint32_t> output_indexes = {output_index};
  NNADAPTER_CHECK_EQ(NeuronModel_addOperation_invoke(model_,
                                                     NEURON_SOFTMAX,
                                                     input_indexes.size(),
                                                     &input_indexes[0],
                                                     output_indexes.size(),
                                                     &output_indexes[0]),
                     NNADAPTER_NO_ERROR);
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
  NNADAPTER_VLOG(5) << "input: " << OperandTypeToString(&input_operand->type);
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandTypeToString(&output_operand->type);

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
  NNADAPTER_CHECK_EQ(NeuronModel_addOperation_invoke(model_,
                                                     op_type,
                                                     input_indexes.size(),
                                                     &input_indexes[0],
                                                     output_indexes.size(),
                                                     &output_indexes[0]),
                     NNADAPTER_NO_ERROR);
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
  NNADAPTER_VLOG(5) << "input: " << OperandTypeToString(&input_operand->type);
  // Perm
  auto perm_count = input_operands[1]->length / sizeof(int32_t);
  auto perm_data = reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  for (uint32_t i = 0; i < perm_count; i++) {
    NNADAPTER_VLOG(5) << "perm[" << i << "]=" << perm_data[i];
  }
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandTypeToString(&output_operand->type);

  // Convert to Neuron operands and operations
  auto input_index = ConvertOperand(input_operand);
  auto perm_index = AddVectorInt32ConstantOperand(perm_data, perm_count);
  auto output_index = ConvertOperand(output_operand);
  std::vector<uint32_t> input_indexes = {input_index, perm_index};
  std::vector<uint32_t> output_indexes = {output_index};
  NNADAPTER_CHECK_EQ(NeuronModel_addOperation_invoke(model_,
                                                     NEURON_TRANSPOSE,
                                                     input_indexes.size(),
                                                     &input_indexes[0],
                                                     output_indexes.size(),
                                                     &output_indexes[0]),
                     NNADAPTER_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

}  // namespace mediatek_apu
}  // namespace driver
}  // namespace nnadapter
