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
#include "../../nnadapter_logging.h"  // NOLINT
#include "utility.h"                  // NOLINT

namespace nnadapter {
namespace driver {
namespace mediatek_apu {

Program::~Program() {}

int Program::Build(driver::Model* model, driver::Cache* cache) {
  NNADAPTER_VLOG(5) << "\n" << Visualize(model);
  // Convert a NNAdapter model
  operand_idx_ = 0;
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
      default:
        NNADAPTER_LOG(ERROR) << "Unsupported operation(" << operation->type
                             << ") is found.";
        break;
    }
  }
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
  return NNADAPTER_NO_ERROR;
}

int Program::ConvertOperand(driver::Operand* operand) {
  if (nodes_.find(operand) != nodes_.end()) {
    return nodes_.at(operand);
  }

  return 0;
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
  auto input_index = ConvertOperand(input_operand);
  auto input_channel_size = input_operand->type.dimensions[1];
  NNADAPTER_LOG(INFO) << "input dims[" << input_operand->type.dimensions[0]
                      << "," << input_channel_size << ","
                      << input_operand->type.dimensions[2] << ","
                      << input_operand->type.dimensions[3] << "]";
  // Filter
  auto filter_operand = input_operands[1];
  auto filter_node = ConvertOperand(filter_operand);
  auto output_channel_size = filter_operand->type.dimensions[0];
  auto filter_channel_size = filter_operand->type.dimensions[1];
  auto filter_height = filter_operand->type.dimensions[2];
  auto filter_width = filter_operand->type.dimensions[3];
  NNADAPTER_LOG(INFO) << "filter dims[" << output_channel_size << ","
                      << filter_channel_size << "," << filter_height << ","
                      << filter_width << "]";
  // Bias
  auto bias_operand = input_operands[2];
  auto bias_node = ConvertOperand(bias_operand);
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
  NNADAPTER_VLOG(5) << "dilations=[" << dilation_width << "," << dilation_height
                    << "]";
  // Output
  auto output_operand = output_operands[0];
  auto output_node = ConvertOperand(output_operand);
  // Check depthwise mode
  bool is_depthwise_mode =
      (input_channel_size == group && output_channel_size == group &&
       filter_channel_size == 1);
  NNADAPTER_VLOG(5) << "depthwise mode(" << is_depthwise_mode << ").";
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
  auto input_node = ConvertOperand(input_operand);
  // Weight
  auto weight_operand = input_operands[1];
  auto weight_node = ConvertOperand(weight_operand);
  NNADAPTER_CHECK_EQ(weight_operand->type.dimension_count, 2);
  auto num_units = weight_operand->type.dimensions[0];
  auto input_size = weight_operand->type.dimensions[1];
  NNADAPTER_LOG(INFO) << "weight dims[" << num_units << "," << input_size
                      << "]";
  // Bias
  auto bias_operand = input_operands[2];
  auto bias_node = ConvertOperand(bias_operand);
  NNADAPTER_CHECK_EQ(bias_operand->type.dimension_count, 1);
  NNADAPTER_CHECK_EQ(num_units, bias_operand->type.dimensions[0]);
  NNADAPTER_LOG(INFO) << "bias dims[" << bias_operand->type.dimensions[0]
                      << "]";
  // Fuse code
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[3]->buffer);
  NNADAPTER_VLOG(5) << "fuse_code=" << fuse_code;
  // Output
  auto output_operand = output_operands[0];
  auto output_node = ConvertOperand(output_operand);
  return NNADAPTER_NO_ERROR;
}

int Program::ConvertAverageAndMaxPool2D(driver::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 12);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  auto input_node = ConvertOperand(input_operand);
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
  // Count include pad
  bool count_include_pad =
      *reinterpret_cast<int8_t*>(input_operands[11]->buffer);
  NNADAPTER_VLOG(5) << "count_include_pad=" << count_include_pad;
  NNADAPTER_CHECK_EQ(count_include_pad, false)
      << "rknpu_ddk doesn't suppport count_include_pad=true";
  // Output
  auto output_operand = output_operands[0];
  auto output_node = ConvertOperand(output_operand);
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
  auto input0_node = ConvertOperand(input0_operand);
  // Input1
  auto input1_operand = input_operands[1];
  auto input1_node = ConvertOperand(input1_operand);
  // Output
  auto output_operand = output_operands[0];
  auto output_node = ConvertOperand(output_operand);
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
  auto input_node = ConvertOperand(input_operand);
  // Axis
  auto axis = *reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  if (axis < 0) {
    axis += input_operand->type.dimension_count;
  }
  NNADAPTER_VLOG(5) << "axis=" << axis;
  // Output
  auto output_operand = output_operands[0];
  auto output_node = ConvertOperand(output_operand);
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
  auto input_node = ConvertOperand(input_operand);
  // Output
  auto output_operand = output_operands[0];
  auto output_node = ConvertOperand(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace mediatek_apu
}  // namespace driver
}  // namespace nnadapter
