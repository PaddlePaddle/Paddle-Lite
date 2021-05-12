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
namespace rockchip_npu {

Program::~Program() {
  if (!execution_) {
    delete execution_;
  }
  if (!graph_) {
    delete graph_;
  }
}

int Program::Build(driver::Model* model, driver::Cache* cache) {
  NNADAPTER_VLOG(5) << "\n" << Visualize(model);
  // Convert a NNAdapter model to a rknpu graph
  graph_ = new rk::nn::Graph();
  if (!graph_) {
    return NNADAPTER_OUT_OF_MEMORY;
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
      default:
        NNADAPTER_LOG(ERROR) << "Unsupported operation(" << operation->type
                             << ") is found.";
        break;
    }
  }
  // Indentify the inputs and outputs
  auto input_count = model->input_operands.size();
  std::vector<std::shared_ptr<rk::nn::Tensor>> input_nodes(input_count);
  input_info_.resize(input_count);
  for (size_t i = 0; i < input_count; i++) {
    auto operand = model->input_operands[i];
    NNADAPTER_CHECK(nodes_.find(operand) != nodes_.end());
    input_nodes[i] = nodes_[operand];
    // Initialize the input info for the execution
    input_info_[i].index = i;
    input_info_[i].buf = nullptr;
    input_info_[i].size = 0;
    input_info_[i].pass_through = false;
    input_info_[i].type = ConvertPrecision(operand->type.precision);
    input_info_[i].layout = ConvertDataLayout(operand->type.layout);
  }
  auto output_count = model->output_operands.size();
  std::vector<std::shared_ptr<rk::nn::Tensor>> output_nodes(output_count);
  output_info_.resize(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto operand = model->output_operands[i];
    NNADAPTER_CHECK(nodes_.find(operand) != nodes_.end());
    output_nodes[i] = nodes_[operand];
    // Initialize the output info for the execution
    output_info_[i].index = i;
    output_info_[i].buf = nullptr;
    output_info_[i].size = 0;
    output_info_[i].want_float = false;
    output_info_[i].type = ConvertPrecision(operand->type.precision);
    output_info_[i].layout = ConvertDataLayout(operand->type.layout);
  }
  graph_->SetInputsOutputs(input_nodes, output_nodes);
  // Create an execution to build the graph to the device-related program.
  execution_ = new rk::nn::Exection(graph_);
  execution_->Build();
  return NNADAPTER_NO_ERROR;
}

int Program::Execute(uint32_t input_count,
                     driver::Argument* input_arguments,
                     uint32_t output_count,
                     driver::Argument* output_arguments) {
  NNADAPTER_CHECK_EQ(input_info_.size(), input_count);
  NNADAPTER_CHECK_EQ(output_info_.size(), output_count);
  for (uint32_t i = 0; i < input_count; i++) {
    auto& argument = input_arguments[i];
    input_info_[argument.index].buf = argument.buffer;
    input_info_[argument.index].size = argument.length;
  }
  for (uint32_t i = 0; i < output_count; i++) {
    auto& argument = output_arguments[i];
    output_info_[argument.index].buf = argument.buffer;
    output_info_[argument.index].size = argument.length;
  }
  NNADAPTER_CHECK_EQ(execution_->SetInputs(input_info_), rk::nn::RK_SUCCESS);
  NNADAPTER_CHECK_EQ(execution_->Run(), rk::nn::RK_SUCCESS);
  NNADAPTER_CHECK_EQ(execution_->GetOutputs(output_info_), rk::nn::RK_SUCCESS);
  return NNADAPTER_NO_ERROR;
}

std::shared_ptr<rk::nn::Tensor> Program::ConvertOperand(
    driver::Operand* operand) {
  if (nodes_.find(operand) != nodes_.end()) {
    return nodes_.at(operand);
  }

#define CONVERT_QUANT_INTx_SYMM_PER_LAYER(bits)           \
  case NNADAPTER_TENSOR_QUANT_INT##bits##_SYMM_PER_LAYER: \
    attr->qntBits = bits;                                 \
    attr->qntType = rk::nn::QuantizationType::SYMMETRIC;  \
    attr->qntParamSymmetric.scale.resize(1);              \
    attr->qntParamSymmetric.scale[0] =                    \
        operand->type.symm_per_layer_params.scale;        \
    break;
#define CONVERT_QUANT_INTx_SYMM_PER_CHANNEL(bits)                              \
  case NNADAPTER_TENSOR_QUANT_INT##bits##_SYMM_PER_CHANNEL:                    \
    attr->qntBits = bits;                                                      \
    attr->qntType = rk::nn::QuantizationType::SYMMETRIC;                       \
    attr->qntParamSymmetric.scale.resize(                                      \
        operand->type.symm_per_channel_params.scale_count);                    \
    memcpy(&attr->qntParamSymmetric.scale[0],                                  \
           operand->type.symm_per_channel_params.scales,                       \
           operand->type.symm_per_channel_params.scale_count * sizeof(float)); \
    break;

  auto attr = std::make_shared<rk::nn::TensorAttr>();
  attr->name = driver::string_format("0x%X", operand);
  attr->role = operand->buffer == nullptr ? rk::nn::TensorRole::VAR
                                          : rk::nn::TensorRole::CONST;
  attr->dims = ConvertDimensions(operand->type.dimensions,
                                 operand->type.dimension_count);
  attr->precision = ConvertPrecision(operand->type.precision);
  attr->layout = ConvertDataLayout(operand->type.layout);
  switch (operand->type.precision) {
    CONVERT_QUANT_INTx_SYMM_PER_LAYER(8) CONVERT_QUANT_INTx_SYMM_PER_LAYER(32)
            CONVERT_QUANT_INTx_SYMM_PER_CHANNEL(8)
                CONVERT_QUANT_INTx_SYMM_PER_CHANNEL(32) default
        : NNADAPTER_LOG(WARNING)
          << "Can not convert an operand@0x"
          << std::hex
          << operand
          << " with precision="
          << operand->type.precision
          << " to rk::nn::Tensor !";
    break;
  }
  auto tensor = graph_->CreateTensor(attr, operand->buffer);
  NNADAPTER_CHECK(tensor);
  // Use to find the tensor based on the pointer of operand
  nodes_[operand] = tensor;

#undef CONVERT_QUANT_INTx_SYMM_PER_LAYER
#undef CONVERT_QUANT_INTx_SYMM_PER_CHANNEL
  return tensor;
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
  auto input_node = ConvertOperand(input_operand);
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

  rk::nn::Conv2DAttr attr;
  attr.ksize[0] = filter_height;
  attr.ksize[1] = filter_width;
  attr.stride[0] = stride_width;
  attr.stride[1] = stride_height;
  attr.pad[0] = padding_width_left;
  attr.pad[1] = padding_width_right;
  attr.pad[2] = padding_height_top;
  attr.pad[3] = padding_height_bottom;
  attr.group = group;
  attr.multiplier = is_depthwise_mode ? 1 : 0;
  attr.weights = output_channel_size;
  attr.dilation[0] = dilation_width;
  attr.dilation[1] = dilation_height;
  attr.pad_type = rk::nn::PadType::AUTO;
  // fuse RELU ?
  if (fuse_code == NNADAPTER_FUSED_NONE) {
    attr.has_relu = false;
  } else if (fuse_code == NNADAPTER_FUSED_RELU) {
    attr.has_relu = true;
  } else {
    NNADAPTER_LOG(ERROR) << "Unsupported fuse_code(" << fuse_code
                         << ") is found.";
  }
  std::vector<std::shared_ptr<rk::nn::Tensor>> input_nodes = {
      input_node, filter_node, bias_node};
  std::vector<std::shared_ptr<rk::nn::Tensor>> output_nodes = {output_node};
  graph_->AddOperator(
      rk::nn::OperatorType::CONV2D, input_nodes, output_nodes, &attr);
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

  rk::nn::FCAttr attr;
  attr.weights = num_units;
  // fuse RELU ?
  if (fuse_code == NNADAPTER_FUSED_NONE) {
    attr.has_relu = false;
  } else if (fuse_code == NNADAPTER_FUSED_RELU) {
    attr.has_relu = true;
  } else {
    NNADAPTER_LOG(ERROR) << "Unsupported fuse_code(" << fuse_code
                         << ") is found.";
  }
  std::vector<std::shared_ptr<rk::nn::Tensor>> input_nodes = {
      input_node, weight_node, bias_node};
  std::vector<std::shared_ptr<rk::nn::Tensor>> output_nodes = {output_node};
  graph_->AddOperator(
      rk::nn::OperatorType::FULLCONNECT, input_nodes, output_nodes, &attr);
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

  rk::nn::PoolAttr attr;
  attr.ksize[0] = filter_height;
  attr.ksize[1] = filter_width;
  attr.stride[0] = stride_width;
  attr.stride[1] = stride_height;
  attr.pad[0] = padding_width_left;
  attr.pad[1] = padding_width_right;
  attr.pad[2] = padding_height_top;
  attr.pad[3] = padding_height_bottom;
  attr.pad_type = rk::nn::PadType::AUTO;
  attr.global_pooling = global_pooling;
  attr.round_type = ceil_mode ? rk::nn::RoundType::ROUND_CEIL
                              : rk::nn::RoundType::ROUND_FLOOR;
  std::vector<std::shared_ptr<rk::nn::Tensor>> input_nodes = {input_node};
  std::vector<std::shared_ptr<rk::nn::Tensor>> output_nodes = {output_node};
  if (operation->type == NNADAPTER_AVERAGE_POOL_2D) {
    attr.pool_type = rk::nn::PoolType::POOLING_AVG;
  } else if (operation->type == NNADAPTER_MAX_POOL_2D) {
    attr.pool_type = rk::nn::PoolType::POOLING_MAX;
  } else {
    NNADAPTER_LOG(ERROR) << "Unsupported pooling operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  graph_->AddOperator(
      rk::nn::OperatorType::POOL, input_nodes, output_nodes, &attr);
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

  std::vector<std::shared_ptr<rk::nn::Tensor>> input_nodes = {input0_node,
                                                              input1_node};
  std::vector<std::shared_ptr<rk::nn::Tensor>> output_nodes = {output_node};
  rk::nn::OperatorType op_type;
  if (operation->type == NNADAPTER_ADD) {
    op_type = rk::nn::OperatorType::ADD;
  } else if (operation->type == NNADAPTER_SUB) {
    op_type = rk::nn::OperatorType::SUBTRACT;
  } else if (operation->type == NNADAPTER_MUL) {
    op_type = rk::nn::OperatorType::MULTIPLY;
  } else if (operation->type == NNADAPTER_DIV) {
    op_type = rk::nn::OperatorType::DIVIDE;
  } else {
    NNADAPTER_LOG(ERROR) << "Unsupported element-wise binary operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  graph_->AddOperator(op_type, input_nodes, output_nodes, nullptr);
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

  rk::nn::SoftmaxAttr attr;
  attr.axis = axis;
  attr.beta = 1.0;
  std::vector<std::shared_ptr<rk::nn::Tensor>> input_nodes = {input_node};
  std::vector<std::shared_ptr<rk::nn::Tensor>> output_nodes = {output_node};
  graph_->AddOperator(
      rk::nn::OperatorType::SOFTMAX, input_nodes, output_nodes, &attr);
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

  std::vector<std::shared_ptr<rk::nn::Tensor>> input_nodes = {input_node};
  std::vector<std::shared_ptr<rk::nn::Tensor>> output_nodes = {output_node};
  rk::nn::OperatorType op_type;
  if (operation->type == NNADAPTER_SIGMOID) {
    op_type = rk::nn::OperatorType::SIGMOID;
  } else if (operation->type == NNADAPTER_RELU) {
    op_type = rk::nn::OperatorType::RELU;
  } else if (operation->type == NNADAPTER_RELU6) {
    op_type = rk::nn::OperatorType::RELU6;
  } else if (operation->type == NNADAPTER_TANH) {
    op_type = rk::nn::OperatorType::TANH;
  } else {
    NNADAPTER_LOG(ERROR) << "Unsupported activation unary operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  graph_->AddOperator(op_type, input_nodes, output_nodes, nullptr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace rockchip_npu
}  // namespace driver
}  // namespace nnadapter
