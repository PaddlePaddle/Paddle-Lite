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

namespace nnadapter {
namespace driver {
namespace rockchip_npu {

rk::nn::PrecisionType ConvertPrecision(
    NNAdapterOperandPrecisionCode input_precision) {
  rk::nn::PrecisionType output_precision = rk::nn::PrecisionType::UNKNOWN;
  switch (input_precision) {
    case NNADAPTER_TENSOR_BOOL8:
      output_precision = rk::nn::PrecisionType::BOOL8;
      break;
    case NNADAPTER_TENSOR_INT8:
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER:
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL:
      output_precision = rk::nn::PrecisionType::INT8;
      break;
    case NNADAPTER_TENSOR_INT16:
      output_precision = rk::nn::PrecisionType::INT16;
      break;
    case NNADAPTER_TENSOR_INT32:
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER:
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL:
      output_precision = rk::nn::PrecisionType::INT32;
      break;
    case NNADAPTER_TENSOR_INT64:
      output_precision = rk::nn::PrecisionType::INT64;
      break;
    case NNADAPTER_TENSOR_UINT8:
    case NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER:
      output_precision = rk::nn::PrecisionType::UINT8;
      break;
    case NNADAPTER_TENSOR_UINT16:
      output_precision = rk::nn::PrecisionType::UINT16;
      break;
    case NNADAPTER_TENSOR_QUANT_UINT32_ASYMM_PER_LAYER:
    case NNADAPTER_TENSOR_UINT32:
      output_precision = rk::nn::PrecisionType::UINT32;
      break;
    case NNADAPTER_TENSOR_UINT64:
      output_precision = rk::nn::PrecisionType::UINT64;
      break;
    case NNADAPTER_TENSOR_FLOAT16:
      output_precision = rk::nn::PrecisionType::FLOAT16;
      break;
    case NNADAPTER_TENSOR_FLOAT32:
      output_precision = rk::nn::PrecisionType::FLOAT32;
      break;
    case NNADAPTER_TENSOR_FLOAT64:
      output_precision = rk::nn::PrecisionType::FLOAT64;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << input_precision << ") to rk::nn::PrecisionType !";
      break;
  }
  return output_precision;
}

rk::nn::DataLayoutType ConvertDataLayout(
    NNAdapterOperandLayoutCode input_layout) {
  rk::nn::DataLayoutType output_layout = rk::nn::DataLayoutType::UNKNOWN;
  switch (input_layout) {
    case NNADAPTER_NCHW:
      output_layout = rk::nn::DataLayoutType::NCHW;
      break;
    case NNADAPTER_NHWC:
      output_layout = rk::nn::DataLayoutType::NHWC;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand layout code("
          << input_layout << ") to rk::nn::DataLayoutType !";
      break;
  }
  return output_layout;
}

std::vector<int32_t> ConvertDimensions(int32_t* input_dimensions,
                                       uint32_t input_dimensions_count) {
  std::vector<int32_t> output_dimensions(input_dimensions_count);
  memcpy(&output_dimensions[0],
         input_dimensions,
         input_dimensions_count * sizeof(int32_t));
  return output_dimensions;
}

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
    switch (operation->type) {
      case NNADAPTER_CONV_2D:
        ConvertConv2D(operation);
        break;
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
  NNADAPTER_CHECK_GE(input_count, 10);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  auto input_node = ConvertOperand(input_operand);
  NNADAPTER_LOG(INFO) << "input dims[" << input_operand->type.dimensions[0]
                      << "," << input_operand->type.dimensions[1] << ","
                      << input_operand->type.dimensions[2] << ","
                      << input_operand->type.dimensions[3] << "]";
  // Filter
  auto filter_operand = input_operands[1];
  auto filter_node = ConvertOperand(filter_operand);
  // Bias
  auto bias_operand = input_operands[2];
  auto bias_node = ConvertOperand(bias_operand);
  // Output
  auto output_operand = output_operands[0];
  auto output_node = ConvertOperand(output_operand);
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
  // Fuse code
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[9]->buffer);
  NNADAPTER_VLOG(5) << "fuse_code=" << fuse_code;
  // Dilations
  int32_t dilation_width = 1;
  int32_t dilation_height = 1;
  if (input_count >= 12) {
    dilation_width = *reinterpret_cast<int32_t*>(input_operands[10]->buffer);
    dilation_height = *reinterpret_cast<int32_t*>(input_operands[11]->buffer);
  }
  NNADAPTER_VLOG(5) << "dilations=[" << dilation_width << "," << dilation_height
                    << "]";
  rk::nn::Conv2DAttr attr;
  attr.ksize[0] = filter_operand->type.dimensions[2];
  attr.ksize[1] = filter_operand->type.dimensions[3];
  attr.stride[0] = stride_width;
  attr.stride[1] = stride_height;
  attr.pad[0] = padding_width_left;
  attr.pad[1] = padding_width_right;
  attr.pad[2] = padding_height_top;
  attr.pad[3] = padding_height_bottom;
  attr.group = 1;
  attr.multiplier = 0;  // 1 represents depthwise_conv2d
  attr.weights = filter_operand->type.dimensions[0];
  attr.dilation[0] = dilation_width;
  attr.dilation[1] = dilation_height;
  attr.pad_type = rk::nn::PadType::AUTO;
  // fuse RELU ?
  if (fuse_code == NNADAPTER_FUSED_NONE) {
    attr.has_relu = false;
  } else if (fuse_code == NNADAPTER_FUSED_RELU) {
    attr.has_relu = true;
  } else {
    NNADAPTER_LOG(ERROR) << "Unsupported fuse_code(" << operation->type
                         << ") is found.";
  }
  std::vector<std::shared_ptr<rk::nn::Tensor>> input_nodes = {
      input_node, filter_node, bias_node};
  std::vector<std::shared_ptr<rk::nn::Tensor>> output_nodes = {output_node};
  graph_->AddOperator(
      rk::nn::OperatorType::CONV2D, input_nodes, output_nodes, &attr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace rockchip_npu
}  // namespace driver
}  // namespace nnadapter
