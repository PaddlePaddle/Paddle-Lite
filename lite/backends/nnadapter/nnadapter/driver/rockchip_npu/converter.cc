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

#include "driver/rockchip_npu/converter.h"
#include <vector>
#include "optimizer/symm2asymm.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"

namespace nnadapter {
namespace rockchip_npu {

Context::Context() {
  // TODO(hong19860320) create the raw context from rknpu ddk driver
}

Context::~Context() {}

Program::~Program() {
  if (!execution_) {
    delete execution_;
  }
  if (!graph_) {
    delete graph_;
  }
}

int Program::Build(hal::Model* model, hal::Cache* cache) {
  // Convert the quantization parameters of the operands in the NNAdapter model
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  ConvertModelFromSymmToAsymmQuantization(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  // Convert a NNAdapter model to a rknpu graph
  tensors_.clear();
  graph_ = new rk::nn::Graph();
  if (!graph_) {
    return NNADAPTER_OUT_OF_MEMORY;
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
  std::vector<std::shared_ptr<rk::nn::Tensor>> input_tensors(input_count);
  input_info_.resize(input_count);
  input_zero_points_.resize(input_count);
  for (size_t i = 0; i < input_count; i++) {
    auto operand = model->input_operands[i];
    NNADAPTER_CHECK(tensors_.find(operand) != tensors_.end());
    input_tensors[i] = tensors_[operand];
    // Initialize the input info for the execution
    input_info_[i].index = i;
    input_info_[i].buf = nullptr;
    input_info_[i].size = 0;
    input_info_[i].pass_through = false;
    input_info_[i].type = ConvertPrecision(operand->type.precision);
    input_info_[i].layout = ConvertDataLayout(operand->type.layout);
    input_zero_points_[i] = operand->type.asymm_per_layer_params.zero_point;
  }
  auto output_count = model->output_operands.size();
  std::vector<std::shared_ptr<rk::nn::Tensor>> output_tensors(output_count);
  output_info_.resize(output_count);
  output_zero_points_.resize(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto operand = model->output_operands[i];
    NNADAPTER_CHECK(tensors_.find(operand) != tensors_.end());
    output_tensors[i] = tensors_[operand];
    // Initialize the output info for the execution
    output_info_[i].index = i;
    output_info_[i].buf = nullptr;
    output_info_[i].size = 0;
    output_info_[i].want_float = false;
    output_info_[i].type = ConvertPrecision(operand->type.precision);
    output_info_[i].layout = ConvertDataLayout(operand->type.layout);
    output_zero_points_[i] = operand->type.asymm_per_layer_params.zero_point;
  }
  graph_->SetInputsOutputs(input_tensors, output_tensors);
  // Create an execution to build the graph to the device-related program.
  execution_ = new rk::nn::Exection(graph_);
  execution_->Build();
  return NNADAPTER_NO_ERROR;
}

int Program::Execute(uint32_t input_count,
                     hal::Argument* input_arguments,
                     uint32_t output_count,
                     hal::Argument* output_arguments) {
  NNADAPTER_CHECK_EQ(input_info_.size(), input_count);
  NNADAPTER_CHECK_EQ(output_info_.size(), output_count);
  for (uint32_t i = 0; i < input_count; i++) {
    auto& argument = input_arguments[i];
    auto buffer = reinterpret_cast<uint8_t*>(argument.buffer);
    auto zero_point = input_zero_points_[argument.index];
    for (int j = 0; j < argument.length; j++) {
      buffer[j] =
          static_cast<uint8_t>(static_cast<int16_t>(buffer[j]) + zero_point);
    }
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

std::shared_ptr<rk::nn::Tensor> Program::ConvertOperand(hal::Operand* operand) {
  if (tensors_.find(operand) != tensors_.end()) {
    return tensors_.at(operand);
  }
  auto attr = std::make_shared<rk::nn::TensorAttr>();
  attr->name = string_format("0x%X", operand);
  attr->role = operand->buffer == nullptr ? rk::nn::TensorRole::VAR
                                          : rk::nn::TensorRole::CONST;
  attr->dims = ConvertDimensions(operand->type.dimensions,
                                 operand->type.dimension_count);
  attr->precision = ConvertPrecision(operand->type.precision);
  attr->layout = ConvertDataLayout(operand->type.layout);
  switch (operand->type.precision) {
    case NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER:
      attr->qntBits = 8;
      attr->qntType = rk::nn::QuantizationType::AFFINE_ASYMMETRIC;
      attr->qntParamAffineAsymmetric.scale.resize(1);
      attr->qntParamAffineAsymmetric.scale[0] =
          operand->type.asymm_per_layer_params.scale;
      attr->qntParamAffineAsymmetric.zero_point.resize(1);
      attr->qntParamAffineAsymmetric.zero_point[0] =
          operand->type.asymm_per_layer_params.zero_point;
      break;
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER:
      attr->qntBits = 32;
      attr->qntType = rk::nn::QuantizationType::SYMMETRIC;
      attr->qntParamSymmetric.scale.resize(1);
      attr->qntParamSymmetric.scale[0] =
          operand->type.symm_per_layer_params.scale;
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Can not convert an operand@0x" << std::hex
                           << operand << " with precision="
                           << OperandPrecisionCodeToString(
                                  operand->type.precision)
                           << " to rk::nn::Tensor !";
      break;
  }
  auto tensor = graph_->CreateTensor(attr, operand->buffer);
  NNADAPTER_CHECK(tensor);
  // Use to find the tensor based on the pointer of operand
  tensors_[operand] = tensor;
  return tensor;
}

}  // namespace rockchip_npu
}  // namespace nnadapter
