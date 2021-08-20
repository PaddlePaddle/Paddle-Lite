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

#include "driver/amlogic_npu/converter.h"
#include <algorithm>
#include <vector>
#include "driver/amlogic_npu/optimizer/unpack_op_fusion.h"
#include "optimizer/symm2asymm.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace amlogic_npu {

Context::Context(void* device, const char* properties) : device_(device) {
  // TODO(hong19860320) create the raw context from rknpu_ddk
}

Context::~Context() {}

Program::~Program() { Clear(); }

void Program::Clear() {
  tensors_.clear();
  input_types_.clear();
  output_types_.clear();
}

int Program::Build(hal::Model* model, hal::Cache* cache) {
  Clear();
  // Convert the quantization parameters of the operands in the NNAdapter model
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  UnpackOpFusion(model);
  ConvertQuantizationSymmToAsymm(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  // Convert a NNAdapter model to a amlnpu graph
  graph_ = std::make_shared<aml::nn::Graph>();
  if (!graph_) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
      case NNADAPTER_ADD:
      case NNADAPTER_SUB:
      case NNADAPTER_MUL:
      case NNADAPTER_DIV:
        ConvertElementwise(operation);
        break;
      case NNADAPTER_AVERAGE_POOL_2D:
      case NNADAPTER_MAX_POOL_2D:
        ConvertPool2D(operation);
        break;
      case NNADAPTER_CONV_2D:
        ConvertConv2D(operation);
        break;
      case NNADAPTER_CONV_2D_TRANSPOSE:
        ConvertConv2DTranspose(operation);
        break;
      case NNADAPTER_CONCAT:
        ConvertConcat(operation);
        break;
      case NNADAPTER_FULLY_CONNECTED:
        ConvertFullyConnected(operation);
        break;
      case NNADAPTER_HARD_SIGMOID:
      case NNADAPTER_RELU:
      case NNADAPTER_RELU6:
      case NNADAPTER_SIGMOID:
      case NNADAPTER_TANH:
        ConvertActivation(operation);
        break;
      case NNADAPTER_SOFTMAX:
        ConvertSoftmax(operation);
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
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  std::vector<std::shared_ptr<aml::nn::Tensor>> input_tensors;
  if (input_count > 0) {
    input_tensors.resize(input_count);
    input_types_.resize(input_count);
    for (size_t i = 0; i < input_count; i++) {
      auto operand = model->input_operands[i];
      const auto& type = operand->type;
      NNADAPTER_CHECK(tensors_.find(operand) != tensors_.end());
      input_tensors[i] = tensors_[operand].back();
      NNADAPTER_CHECK(input_tensors[i]);
      input_types_[i] = type;
    }
  }
  auto output_count = model->output_operands.size();
  NNADAPTER_VLOG(3) << "Model output count: " << output_count;
  NNADAPTER_CHECK_GT(output_count, 0);
  std::vector<std::shared_ptr<aml::nn::Tensor>> output_tensors(output_count);
  output_types_.resize(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto operand = model->output_operands[i];
    const auto& type = operand->type;
    NNADAPTER_CHECK(tensors_.find(operand) != tensors_.end());
    output_tensors[i] = tensors_[operand].back();
    NNADAPTER_CHECK(output_tensors[i]);
    output_types_[i] = type;
  }
  graph_->SetInputsOutputs(input_tensors, output_tensors);
  // Create an execution to build the graph to the device-specific program.
  execution_ = std::make_shared<aml::nn::Exection>(graph_.get());
  execution_->Build();
  return NNADAPTER_NO_ERROR;
}

int Program::Execute(uint32_t input_count,
                     hal::Argument* input_arguments,
                     uint32_t output_count,
                     hal::Argument* output_arguments) {
  NNADAPTER_CHECK_EQ(input_types_.size(), input_count);
  NNADAPTER_CHECK_EQ(output_types_.size(), output_count);
  std::vector<aml::nn::InputInfo> input_info(input_count);
  std::vector<aml::nn::OutputInfo> output_info(output_count);
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
    // Initialize the input info for the execution
    input_info[arg.index].index = arg.index;
    input_info[arg.index].buf = buffer;
    input_info[arg.index].size = length;
    input_info[arg.index].pass_through = false;
    input_info[arg.index].type =
        static_cast<int>(ConvertPrecision(type->precision));
    input_info[arg.index].layout =
        static_cast<int>(ConvertDataLayout(type->layout));
  }
  for (uint32_t i = 0; i < output_count; i++) {
    auto& arg = output_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, output_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = &output_types_[arg.index];
    auto buffer = arg.access(arg.memory, type);
    NNADAPTER_CHECK(buffer);
    auto length = GetOperandTypeBufferLength(*type);
    // Initialize the output info for the execution
    output_info[arg.index].index = arg.index;
    output_info[arg.index].buf = buffer;
    output_info[arg.index].size = length;
    output_info[arg.index].want_float = false;
    output_info[arg.index].type =
        static_cast<int>(ConvertPrecision(type->precision));
    output_info[arg.index].layout =
        static_cast<int>(ConvertDataLayout(type->layout));
  }
  auto start_time = GetCurrentUS();
  NNADAPTER_CHECK_EQ(execution_->SetInputs(input_info), aml::nn::AML_SUCCESS);
  NNADAPTER_CHECK_EQ(execution_->Run(), aml::nn::AML_SUCCESS);
  NNADAPTER_CHECK_EQ(execution_->GetOutputs(output_info), aml::nn::AML_SUCCESS);
  NNADAPTER_VLOG(3) << "Process cost " << GetCurrentUS() - start_time << " us";
  for (uint32_t i = 0; i < output_count; i++) {
    auto type = &output_types_[i];
    auto buffer = output_info[i].buf;
    auto length = output_info[i].size;
    if (IsUInt8AsymmPerLayerQuantType(type->precision)) {
      Asymm2SymmData(reinterpret_cast<const uint8_t*>(buffer),
                     length,
                     type->asymm_per_layer_params.zero_point,
                     reinterpret_cast<int8_t*>(buffer));
    }
  }
  return NNADAPTER_NO_ERROR;
}

std::string Program::GetTensorName(hal::Operand* operand) {
  auto operand_id = OperandIdToString(operand);
  auto index = 0;
  auto it = tensors_.find(operand);
  if (it != tensors_.end()) {
    index = it->second.size();
  }
  return operand_id + string_format("_%d", index);
}

std::shared_ptr<aml::nn::Tensor> Program::GetMappedTensor(
    hal::Operand* operand) {
  auto it = tensors_.find(operand);
  if (it != tensors_.end()) {
    return it->second.back();
  }
  return nullptr;
}

std::shared_ptr<aml::nn::Tensor> Program::UpdateTensorMap(
    hal::Operand* operand, std::shared_ptr<aml::nn::Tensor> tensor) {
  auto it = tensors_.find(operand);
  if (it == tensors_.end()) {
    auto result = tensors_.insert(std::make_pair(
        operand, std::vector<std::shared_ptr<aml::nn::Tensor>>()));
    NNADAPTER_CHECK(result.second);
    it = result.first;
  }
  it->second.push_back(tensor);
  return tensor;
}

std::shared_ptr<aml::nn::Tensor> Program::ConvertOperand(
    hal::Operand* operand, std::vector<int32_t> dimensions) {
  if (dimensions.empty()) {
    for (uint32_t i = 0; i < operand->type.dimension_count; i++) {
      dimensions.push_back(operand->type.dimensions[i]);
    }
  }
  auto attr = std::make_shared<aml::nn::TensorAttr>();
  attr->name = GetTensorName(operand);
  attr->role = !IsConstantOperand(operand) ? aml::nn::TensorRole::VAR
                                           : aml::nn::TensorRole::CONST;
  attr->dims = ConvertDimensions(&dimensions[0], dimensions.size());
  attr->precision = ConvertPrecision(operand->type.precision);
  attr->layout = ConvertDataLayout(operand->type.layout);
  switch (operand->type.precision) {
    case NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER:
      attr->qntBits = 8;
      attr->qntType = aml::nn::QuantizationType::AFFINE_ASYMMETRIC;
      attr->qntParamAffineAsymmetric.scale.resize(1);
      attr->qntParamAffineAsymmetric.scale[0] =
          operand->type.asymm_per_layer_params.scale;
      attr->qntParamAffineAsymmetric.zero_point.resize(1);
      attr->qntParamAffineAsymmetric.zero_point[0] =
          operand->type.asymm_per_layer_params.zero_point;
      break;
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER:
      attr->qntBits = 32;
      attr->qntType = aml::nn::QuantizationType::SYMMETRIC;
      attr->qntParamSymmetric.scale.resize(1);
      attr->qntParamSymmetric.scale[0] =
          operand->type.symm_per_layer_params.scale;
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Can not convert an operand@0x" << std::hex
                           << operand << " with precision="
                           << OperandPrecisionCodeToString(
                                  operand->type.precision)
                           << " to aml::nn::Tensor !";
      break;
  }
  auto tensor = graph_->CreateTensor(
      attr,
      operand->buffer,
      IsModelInputOperand(operand) || IsModelOutputOperand(operand));
  NNADAPTER_CHECK(tensor);
  // Use to find the tensor based on the pointer of operand
  UpdateTensorMap(operand, tensor);
  return tensor;
}

}  // namespace amlogic_npu
}  // namespace nnadapter
