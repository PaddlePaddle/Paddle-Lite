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

#include "driver/huawei_kirin_npu/converter.h"
#include <utility>
#include "driver/huawei_kirin_npu/optimizer/fix_multiple_outputs_ops.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_kirin_npu {

Context::Context(void* device, const char* properties) : device_(device) {
  // TODO(hong19860320) create the raw context from HiAI service
}

Context::~Context() {}

Program::~Program() { Clear(); }

void Program::Clear() {
  operators_.clear();
  model_name_ = "";
  model_client_ = nullptr;
  input_tensors_.clear();
  output_tensors_.clear();
}

int Program::Build(hal::Model* model, hal::Cache* cache) {
  Clear();
  std::vector<int64_t> input_sizes;
  std::vector<int64_t> output_sizes;
  std::vector<uint8_t> model_content;
  std::vector<uint8_t>* model_buffer = nullptr;
  if (!cache->buffer.empty()) {
    // Build from cache
    model_buffer = &cache->buffer;
    auto input_count = cache->input_types.size();
    NNADAPTER_VLOG(3) << "Model input count: " << input_count;
    if (input_count > 0) {
      input_sizes.resize(input_count);
      for (size_t i = 0; i < input_count; i++) {
        const auto& type = cache->input_types[i];
        input_sizes[i] =
            ProductionOfDimensions(type.dimensions, type.dimension_count);
      }
    }
    auto output_count = cache->output_types.size();
    NNADAPTER_VLOG(3) << "Model output count: " << output_count;
    NNADAPTER_CHECK_GT(output_count, 0);
    output_sizes.resize(output_count);
    for (size_t i = 0; i < output_count; i++) {
      const auto& type = cache->output_types[i];
      output_sizes[i] =
          ProductionOfDimensions(type.dimensions, type.dimension_count);
    }
  } else {
    // Build from model
    NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
    FixMultipleOutputsOps(model);
    NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
    // Convert a NNAdapter model to a GE graph
    std::vector<hal::Operation*> operations =
        SortOperationsInTopologicalOrder(model);
    for (auto operation : operations) {
      NNADAPTER_VLOG(5) << "Converting "
                        << OperationTypeToString(operation->type) << " ...";
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
        case NNADAPTER_CONCAT:
          ConvertConcat(operation);
          break;
        case NNADAPTER_CONV_2D:
          ConvertConv2D(operation);
          break;
        case NNADAPTER_FULLY_CONNECTED:
          ConvertFullyConnected(operation);
          break;
        case NNADAPTER_RELU:
        case NNADAPTER_RELU6:
        case NNADAPTER_SIGMOID:
        case NNADAPTER_TANH:
          ConvertActivation(operation);
          break;
        case NNADAPTER_RESHAPE:
          ConvertReshape(operation);
          break;
        case NNADAPTER_SOFTMAX:
          ConvertSoftmax(operation);
          break;
        case NNADAPTER_SPLIT:
          ConvertSplit(operation);
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
    // Identify the inputs and outputs
    auto input_count = model->input_operands.size();
    NNADAPTER_VLOG(3) << "Model input count: " << input_count;
    std::vector<ge::Operator> input_operators;
    if (input_count > 0) {
      input_operators.resize(input_count);
      input_sizes.resize(input_count);
      for (size_t i = 0; i < input_count; i++) {
        auto operand = model->input_operands[i];
        NNADAPTER_CHECK(operators_.find(operand) != operators_.end());
        input_operators[i] = *operators_[operand].back()->op();
        input_sizes[i] = ProductionOfDimensions(operand->type.dimensions,
                                                operand->type.dimension_count);
      }
    }
    auto output_count = model->output_operands.size();
    NNADAPTER_VLOG(3) << "Model output count: " << output_count;
    NNADAPTER_CHECK_GT(output_count, 0);
    std::vector<ge::Operator> output_operators(output_count);
    output_sizes.resize(output_count);
    for (size_t i = 0; i < output_count; i++) {
      auto operand = model->output_operands[i];
      NNADAPTER_CHECK(operators_.find(operand) != operators_.end());
      output_operators[i] = *operators_[operand].back()->op();
      output_sizes[i] = ProductionOfDimensions(operand->type.dimensions,
                                               operand->type.dimension_count);
    }
    if (cache->key && cache->dir) {
      model_buffer = &cache->buffer;
    } else {
      model_buffer = &model_content;
    }
    // Build a GE graph to a HiAI OM model, and serialize it into a buffer
    if (!BuildOMModelToBuffer(
            input_operators, output_operators, model_buffer)) {
      NNADAPTER_LOG(FATAL)
          << "Failed to build a HiAI OM model and serialize it into a buffer!";
      return NNADAPTER_DEVICE_INTERNAL_ERROR;
    } else {
      NNADAPTER_VLOG(3)
          << "Build a HiAI OM model and serialize it into a buffer success.";
    }
  }
  NNADAPTER_CHECK(model_buffer);
  // Load a HiAI OM model from a buffer, and create a HiAI model manager
  // client(from HiAI service) for inference
  bool model_comp = true;
  model_name_ = string_format("0x%X.om", cache);
  model_client_ = LoadOMModelFromBuffer(model_name_,
                                        model_buffer,
                                        &model_comp,
                                        context_->freq_level(),
                                        context_->framework_type(),
                                        context_->model_type(),
                                        context_->device_type());
  if (!model_client_) {
    NNADAPTER_LOG(FATAL) << "Failed to load a HiAI OM model from a buffer!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  // Initialize the HiAI input and output tensors
  std::vector<hiai::TensorDimension> input_dimensions, output_dimensions;
  if (model_client_->GetModelIOTensorDim(
          model_name_, input_dimensions, output_dimensions) !=
      hiai::AI_SUCCESS) {
    NNADAPTER_LOG(FATAL) << "Failed to call GetModelIOTensorDim to get the "
                            "dimensions of input and output tensors!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  auto input_count = input_sizes.size();
  NNADAPTER_CHECK_EQ(input_dimensions.size(), input_count);
  if (input_count > 0) {
    input_tensors_.resize(input_count);
    for (size_t i = 0; i < input_count; i++) {
      auto n = input_dimensions[i].GetNumber();
      auto c = input_dimensions[i].GetChannel();
      auto h = input_dimensions[i].GetHeight();
      auto w = input_dimensions[i].GetWidth();
      NNADAPTER_VLOG(3) << "HiAI input tensors[" << i << "]: " << n << "," << c
                        << "," << h << "," << w;
      NNADAPTER_CHECK_EQ(input_sizes[i], n * c * h * w);
      input_tensors_[i].reset(new hiai::AiTensor);
      input_tensors_[i]->Init(&(input_dimensions[i]));
    }
  }
  auto output_count = output_sizes.size();
  NNADAPTER_CHECK_EQ(output_dimensions.size(), output_count);
  output_tensors_.resize(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto n = output_dimensions[i].GetNumber();
    auto c = output_dimensions[i].GetChannel();
    auto h = output_dimensions[i].GetHeight();
    auto w = output_dimensions[i].GetWidth();
    NNADAPTER_VLOG(3) << "HiAI output tensors[" << i << "]: " << n << "," << c
                      << "," << h << "," << w;
    NNADAPTER_CHECK_EQ(output_sizes[i], n * c * h * w);
    output_tensors_[i].reset(new hiai::AiTensor);
    output_tensors_[i]->Init(&(output_dimensions[i]));
  }
  NNADAPTER_VLOG(3) << "Build success.";
  return NNADAPTER_NO_ERROR;
}

int Program::Execute(uint32_t input_count,
                     hal::Argument* input_arguments,
                     uint32_t output_count,
                     hal::Argument* output_arguments) {
  for (uint32_t i = 0; i < input_count; i++) {
    auto& argument = input_arguments[i];
    std::memcpy(input_tensors_[argument.index]->GetBuffer(),
                argument.buffer,
                argument.length);
  }
  std::string key = "model_name";  // Note: key seems must be model_name
  hiai::AiContext model_context;
  model_context.AddPara(key, model_name_);
  int istamp;
  NNADAPTER_CHECK_EQ(
      model_client_->Process(
          model_context, input_tensors_, output_tensors_, 1000, istamp),
      hiai::AI_SUCCESS);
  for (uint32_t i = 0; i < output_count; i++) {
    auto& argument = output_arguments[i];
    std::memcpy(argument.buffer,
                output_tensors_[argument.index]->GetBuffer(),
                argument.length);
  }
  return NNADAPTER_NO_ERROR;
}

std::string Program::GetOperatorName(hal::Operand* operand) {
  auto operand_id = OperandIdToString(operand);
  auto index = 0;
  auto it = operators_.find(operand);
  if (it != operators_.end()) {
    index = it->second.size();
  }
  return operand_id + string_format("_%d", index);
}

std::shared_ptr<Operator> Program::GetMappedOperator(hal::Operand* operand) {
  auto it = operators_.find(operand);
  if (it != operators_.end()) {
    return it->second.back();
  }
  return nullptr;
}

std::shared_ptr<Operator> Program::UpdateOperatorMap(
    hal::Operand* operand, std::shared_ptr<Operator> op) {
  auto it = operators_.find(operand);
  if (it == operators_.end()) {
    auto result = operators_.insert(
        std::make_pair(operand, std::vector<std::shared_ptr<Operator>>()));
    NNADAPTER_CHECK(result.second);
    it = result.first;
  }
  it->second.push_back(op);
  return op;
}

std::shared_ptr<Operator> Program::AddConstantOperator(
    const void* values,
    NNAdapterOperandPrecisionCode precision,
    const std::vector<int32_t>& dimensions) {
  NNADAPTER_CHECK(values)
      << "The values of constant operator should not be nullptr.";
  auto num_values = ProductionOfDimensions(dimensions);
  auto shape = dimensions.size() > 0 ? ge::Shape(ConvertDimensions(dimensions))
                                     : ge::Shape();
  auto tensor_desc = std::make_shared<ge::TensorDesc>(
      shape, ge::FORMAT_NCHW, ConvertPrecision(precision));
  // Add anonymous constant operator
  auto name = GetOperatorName(nullptr);
  auto op = std::make_shared<hiai::op::Const>(name);
  auto tensor = std::make_shared<ge::Tensor>();
  tensor->SetTensorDesc(*tensor_desc);
  tensor->SetData(reinterpret_cast<const uint8_t*>(values),
                  num_values * OperandPrecisionLength(precision));
  op->set_attr_value(tensor);
  auto constant_operator = std::make_shared<Operator>(op, tensor_desc, "", -1);
  UpdateOperatorMap(nullptr, constant_operator);
  return constant_operator;
}

std::shared_ptr<Operator> Program::AddInt32ConstantOperator(
    const int32_t* values, const std::vector<int32_t>& dimensions) {
  return AddConstantOperator(values, NNADAPTER_TENSOR_INT32, dimensions);
}

std::shared_ptr<Operator> Program::AddInt32ConstantOperator(
    const std::vector<int32_t>& values,
    const std::vector<int32_t>& dimensions) {
  int num_values = values.size();
  return AddInt32ConstantOperator(
      &values[0],
      dimensions.empty() ? std::vector<int32_t>({num_values}) : dimensions);
}

std::shared_ptr<Operator> Program::AddFloat32ConstantOperator(
    const float* values, const std::vector<int32_t>& dimensions) {
  return AddConstantOperator(values, NNADAPTER_TENSOR_FLOAT32, dimensions);
}

std::shared_ptr<Operator> Program::AddFloat32ConstantOperator(
    const std::vector<float>& values, const std::vector<int32_t>& dimensions) {
  int num_values = values.size();
  return AddFloat32ConstantOperator(
      &values[0],
      dimensions.empty() ? std::vector<int32_t>({num_values}) : dimensions);
}

std::shared_ptr<Operator> Program::ConvertOperand(
    hal::Operand* operand, std::vector<int32_t> dimensions) {
  if (dimensions.empty()) {
    for (uint32_t i = 0; i < operand->type.dimension_count; i++) {
      dimensions.push_back(operand->type.dimensions[i]);
    }
  }
  auto shape = dimensions.size() > 0 ? ge::Shape(ConvertDimensions(dimensions))
                                     : ge::Shape();
  auto tensor_desc = std::make_shared<ge::TensorDesc>(
      shape, ge::FORMAT_NCHW, ConvertPrecision(operand->type.precision));
  auto name = GetOperatorName(operand);
  if (IsConstantOperand(operand)) {
    auto op = std::make_shared<hiai::op::Const>(name);
    auto tensor = std::make_shared<ge::Tensor>();
    tensor->SetTensorDesc(*tensor_desc);
    tensor->SetData(reinterpret_cast<const uint8_t*>(operand->buffer),
                    operand->length);
    op->set_attr_value(tensor);
    auto constant_operator =
        std::make_shared<Operator>(op, tensor_desc, "", -1);
    UpdateOperatorMap(operand, constant_operator);
    return constant_operator;
  } else if (IsModelInputOperand(operand)) {
    auto op = std::make_shared<hiai::op::Data>(name);
    op->update_input_desc_x(*tensor_desc);
    auto data_operator = std::make_shared<Operator>(op, tensor_desc, "", -1);
    UpdateOperatorMap(operand, data_operator);
    return data_operator;
  }
  NNADAPTER_LOG(FATAL) << "Only constant and model input operands can be "
                          "converted to ge::Operator!";
  return nullptr;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
