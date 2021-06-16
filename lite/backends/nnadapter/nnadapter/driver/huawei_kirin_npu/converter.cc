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
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_kirin_npu {

Context::Context() {
  // TODO(hong19860320) create the raw context from HiAI service
}

Context::~Context() {}

Program::~Program() {}

int Program::Build(hal::Model* model, hal::Cache* cache) {
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  // Convert a NNAdapter model to a HiAI IR graph
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
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
      default:
        NNADAPTER_LOG(FATAL) << "Unsupported operation("
                             << OperationTypeToString(operation->type)
                             << ") is found.";
        break;
    }
  }
  // Indentify the inputs and outputs
  auto input_count = model->input_operands.size();
  std::vector<ge::Operator> input_operators(input_count);
  for (size_t i = 0; i < input_count; i++) {
    auto operand = model->input_operands[i];
    NNADAPTER_CHECK(operators_.find(operand) != operators_.end());
    input_operators[i] = *operators_[operand].back();
  }
  auto output_count = model->output_operands.size();
  std::vector<ge::Operator> output_operators(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto operand = model->output_operands[i];
    NNADAPTER_CHECK(operators_.find(operand) != operators_.end());
    output_operators[i] = *operators_[operand].back();
  }
  // Build a HiAI IR graph to a HiAI OM model, and serialize it into a buffer
  std::vector<char> model_buffer;
  if (!BuildOMModelToBuffer(input_operators, output_operators, &model_buffer)) {
    NNADAPTER_LOG(FATAL)
        << "Failed to build a HiAI OM model and serialize it into a buffer!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  // Load a HiAI OM model from a buffer, and create a HiAI model manager
  // client(from HiAI service) for inference
  bool model_comp = true;
  model_name_ = string_format("@0x%X", model);
  model_client_ = LoadOMModelFromBuffer(model_name_,
                                        &model_buffer,
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
  NNADAPTER_CHECK_EQ(input_dimensions.size(), input_count);
  NNADAPTER_CHECK_EQ(output_dimensions.size(), output_count);
  input_tensors_.resize(input_count);
  output_tensors_.resize(output_count);
  for (size_t i = 0; i < input_count; i++) {
    auto n = input_dimensions[i].GetNumber();
    auto c = input_dimensions[i].GetChannel();
    auto h = input_dimensions[i].GetHeight();
    auto w = input_dimensions[i].GetWidth();
    NNADAPTER_VLOG(3) << "HiAI input tensors[" << i << "]: " << n << "," << c
                      << "," << h << "," << w;
    NNADAPTER_CHECK_EQ(
        ProductionOfDimensions(model->input_operands[i]->type.dimensions,
                               model->input_operands[i]->type.dimension_count),
        n * c * h * w);
    input_tensors_[i].reset(new hiai::AiTensor);
    input_tensors_[i]->Init(&(input_dimensions[i]));
  }
  for (size_t i = 0; i < output_count; i++) {
    auto n = output_dimensions[i].GetNumber();
    auto c = output_dimensions[i].GetChannel();
    auto h = output_dimensions[i].GetHeight();
    auto w = output_dimensions[i].GetWidth();
    NNADAPTER_VLOG(3) << "HiAI output tensors[" << i << "]: " << n << "," << c
                      << "," << h << "," << w;
    NNADAPTER_CHECK_EQ(
        ProductionOfDimensions(model->output_operands[i]->type.dimensions,
                               model->output_operands[i]->type.dimension_count),
        n * c * h * w);
    output_tensors_[i].reset(new hiai::AiTensor);
    output_tensors_[i]->Init(&(output_dimensions[i]));
  }
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

std::shared_ptr<ge::Operator> Program::ConvertOperand(
    hal::Operand* operand, std::vector<int64_t> dimensions) {
  if (operators_.find(operand) != operators_.end()) {
    return operators_.at(operand).back();
  }
  auto& type = operand->type;
  auto is_constant_copy = type.lifetime == NNADAPTER_CONSTANT_COPY;
  auto is_constant_reference = type.lifetime == NNADAPTER_CONSTANT_REFERENCE;
  auto is_constant = is_constant_copy || is_constant_reference;
  auto shape =
      ge::Shape(dimensions.empty()
                    ? ConvertDimensions(type.dimensions, type.dimension_count)
                    : dimensions);
  auto layout = ConvertDataLayout(type.layout);
  auto precision = ConvertPrecision(type.precision);
  ge::TensorDesc desc(shape, layout, precision);
  if (is_constant) {
    ge::TensorPtr tensor = std::make_shared<ge::Tensor>();
    tensor->SetTensorDesc(desc);
    tensor->SetData(reinterpret_cast<const uint8_t*>(operand->buffer),
                    operand->length);
    auto constant_operator = AddOperator<ge::op::Const>(operand);
    constant_operator->set_attr_value(tensor);
    return constant_operator;
  }
  // Only mapping the temporary operand
  auto data_operator = AddOperator<ge::op::Data>(operand);
  data_operator->update_input_desc_x(desc);
  return data_operator;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
