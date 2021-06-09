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

#include "driver/device/huawei_kirin_npu/converter.h"
#include "driver/utility/debug.h"
#include "driver/utility/modeling.h"
#include "utility/micros.h"

namespace nnadapter {
namespace driver {
namespace huawei_kirin_npu {

Context::Context() {
  // TODO(hong19860320) create the raw context from HiAI service
}

Context::~Context() {}

Program::~Program() {}

int Program::Build(Model* model, Cache* cache) {
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  // Convert a NNAdapter model to a HiAI IR graph
  std::vector<Operation*> operations = SortOperationsInTopologicalOrder(model);
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
        NNADAPTER_LOG(ERROR) << "Unsupported operation("
                             << OperationTypeToString(operation->type)
                             << ") is found.";
        break;
    }
  }
  // Indentify the inputs and outputs
  auto input_count = model->input_operands.size();
  std::vector<ge::Operator> input_nodes(input_count);
  for (size_t i = 0; i < input_count; i++) {
    auto operand = model->input_operands[i];
    NNADAPTER_CHECK(operators_.find(operand) != operators_.end());
    input_nodes[i] = *operators_[operand].back();
  }
  auto output_count = model->output_operands.size();
  std::vector<ge::Operator> output_nodes(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto operand = model->output_operands[i];
    NNADAPTER_CHECK(operators_.find(operand) != operators_.end());
    output_nodes[i] = *operators_[operand].back();
  }
  // Build a HiAI IR graph to a HiAI OM model, and serialize it into a buffer
  std::vector<char> model_buffer;
  if (!BuildOMModelToBuffer(input_nodes, output_nodes, &model_buffer)) {
    NNADAPTER_LOG(WARNING)
        << "Failed to build a HiAI OM model and serialize it into a buffer!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  // Load a HiAI OM model from a buffer, and create a HiAI model manager
  // client(from HiAI service) for inference
  bool model_comp = true;
  model_client_ = LoadOMModelFromBuffer(model_name_,
                                        &model_buffer,
                                        &model_comp,
                                        context_->freq_level(),
                                        context_->framework_type(),
                                        context_->model_type(),
                                        context_->device_type());
  if (!model_client_) {
    NNADAPTER_LOG(WARNING) << "Failed to load a HiAI OM model from a buffer!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  return NNADAPTER_NO_ERROR;
}

int Program::Execute(uint32_t input_count,
                     Argument* input_arguments,
                     uint32_t output_count,
                     Argument* output_arguments) {
  for (uint32_t i = 0; i < input_count; i++) {
    auto& argument = input_arguments[i];
  }
  for (uint32_t i = 0; i < output_count; i++) {
    auto& argument = output_arguments[i];
  }
  return NNADAPTER_NO_ERROR;
}

std::shared_ptr<ge::Operator> Program::ConvertOperand(
    Operand* operand, std::vector<int64_t> dimensions) {
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
}  // namespace driver
}  // namespace nnadapter
