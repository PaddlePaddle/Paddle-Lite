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

#include "driver/huawei_ascend_npu/converter.h"
#include <map>
#include <utility>
#include "driver/huawei_ascend_npu/optimizer/fix_multiple_outputs_ops.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

Device::Device() { InitializeAscendDevice(); }

Device::~Device() {
  // TODO(hong19860320) fix the problem destruction order that the resource of
  // ACL is released before the function is called.
  // FinalizeAscendDevice();
}

Context::Context(void* device, const char* properties) : device_(device) {
  // Extract the runtime parameters from the context properties
  NNADAPTER_LOG(INFO) << "properties: " << std::string(properties);
  auto key_values = GetKeyValues(properties);
  if (key_values.count("HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS")) {
    auto selected_device_ids = string_split<int>(
        key_values["HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS"], ",");
    NNADAPTER_CHECK_GE(selected_device_ids.size(), 1);
    // Only supports specifying one device
    if (selected_device_ids.size() > 1) {
      NNADAPTER_LOG(WARNING) << "Only supports specifying one device, so the "
                                "first one is selected and others will be "
                                "ignored.";
      selected_device_ids_.push_back(selected_device_ids[0]);
    }
  }
  if (selected_device_ids_.empty()) {
    selected_device_ids_.push_back(0);
  }
  NNADAPTER_LOG(INFO) << "selected device ids: ";
  for (auto& selected_device_id : selected_device_ids_) {
    NNADAPTER_LOG(INFO) << selected_device_id;
  }
}

Context::~Context() {}

Program::~Program() { Clear(); }

void Program::Clear() {
  operators_.clear();
  model_client_ = nullptr;
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
        case NNADAPTER_RESIZE_NEAREST:
          ConvertResizeNearest(operation);
          break;
        case NNADAPTER_RESIZE_LINEAR:
          ConvertResizeLinear(operation);
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
        case NNADAPTER_ABS:
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
        case NNADAPTER_CAST:
          ConvertCast(operation);
          break;
        case NNADAPTER_SHAPE:
          ConvertShape(operation);
          break;
        case NNADAPTER_ASSIGN:
          ConvertAssign(operation);
          break;
        case NNADAPTER_LP_NORMALIZATION:
          ConvertLpNormalization(operation);
          break;
        case NNADAPTER_DEFORMABLE_CONV_2D:
          ConvertDeformableConv2d(operation);
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
    // Build a GE graph to a CANN OM model, and serialize it into a buffer
    if (!BuildOMModelToBuffer(
            input_operators, output_operators, model_buffer)) {
      NNADAPTER_LOG(FATAL)
          << "Failed to build a CANN OM model and serialize it into a buffer!";
      return NNADAPTER_DEVICE_INTERNAL_ERROR;
    } else {
      NNADAPTER_VLOG(3)
          << "Build a CANN OM model and serialize it into a buffer success.";
    }
  }
  NNADAPTER_CHECK(model_buffer);
  // Load a CANN OM model from a buffer, and create a CANN model manager
  // client(from CANN service) for inference
  model_client_ =
      LoadOMModelFromBuffer(*model_buffer, context_->GetFirstDeviceID());
  if (!model_client_) {
    NNADAPTER_LOG(FATAL) << "Failed to load a CANN OM model from a buffer!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  // Initialize the CANN input and output tensors
  std::vector<ge::TensorDesc> input_tensor_descs, output_tensor_descs;
  if (!model_client_->GetModelIOTensorDim(&input_tensor_descs,
                                          &output_tensor_descs)) {
    NNADAPTER_LOG(FATAL) << "Failed to call GetModelIOTensorDim to get the "
                            "description of input and output tensors!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  auto input_count = input_sizes.size();
  auto output_count = output_sizes.size();
  NNADAPTER_CHECK_EQ(input_tensor_descs.size(), input_count);
  NNADAPTER_CHECK_EQ(output_tensor_descs.size(), output_count);
  for (size_t i = 0; i < input_count; i++) {
    auto shape = input_tensor_descs[i].GetShape();
    NNADAPTER_VLOG(3) << "CANN input tensors[" << i
                      << "]: " << GEShapeToString(shape);
    NNADAPTER_CHECK_EQ(input_sizes[i], ProductionOfGEShape(shape));
  }
  for (size_t i = 0; i < output_count; i++) {
    auto shape = output_tensor_descs[i].GetShape();
    NNADAPTER_VLOG(3) << "CANN output tensors[" << i
                      << "]: " << GEShapeToString(shape);
    NNADAPTER_CHECK_EQ(output_sizes[i], ProductionOfGEShape(shape));
  }
  NNADAPTER_VLOG(3) << "Build success.";
  return NNADAPTER_NO_ERROR;
}

int Program::Execute(uint32_t input_count,
                     hal::Argument* input_arguments,
                     uint32_t output_count,
                     hal::Argument* output_arguments) {
  NNADAPTER_CHECK(model_client_->Process(
      input_count, input_arguments, output_count, output_arguments));
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
  auto op = std::make_shared<ge::op::Const>();
  auto tensor = std::make_shared<ge::Tensor>();
  tensor->SetTensorDesc(*tensor_desc);
  tensor->SetData(reinterpret_cast<const uint8_t*>(values),
                  num_values * OperandPrecisionLength(precision));
  op->set_attr_value(*tensor);
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
    auto op = std::make_shared<ge::op::Const>(name);
    auto tensor = std::make_shared<ge::Tensor>();
    tensor->SetTensorDesc(*tensor_desc);
    tensor->SetData(reinterpret_cast<const uint8_t*>(operand->buffer),
                    operand->length);
    op->set_attr_value(*tensor);
    auto constant_operator =
        std::make_shared<Operator>(op, tensor_desc, "", -1);
    UpdateOperatorMap(operand, constant_operator);
    return constant_operator;
  } else if (IsModelInputOperand(operand)) {
    auto op = std::make_shared<ge::op::Data>(name);
    op->update_input_desc_x(*tensor_desc);
    op->update_output_desc_y(*tensor_desc);
    auto data_operator = std::make_shared<Operator>(op, tensor_desc, "", -1);
    UpdateOperatorMap(operand, data_operator);
    return data_operator;
  }
  NNADAPTER_LOG(FATAL) << "Only constant and model input operands can be "
                          "converted to ge::Operator!";
  return nullptr;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
