// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/huawei_ascend_npu/converter/converter.h"
#include <utility>
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

#define REGISTER_CONVERTER(__op_type__, __func_name__) \
  extern int __func_name__(Converter* converter, core::Operation* operation);
#include "driver/huawei_ascend_npu/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_HUAWEI_ASCEND_NPU_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER

int Converter::Apply(core::Model* model) {
  operator_index_ = 0;
  // Convert the NNAdapter operations to GE operators
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
#define REGISTER_CONVERTER(__op_type__, __func_name__) \
  case NNADAPTER_##__op_type__:                        \
    __func_name__(this, operation);                    \
    break;
#include "driver/huawei_ascend_npu/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_HUAWEI_ASCEND_NPU_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER
      default:
        NNADAPTER_LOG(FATAL) << "Unsupported operation("
                             << OperationTypeToString(operation->type)
                             << ") is found.";
        break;
    }
  }
  return NNADAPTER_NO_ERROR;
}

std::shared_ptr<Operator> Converter::GetMappedOperator(core::Operand* operand) {
  auto it = operators_->find(operand);
  if (it != operators_->end()) {
    return it->second.back();
  }
  return nullptr;
}

std::shared_ptr<Operator> Converter::UpdateOperatorMap(
    core::Operand* operand, std::shared_ptr<Operator> op) {
  auto it = operators_->find(operand);
  if (it == operators_->end()) {
    auto result = operators_->insert(
        std::make_pair(operand, std::vector<std::shared_ptr<Operator>>()));
    NNADAPTER_CHECK(result.second);
    it = result.first;
  }
  it->second.push_back(op);
  return op;
}

std::shared_ptr<Operator> Converter::AddConstantOperator(
    const void* values,
    NNAdapterOperandPrecisionCode precision,
    const std::vector<int32_t>& dimensions) {
  NNADAPTER_CHECK(values)
      << "The values of constant operator should not be nullptr.";
  auto num_values = ProductionOfDimensions(dimensions);
  auto shape = dimensions.size() > 0
                   ? ge::Shape(ConvertToGEDimensions(dimensions))
                   : ge::Shape();
  auto tensor_desc = std::make_shared<ge::TensorDesc>(
      shape, ge::FORMAT_NCHW, ConvertToGEPrecision(precision));
  // Add anonymous constant operator
  auto op = AddOperator<ge::op::Const>();
  auto tensor = std::make_shared<ge::Tensor>();
  tensor->SetTensorDesc(*tensor_desc);
  tensor->SetData(reinterpret_cast<const uint8_t*>(values),
                  num_values * GetOperandPrecisionDataLength(precision));
  op->set_attr_value(*tensor);
  auto constant_operator = std::make_shared<Operator>(op, tensor_desc, "", -1);
  UpdateOperatorMap(nullptr, constant_operator);
  return constant_operator;
}

std::shared_ptr<Operator> Converter::AddZeroConstantOperator(
    NNAdapterOperandPrecisionCode precision,
    const std::vector<int32_t>& dimensions) {
  auto precision_data_length = GetOperandPrecisionDataLength(precision);
  auto num_values = ProductionOfDimensions(dimensions);
  std::vector<uint8_t> zero_values(precision_data_length * num_values, 0);
  return AddConstantOperator(&zero_values[0], precision, dimensions);
}

std::shared_ptr<Operator> Converter::AddInt32ConstantOperator(
    const int32_t* values, const std::vector<int32_t>& dimensions) {
  return AddConstantOperator(values, NNADAPTER_INT32, dimensions);
}

std::shared_ptr<Operator> Converter::AddInt32ConstantOperator(
    const std::vector<int32_t>& values,
    const std::vector<int32_t>& dimensions) {
  int num_values = values.size();
  return AddInt32ConstantOperator(
      &values[0],
      dimensions.empty() ? std::vector<int32_t>({num_values}) : dimensions);
}

std::shared_ptr<Operator> Converter::AddInt32ConstantOperator(
    const int32_t values) {
  auto num_values_vec = std::vector<int32_t>({values});
  return AddInt32ConstantOperator(&num_values_vec[0], std::vector<int32_t>());
}

std::shared_ptr<Operator> Converter::AddFloat32ConstantOperator(
    const float* values, const std::vector<int32_t>& dimensions) {
  return AddConstantOperator(values, NNADAPTER_FLOAT32, dimensions);
}

std::shared_ptr<Operator> Converter::AddFloat32ConstantOperator(
    const std::vector<float>& values, const std::vector<int32_t>& dimensions) {
  int num_values = values.size();
  return AddFloat32ConstantOperator(
      &values[0],
      dimensions.empty() ? std::vector<int32_t>({num_values}) : dimensions);
}

std::shared_ptr<Operator> Converter::AddUInt64ConstantOperator(
    const uint64_t* values, const std::vector<int32_t>& dimensions) {
  return AddConstantOperator(
      reinterpret_cast<const void*>(values), NNADAPTER_UINT64, dimensions);
}

std::shared_ptr<Operator> Converter::AddUInt64ConstantOperator(
    const std::vector<uint64_t>& values,
    const std::vector<int32_t>& dimensions) {
  int num_values = values.size();
  return AddUInt64ConstantOperator(
      &values[0],
      dimensions.empty() ? std::vector<int32_t>({num_values}) : dimensions);
}

std::shared_ptr<Operator> Converter::ConvertOperand(
    core::Operand* operand, std::vector<int32_t> dimensions) {
  if (dimensions.empty()) {
    for (uint32_t i = 0; i < operand->type.dimensions.count; i++) {
      dimensions.push_back(operand->type.dimensions.data[i]);
    }
  }
  auto shape = dimensions.size() > 0
                   ? ge::Shape(ConvertToGEDimensions(dimensions))
                   : ge::Shape();
  auto tensor_desc = std::make_shared<ge::TensorDesc>(
      shape, ge::FORMAT_NCHW, ConvertToGEPrecision(operand->type.precision));
  if (IsConstantOperand(operand)) {
    auto op = AddOperator<ge::op::Const>(operand);
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
    auto op = AddOperator<ge::op::Data>(operand);
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
