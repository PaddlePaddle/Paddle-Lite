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

#include "driver/cambricon_mlu/converter.h"
#include <unistd.h>
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace cambricon_mlu {

#define REGISTER_CONVERTER(__op_type__, __func_name__) \
  extern int __func_name__(Converter* converter, core::Operation* operation);
#include "driver/cambricon_mlu/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_CAMBRICON_MLU_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER

magicmind::INetwork* Converter::network() { return network_; }

const std::string& Converter::op_params() { return op_params_; }

int Converter::Apply(core::Model* model) {
  // Convert the NNAdapter operations to MagicMind nodes
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
#include "driver/cambricon_mlu/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_CAMBRICON_MLU_CONVERTER_ALL_H__
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

magicmind::ITensor* Converter::GetMappedTensor(core::Operand* operand) {
  auto it = tensors_->find(operand);
  if (it != tensors_->end()) {
    return it->second.back();
  }
  return nullptr;
}

void Converter::UpdateTensorMap(core::Operand* operand,
                                magicmind::ITensor* tensor) {
  auto it = tensors_->find(operand);
  if (it == tensors_->end()) {
    auto result = tensors_->insert(
        std::make_pair(operand, std::vector<magicmind::ITensor*>()));
    NNADAPTER_CHECK(result.second);
    it = result.first;
  }
  it->second.push_back(tensor);
  return;
}

magicmind::ITensor* Converter::AddInt32ConstantTensor(
    void* buffer, std::vector<int64_t> dimensions) {
  NNADAPTER_CHECK(buffer);
  auto const_node = network_->AddIConstNode(
      magicmind::DataType::INT32, magicmind::Dims(dimensions), buffer);
  if (const_node == nullptr) {
    NNADAPTER_LOG(FATAL) << "Failed to add const node.";
  }
  return const_node->GetOutput(0);
}

magicmind::ITensor* Converter::AddFloat32ConstantTensor(
    void* buffer, std::vector<int64_t> dimensions) {
  NNADAPTER_CHECK(buffer);
  auto const_node = network_->AddIConstNode(
      magicmind::DataType::FLOAT32, magicmind::Dims(dimensions), buffer);
  NNADAPTER_CHECK(const_node) << "Failed to add const node.";
  return const_node->GetOutput(0);
}

magicmind::ITensor* Converter::AddTensor(const NNAdapterOperandType* type,
                                         void* buffer,
                                         std::vector<int64_t> dimensions) {
  NNADAPTER_CHECK(buffer);
  if (dimensions.empty()) {
    for (uint32_t i = 0; i < type->dimensions.count; i++) {
      dimensions.push_back(static_cast<int64_t>(type->dimensions.data[i]));
    }
  }
  auto mm_dtype = ConvertToMagicMindDtype(type->precision);
  auto const_node =
      network_->AddIConstNode(mm_dtype, magicmind::Dims(dimensions), buffer);
  NNADAPTER_CHECK(const_node) << "Failed to add const node.";
  return const_node->GetOutput(0);
}

magicmind::ITensor* Converter::AddTensor(const NNAdapterOperandType* type,
                                         std::vector<int64_t> dimensions) {
  if (dimensions.empty()) {
    for (uint32_t i = 0; i < type->dimensions.count; i++) {
      dimensions.push_back(static_cast<int64_t>(type->dimensions.data[i]));
    }
  }
  auto mm_dtype = ConvertToMagicMindDtype(type->precision);
  return network_->AddInput(mm_dtype, magicmind::Dims(dimensions));
}

magicmind::ITensor* Converter::ConvertOperand(core::Operand* operand,
                                              std::vector<int64_t> dimensions) {
  magicmind::ITensor* rst = nullptr;
  if (IsConstantOperand(operand)) {
    auto constant_tensor =
        AddTensor(&operand->type, operand->buffer, dimensions);
    UpdateTensorMap(operand, constant_tensor);
    if (constant_tensor->GetDataType() == magicmind::DataType::INT64) {
      auto cast_node =
          network_->AddICastNode(constant_tensor, magicmind::DataType::INT32);
      NNADAPTER_CHECK(cast_node) << "Failed to add cast node.";
      auto cast_out_tensor = cast_node->GetOutput(0);
      UpdateTensorMap(operand, cast_out_tensor);
      rst = cast_out_tensor;
    } else {
      rst = constant_tensor;
    }
  } else if (IsModelInputOperand(operand)) {
    auto input_tensor = AddTensor(&operand->type, dimensions);
    UpdateTensorMap(operand, input_tensor);
    if (input_tensor->GetDataType() == magicmind::DataType::INT64) {
      auto cast_node =
          network_->AddICastNode(input_tensor, magicmind::DataType::INT32);
      NNADAPTER_CHECK(cast_node) << "Failed to add cast node.";
      auto cast_out_tensor = cast_node->GetOutput(0);
      UpdateTensorMap(operand, cast_out_tensor);
      rst = cast_out_tensor;
    } else {
      rst = input_tensor;
    }
  } else {
    NNADAPTER_LOG(FATAL) << "Only constant and model input operands can be "
                            "converted to camb_tensor!";
  }
  return rst;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
