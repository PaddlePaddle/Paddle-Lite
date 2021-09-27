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

#include "core/operation/fill.h"

#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertFill(Converter* converter, hal::Operation* operation) {
  FILL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // If lite time of shape_operand and value_operand are
  // NNADAPTER_CONSTANT_COPY,
  // change to const op.
  // if (shape_operand->type.lifetime == NNADAPTER_CONSTANT_COPY &&
  //     value_operand->type.lifetime == NNADAPTER_CONSTANT_COPY) {
  //   std::vector<int32_t> dimensions = {};
  //   for (uint32_t i = 0; i < shape_operand->type.dimensions.count; i++) {
  //     dimensions.push_back(shape_operand->type.dimensions.data[i]);
  //   }
  //   auto shape = dimensions.size() > 0
  //                    ? ge::Shape(ConvertToGEDimensions(dimensions))
  //                    : ge::Shape();
  //   auto tensor_desc = std::make_shared<ge::TensorDesc>(
  //       shape,
  //       ge::FORMAT_NCHW,
  //       ConvertToGEPrecision(value_operand->type.precision));
  //   auto op = converter->AddOperator<ge::op::Const>(output_operand);
  //   auto tensor = std::make_shared<ge::Tensor>();
  //   tensor->SetTensorDesc(*tensor_desc);
  //   tensor->SetData(reinterpret_cast<const uint8_t*>(value_operand->buffer),
  //                   value_operand->length);
  //   op->set_attr_value(*tensor);
  //   auto constant_operator =
  //       std::make_shared<Operator>(op, tensor_desc, "", -1);
  //   converter->UpdateOperatorMap(output_operand, constant_operator);
  //   return NNADAPTER_NO_ERROR;
  // }

  // Convert to GE operators
  auto shape_operator = converter->GetMappedOperator(shape_operand);
  if (shape_operator == nullptr) {
    shape_operator = converter->ConvertOperand(shape_operand);
  }
  auto value_operator = converter->GetMappedOperator(value_operand);
  if (value_operator == nullptr) {
    value_operator = converter->ConvertOperand(value_operand);
  }
  auto fill_op = converter->AddOperator<ge::op::Fill>(output_operand);
  SET_INPUT(fill_op, dims, shape_operator);
  SET_INPUT(fill_op, value, value_operator);
  MAP_OUTPUT(fill_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
