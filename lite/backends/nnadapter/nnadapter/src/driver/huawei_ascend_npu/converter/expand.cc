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

#include "operation/expand.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertExpand(Converter* converter, core::Operation* operation) {
  EXPAND_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  std::shared_ptr<Operator> shape_operator = nullptr;
  if (IsTemporaryShapeOperand(shape_operand)) {
    if (IsOperandWithDynamicShape(shape_operand)) {
      shape_operator = converter->GetMappedOperator(shape_operand);
      if (!shape_operator) {
        shape_operator = converter->ConvertOperand(shape_operand);
      }
    } else {
      auto& temporary_shape = *(GetTemporaryShape(shape_operand));
      auto shape_count = temporary_shape.count;
      auto shape_data = temporary_shape.data;
      std::vector<int> expand_shape(shape_count);
      operation::UpdateExpandInferOutputShape(
          input_operand->type.dimensions.data,
          input_operand->type.dimensions.count,
          expand_shape.data(),
          shape_count,
          shape_data);
      shape_operator = converter->AddInt32ConstantOperator(expand_shape);
    }
  } else if (IsConstantOperand(shape_operand)) {
    auto shape_count = shape_operand->length / sizeof(int32_t);
    auto shape_data = reinterpret_cast<int32_t*>(shape_operand->buffer);
    std::vector<int> expand_shape(shape_count);
    operation::UpdateExpandInferOutputShape(
        input_operand->type.dimensions.data,
        input_operand->type.dimensions.count,
        expand_shape.data(),
        shape_count,
        shape_data);
    shape_operator = converter->AddInt32ConstantOperator(expand_shape);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported shape lifetime: "
                         << OperandLifetimeCodeToString(
                                shape_operand->type.lifetime);
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto expand_op = converter->AddOperator<ge::op::Expand>(output_operand);
  SET_INPUT(expand_op, x, input_operator);
  SET_INPUT(expand_op, shape, shape_operator);
  MAP_OUTPUT(expand_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
