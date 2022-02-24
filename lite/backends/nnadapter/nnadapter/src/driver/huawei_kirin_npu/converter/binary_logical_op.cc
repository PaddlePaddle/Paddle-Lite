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

#include "operation/binary_logical_op.h"
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertBinaryLogicalOp(Converter* converter, core::Operation* operation) {
  BINARY_LOGICAL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input0_operator = converter->GetMappedOperator(input0_operand);
  if (!input0_operator) {
    input0_operator = converter->ConvertOperand(input0_operand);
  }
  auto input1_operator = converter->GetMappedOperator(input1_operand);
  if (!input1_operator) {
    input1_operator = converter->ConvertOperand(input1_operand);
  }
  switch (operation->type) {
#define CONVERT_BINARY_LOGICAL_OP(type, class_name)                   \
  case NNADAPTER_##type: {                                            \
    auto binary_logical_op =                                          \
        converter->AddOperator<hiai::op::class_name>(output_operand); \
    SET_INPUT(binary_logical_op, x1, input0_operator);                \
    SET_INPUT(binary_logical_op, x2, input1_operator);                \
    MAP_OUTPUT(binary_logical_op, y, output_operand);                 \
  } break;
    CONVERT_BINARY_LOGICAL_OP(AND, LogicalAnd);
    CONVERT_BINARY_LOGICAL_OP(OR, LogicalOr);
    CONVERT_BINARY_LOGICAL_OP(XOR, LogicalXor);
#undef CONVERT_BINARY_LOGICAL_OP
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported binary logical operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
