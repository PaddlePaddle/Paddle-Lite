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

#include "operation/unary_logical_op.h"
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertUnaryLogicalOp(Converter* converter, core::Operation* operation) {
  UNARY_LOGICAL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  switch (operation->type) {
#define CONVERT_UNARY_LOGICAL_OP(type, class_name)                    \
  case NNADAPTER_##type: {                                            \
    auto unary_logical_op =                                           \
        converter->AddOperator<hiai::op::class_name>(output_operand); \
    SET_INPUT(unary_logical_op, x, input_operator);                   \
    MAP_OUTPUT(unary_logical_op, y, output_operand);                  \
  } break;
    CONVERT_UNARY_LOGICAL_OP(NOT, LogicalNot);
#undef CONVERT_UNARY_LOGICAL_OP
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported unary logical operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
