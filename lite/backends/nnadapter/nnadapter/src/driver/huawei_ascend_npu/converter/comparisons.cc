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

#include "operation/comparisons.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertComparisons(Converter* converter, core::Operation* operation) {
  COMPARISONS_OPERATION_EXTRACT_INPUTS_OUTPUTS

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
#define CONVERT_COMPARISON(type, class_name)                                   \
  case NNADAPTER_##type: {                                                     \
    auto comp_op = converter->AddOperator<ge::op::class_name>(output_operand); \
    SET_INPUT(comp_op, x1, input0_operator);                                   \
    SET_INPUT(comp_op, x2, input1_operator);                                   \
    MAP_OUTPUT(comp_op, y, output_operand);                                    \
  } break;
    CONVERT_COMPARISON(EQUAL, Equal);
    CONVERT_COMPARISON(NOT_EQUAL, NotEqual);
    CONVERT_COMPARISON(GREATER, Greater);
    CONVERT_COMPARISON(GREATER_EQUAL, GreaterEqual);
    CONVERT_COMPARISON(LESS, Less);
    CONVERT_COMPARISON(LESS_EQUAL, LessEqual);
#undef CONVERT_COMPARISON
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported comparison operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
