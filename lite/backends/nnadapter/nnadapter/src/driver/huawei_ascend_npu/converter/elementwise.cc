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

#include "operation/elementwise.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertElementwise(Converter* converter, core::Operation* operation) {
  ELEMENTWISE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input0_operator = converter->GetMappedOperator(input0_operand);
  if (!input0_operator) {
    input0_operator = converter->ConvertOperand(input0_operand);
  }
  auto input1_operator = converter->GetMappedOperator(input1_operand);
  if (!input1_operator) {
    input1_operator = converter->ConvertOperand(input1_operand);
  }
  std::shared_ptr<Operator> eltwise_operator = nullptr;
  switch (operation->type) {
#define CONVERT_ELEMENTWISE(type, class_name)                       \
  case NNADAPTER_##type: {                                          \
    auto eltwise_op =                                               \
        converter->AddOperator<ge::op::class_name>(output_operand); \
    SET_INPUT(eltwise_op, x1, input0_operator);                     \
    SET_INPUT(eltwise_op, x2, input1_operator);                     \
    eltwise_operator = MAP_OUTPUT(eltwise_op, y, output_operand);   \
  } break;
    CONVERT_ELEMENTWISE(ADD, Add);
    CONVERT_ELEMENTWISE(SUB, Sub);
    CONVERT_ELEMENTWISE(MUL, Mul);
    CONVERT_ELEMENTWISE(DIV, Xdivy);
    CONVERT_ELEMENTWISE(MAX, Maximum);
    CONVERT_ELEMENTWISE(MIN, Minimum);
    CONVERT_ELEMENTWISE(POW, Pow);
#undef CONVERT_ELEMENTWISE
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported element-wise operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  // fuse activations ?
  switch (fuse_code) {
#define CONVERT_UNARY_ACTIVATION(type, class_name)                            \
  case NNADAPTER_FUSED_##type: {                                              \
    auto act_op = converter->AddOperator<ge::op::class_name>(output_operand); \
    SET_INPUT(act_op, x, eltwise_operator);                                   \
    MAP_OUTPUT(act_op, y, output_operand);                                    \
  } break;
    CONVERT_UNARY_ACTIVATION(RELU, Relu);
    CONVERT_UNARY_ACTIVATION(RELU6, Relu6);
#undef CONVERT_UNARY_ACTIVATION
    case NNADAPTER_FUSED_NONE:
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported fuse_code(" << fuse_code
                           << ") is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
