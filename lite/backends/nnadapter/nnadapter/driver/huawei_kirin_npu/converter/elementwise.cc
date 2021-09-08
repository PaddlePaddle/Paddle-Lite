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

#include "core/operation/elementwise.h"
#include "driver/huawei_kirin_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int Program::ConvertElementwise(hal::Operation* operation) {
  ELEMENTWISE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input0_operator = GetMappedOperator(input0_operand);
  if (!input0_operator) {
    input0_operator = ConvertOperand(input0_operand);
  }
  auto input1_operator = GetMappedOperator(input1_operand);
  if (!input1_operator) {
    input1_operator = ConvertOperand(input1_operand);
  }
  auto eltwise_name = GetOperatorName(output_operand);
  std::shared_ptr<Operator> eltwise_operator = nullptr;
  switch (operation->type) {
#define CONVERT_ELEMENTWISE(type, class_name)                               \
  case NNADAPTER_##type: {                                                  \
    auto eltwise_op = std::make_shared<hiai::op::class_name>(eltwise_name); \
    SET_INPUT(eltwise_op, x1, input0_operator);                             \
    SET_INPUT(eltwise_op, x2, input1_operator);                             \
    eltwise_operator = MAP_OUTPUT(eltwise_op, y, output_operand);           \
  } break;
    CONVERT_ELEMENTWISE(ADD, Add);
    CONVERT_ELEMENTWISE(SUB, Sub);
    CONVERT_ELEMENTWISE(MUL, Mul);
    CONVERT_ELEMENTWISE(DIV, RealDiv);
#undef CONVERT_ELEMENTWISE
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported element-wise operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  if (fuse_code != NNADAPTER_FUSED_NONE) {
    auto act_name = GetOperatorName(output_operand);
    auto act_op = std::make_shared<hiai::op::Activation>(act_name);
    act_op->set_attr_mode(ConvertFuseCode(fuse_code));
    SET_INPUT(act_op, x, eltwise_operator);
    MAP_OUTPUT(act_op, y, output_operand);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
