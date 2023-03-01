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

#include "operation/mat_mul.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertMatMul(Converter* converter, core::Operation* operation) {
  MAT_MUL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  std::shared_ptr<Operator> x_operator = nullptr;
  std::shared_ptr<Operator> y_operator = nullptr;
  if (IsDynamicShapeOperandType(x_operand->type)) {
    // Convert to GE operators
    x_operator = converter->GetMappedOperator(x_operand);
    if (x_operator == nullptr) {
      x_operator = converter->ConvertOperand(x_operand);
    }
    y_operator = converter->GetMappedOperator(y_operand);
    if (y_operator == nullptr) {
      y_operator = converter->ConvertOperand(y_operand);
    }
    // Resize dim 1 to 2
    if (x_operand->type.dimensions.count == 1) {
      auto unsqueeze_op = converter->AddOperator<ge::op::Unsqueeze>(
          output_operand, "reshape_x");
      unsqueeze_op->set_attr_axes(
          ge::Operator::OpListInt(std::vector<int64_t>({0})));
      SET_INPUT(unsqueeze_op, x, x_operator);
      x_operator = MAP_OUTPUT(unsqueeze_op, y, output_operand);
    }
    if (y_operand->type.dimensions.count == 1) {
      auto unsqueeze_op = converter->AddOperator<ge::op::Unsqueeze>(
          output_operand, "reshape_y");
      unsqueeze_op->set_attr_axes(
          ge::Operator::OpListInt(std::vector<int64_t>({1})));
      SET_INPUT(unsqueeze_op, x, y_operator);
      y_operator = MAP_OUTPUT(unsqueeze_op, y, output_operand);
    }
  } else {
    // Resize dim 1 to 2
    if (x_operand->type.dimensions.count == 1) {
      x_operand->type.dimensions.count = 2;
      x_operand->type.dimensions.data[1] = x_operand->type.dimensions.data[0];
      x_operand->type.dimensions.data[0] = 1;
    }
    if (y_operand->type.dimensions.count == 1) {
      y_operand->type.dimensions.count = 2;
      y_operand->type.dimensions.data[1] = 1;
    }
    // Convert to GE operators
    x_operator = converter->GetMappedOperator(x_operand);
    if (x_operator == nullptr) {
      x_operator = converter->ConvertOperand(x_operand);
    }
    y_operator = converter->GetMappedOperator(y_operand);
    if (y_operator == nullptr) {
      y_operator = converter->ConvertOperand(y_operand);
    }
  }

  auto mat_mul_op = converter->AddOperator<ge::op::BatchMatMul>(output_operand);
  mat_mul_op->set_attr_adj_x1(transpose_x);
  mat_mul_op->set_attr_adj_x2(transpose_y);
  SET_INPUT(mat_mul_op, x1, x_operator);
  SET_INPUT(mat_mul_op, x2, y_operator);
  MAP_OUTPUT(mat_mul_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
