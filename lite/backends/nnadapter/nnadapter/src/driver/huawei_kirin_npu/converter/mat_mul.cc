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

#include "operation/mat_mul.h"
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertMatMul(Converter* converter, core::Operation* operation) {
  MAT_MUL_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK_EQ(transpose_x, false) << "HiAI not support transpose x.";

  // Convert to GE operators
  auto x_operator = converter->GetMappedOperator(x_operand);
  if (x_operator == nullptr) {
    x_operator = converter->ConvertOperand(x_operand);
  }
  auto y_operator = converter->GetMappedOperator(y_operand);
  if (y_operator == nullptr) {
    y_operator = converter->ConvertOperand(y_operand);
  }
  if (x_operand->type.dimensions.count == 2 &&
      y_operand->type.dimensions.count == 2) {
    auto mat_mul_op = converter->AddOperator<hiai::op::MatMul>(output_operand);
    mat_mul_op->set_attr_transpose_x1(transpose_x);
    mat_mul_op->set_attr_transpose_x2(transpose_y);
    SET_INPUT(mat_mul_op, x1, x_operator);
    SET_INPUT(mat_mul_op, x2, y_operator);
    MAP_OUTPUT(mat_mul_op, y, output_operand);
  } else {
    auto mat_mul_op =
        converter->AddOperator<hiai::op::BatchMatMul>(output_operand);
    mat_mul_op->set_attr_adj_x1(transpose_x);
    mat_mul_op->set_attr_adj_x2(transpose_y);
    SET_INPUT(mat_mul_op, x1, x_operator);
    SET_INPUT(mat_mul_op, x2, y_operator);
    MAP_OUTPUT(mat_mul_op, y, output_operand);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
