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

#include "core/operation/mat_mul.h"
#include "driver/huawei_ascend_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int Program::ConvertMatMul(hal::Operation* operation) {
  MAT_MUL_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // TODO(zhupengyang): support by reshape or squeeze
  NNADAPTER_CHECK_NE(x_operand->type.dimension_count, 1);
  NNADAPTER_CHECK_NE(y_operand->type.dimension_count, 1);

  // Convert to GE operators
  auto x_operator = GetMappedOperator(x_operand);
  if (x_operator == nullptr) {
    x_operator = ConvertOperand(x_operand);
  }
  auto y_operator = GetMappedOperator(y_operand);
  if (y_operator == nullptr) {
    y_operator = ConvertOperand(y_operand);
  }
  auto mat_mul_name = GetOperatorName(output_operand);
  auto mat_mul_op = std::make_shared<ge::op::BatchMatMul>(mat_mul_name);
  mat_mul_op->set_attr_adj_x1(transpose_x);
  mat_mul_op->set_attr_adj_x2(transpose_y);
  SET_INPUT(mat_mul_op, x1, x_operator);
  SET_INPUT(mat_mul_op, x2, y_operator);
  MAP_OUTPUT(mat_mul_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
