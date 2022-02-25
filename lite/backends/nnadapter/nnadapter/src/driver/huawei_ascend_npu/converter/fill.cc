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

#include "operation/fill.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertFill(Converter* converter, core::Operation* operation) {
  FILL_OPERATION_EXTRACT_INPUTS_OUTPUTS

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
