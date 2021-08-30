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

#include "core/operation/fill.h"
#include "driver/huawei_ascend_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int Program::ConvertFill(hal::Operation* operation) {
  FILL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  NNADAPTER_VLOG(5) << "--- ConvertFill, 0";
  // Convert to GE operators
  auto shape_operator = GetMappedOperator(shape_operand);
  if (shape_operator == nullptr) {
    shape_operator = ConvertOperand(shape_operand);
  }
  NNADAPTER_VLOG(5) << "--- ConvertFill, 1";
  auto value_operator = GetMappedOperator(value_operand);
  if (value_operator == nullptr) {
    value_operator = ConvertOperand(value_operand);
  }
  NNADAPTER_VLOG(5) << "--- ConvertFill, 2";
  auto fill_name = GetOperatorName(output_operand);
  auto fill_op = std::make_shared<ge::op::Fill>(fill_name);
  NNADAPTER_VLOG(5) << "--- ConvertFill, 3";
  SET_INPUT(fill_op, dims, shape_operator);
  NNADAPTER_VLOG(5) << "--- ConvertFill, 4";
  SET_INPUT(fill_op, value, value_operator);
  NNADAPTER_VLOG(5) << "--- ConvertFill, 5";
  MAP_OUTPUT(fill_op, y, output_operand);
  NNADAPTER_VLOG(5) << "--- ConvertFill, 6";
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
