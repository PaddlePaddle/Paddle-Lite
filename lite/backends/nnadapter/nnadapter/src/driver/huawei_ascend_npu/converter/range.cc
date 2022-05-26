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

#include "operation/range.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertRange(Converter* converter, core::Operation* operation) {
  RANGE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto start_operator = converter->GetMappedOperator(start_operand);
  if (!start_operator) {
    start_operator = converter->ConvertOperand(start_operand);
  }
  auto limit_operator = converter->GetMappedOperator(limit_operand);
  if (!limit_operator) {
    limit_operator = converter->ConvertOperand(limit_operand);
  }
  auto delta_operator = converter->GetMappedOperator(delta_operand);
  if (!delta_operator) {
    delta_operator = converter->ConvertOperand(delta_operand);
  }
  auto range_op = converter->AddOperator<ge::op::Range>(output_operand);
  SET_INPUT(range_op, start, start_operator);
  SET_INPUT(range_op, limit, limit_operator);
  SET_INPUT(range_op, delta, delta_operator);
  MAP_OUTPUT(range_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
