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

#include "core/operation/expand.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertExpand(Converter* converter, hal::Operation* operation) {
  EXPAND_OPERATION_EXTRACT_INPUTS_OUTPUTS
  if (!IsConstantOperand(shape_operand)) {
    NNADAPTER_LOG(ERROR) << "Shape input only support const tensor.";
    return NNADAPTER_INVALID_PARAMETER;
  }

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto shape_operator = converter->ConvertOperand(shape_operand);
  auto expand_op = converter->AddOperator<ge::op::Expand>(output_operand);
  SET_INPUT(expand_op, x, input_operator);
  SET_INPUT(expand_op, shape, shape_operator);
  MAP_OUTPUT(expand_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
