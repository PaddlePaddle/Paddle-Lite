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

#include "operation/shape.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertShape(Converter* converter, core::Operation* operation) {
  SHAPE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto shape_op = converter->AddOperator<ge::op::Shape>(output_operand);
  switch (dtype) {
    case NNADAPTER_INT32:
      shape_op->set_attr_dtype(ge::DT_INT32);
      break;
    case NNADAPTER_INT64:
      shape_op->set_attr_dtype(ge::DT_INT64);
      break;
    default:
      NNADAPTER_LOG(ERROR) << "Unsupported output data type: "
                           << OperandPrecisionCodeToString(dtype);
  }
  SET_INPUT(shape_op, x, input_operator);
  MAP_OUTPUT(shape_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
