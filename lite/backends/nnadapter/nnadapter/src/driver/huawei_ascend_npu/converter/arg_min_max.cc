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

#include "operation/arg_min_max.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertArgMinMax(Converter* converter, core::Operation* operation) {
  ARG_MIN_MAX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  std::shared_ptr<Operator> arg_operator = nullptr;
  if (operation->type == NNADAPTER_ARG_MAX) {
    auto arg_op = converter->AddOperator<ge::op::ArgMaxV2>(output_operand);
    auto dimension_operator = converter->AddInt32ConstantOperator({axis});
    arg_op->set_attr_dtype(ConvertToGEPrecision(dtype));
    SET_INPUT(arg_op, x, input_operator);
    SET_INPUT(arg_op, dimension, dimension_operator);
    arg_operator = MAP_OUTPUT(arg_op, y, output_operand);
  } else if (operation->type == NNADAPTER_ARG_MIN) {
    auto arg_op = converter->AddOperator<ge::op::ArgMinD>(output_operand);
    arg_op->set_attr_dimension(axis);
    arg_op->set_attr_dtype(ConvertToGEPrecision(dtype));
    SET_INPUT(arg_op, x, input_operator);
    arg_operator = MAP_OUTPUT(arg_op, y, output_operand);
    if (dtype == NNADAPTER_INT64) {
      auto cast_op = converter->AddOperator<ge::op::Cast>(output_operand);
      cast_op->set_attr_dst_type(ConvertToGEPrecision(dtype));
      SET_INPUT(cast_op, x, arg_operator);
      arg_operator = MAP_OUTPUT(cast_op, y, output_operand);
    }
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported arg operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  if (keepdim) {
    auto unsqueeze_op =
        converter->AddOperator<ge::op::Unsqueeze>(output_operand);
    std::vector<int> axes = {axis};
    unsqueeze_op->set_attr_axes(
        ge::Operator::OpListInt(axes.begin(), axes.end()));
    SET_INPUT(unsqueeze_op, x, arg_operator);
    MAP_OUTPUT(unsqueeze_op, y, output_operand);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
