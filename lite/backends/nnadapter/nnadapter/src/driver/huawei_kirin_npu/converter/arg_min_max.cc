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
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertArgMinMax(Converter* converter, core::Operation* operation) {
  ARG_MIN_MAX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto axes_operator = converter->AddInt32ConstantOperator({axis});
  if (operation->type == NNADAPTER_ARG_MAX) {
    auto arg_max_op =
        converter->AddOperator<hiai::op::ArgMaxExt2>(output_operand);
    arg_max_op->set_attr_keep_dims(keepdim);
    SET_INPUT(arg_max_op, x, input_operator);
    SET_INPUT(arg_max_op, axis, axes_operator);
    MAP_OUTPUT(arg_max_op, y, output_operand);
  } else if (operation->type == NNADAPTER_ARG_MIN) {
    auto arg_max_op =
        converter->AddOperator<hiai::op::ReduceMin>(output_operand);
    arg_max_op->set_attr_keep_dims(keepdim);
    SET_INPUT(arg_max_op, x, input_operator);
    SET_INPUT(arg_max_op, axes, axes_operator);
    MAP_OUTPUT(arg_max_op, y, output_operand);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported arg operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
