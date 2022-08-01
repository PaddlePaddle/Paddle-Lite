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

#include "operation/reduce.h"
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertReduce(Converter* converter, core::Operation* operation) {
  REDUCE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto axes_operator = converter->ConvertOperand(axes_operand);
  switch (operation->type) {
#define CONVERT_REDUCE(type, class_name)                              \
  case NNADAPTER_##type: {                                            \
    auto reduce_op =                                                  \
        converter->AddOperator<hiai::op::class_name>(output_operand); \
    reduce_op->set_attr_keep_dims(keep_dim);                          \
    SET_INPUT(reduce_op, x, input_operator);                          \
    SET_INPUT(reduce_op, axes, axes_operator);                        \
    MAP_OUTPUT(reduce_op, y, output_operand);                         \
  } break;
    CONVERT_REDUCE(REDUCE_MEAN, ReduceMean);
    CONVERT_REDUCE(REDUCE_SUM, ReduceSum);
#undef CONVERT_REDUCE
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported reduce operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
