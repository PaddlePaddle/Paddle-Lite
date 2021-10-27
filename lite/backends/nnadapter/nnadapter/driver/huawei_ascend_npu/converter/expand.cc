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
  uint32_t shape_count;
  int32_t* shape_data;
  auto& shape_type = shape_operand->type;
  if (IsConstantOperand(shape_operand)) {
    shape_count = shape_operand->length / sizeof(int32_t);
    shape_data = reinterpret_cast<int32_t*>(shape_operand->buffer);
  } else if (shape_type.lifetime == NNADAPTER_TEMPORARY_SHAPE) {
    auto shape_operand_dimension =
        *reinterpret_cast<NNAdapterOperandDimensionType*>(
            shape_operand->buffer);
    shape_count = shape_operand_dimension.count;
    shape_data = shape_operand_dimension.data;
  } else {
    NNADAPTER_LOG(ERROR) << "Unsupported shape lifetime: "
                         << static_cast<int32_t>(shape_type.lifetime);
    return NNADAPTER_INVALID_PARAMETER;
  }

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto expand_op = converter->AddOperator<ge::op::ExpandD>(output_operand);
  std::vector<int64_t> expand_shape(shape_count);
  for (uint32_t i = 0; i < shape_count; i++) {
    expand_shape[i] = shape_data[i];
  }
  expand_op->set_attr_shape(ge::Operator::OpListInt(expand_shape));
  SET_INPUT(expand_op, x, input_operator);
  MAP_OUTPUT(expand_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
