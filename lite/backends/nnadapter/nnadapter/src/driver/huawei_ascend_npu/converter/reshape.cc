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

#include "operation/reshape.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertReshape(Converter* converter, core::Operation* operation) {
  RESHAPE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  std::shared_ptr<Operator> shape_operator = nullptr;
  if (IsTemporaryShapeOperand(shape_operand)) {
    auto& temporary_shape = *(GetTemporaryShape(shape_operand));
    auto shape_count = temporary_shape.count;
    auto shape_data = temporary_shape.data;
    for (uint32_t i = 0; i < shape_count; i++) {
      if (shape_data[i] == 0) {
        if (input_operand->type.dimensions.data[i] == NNADAPTER_UNKNOWN) {
          shape_data[i] = -1;
        } else {
          shape_data[i] = input_operand->type.dimensions.data[i];
        }
      }
    }
    shape_operator = converter->AddInt32ConstantOperator(
        std::vector<int32_t>(shape_data, shape_data + shape_count));
  } else if (IsConstantOperand(shape_operand)) {
    auto shape_count = shape_operand->length / sizeof(int32_t);
    auto shape_data = reinterpret_cast<int32_t*>(shape_operand->buffer);
    for (uint32_t i = 0; i < shape_count; i++) {
      if (shape_data[i] == 0) {
        if (input_operand->type.dimensions.data[i] == NNADAPTER_UNKNOWN) {
          shape_data[i] = -1;
        } else {
          shape_data[i] = input_operand->type.dimensions.data[i];
        }
      }
    }
    shape_operator = converter->AddInt32ConstantOperator(
        std::vector<int32_t>(shape_data, shape_data + shape_count));
  } else {
    shape_operator = converter->GetMappedOperator(shape_operand);
    if (!shape_operator) {
      shape_operator = converter->ConvertOperand(shape_operand);
    }
  }
  auto reshape_op = converter->AddOperator<ge::op::Reshape>(output_operand);
  SET_INPUT(reshape_op, x, input_operator);
  SET_INPUT(reshape_op, shape, shape_operator);
  MAP_OUTPUT(reshape_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
