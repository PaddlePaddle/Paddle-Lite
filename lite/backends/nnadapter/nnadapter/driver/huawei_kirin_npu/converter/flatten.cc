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

#include "core/operation/flatten.h"
#include "driver/huawei_kirin_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int Program::ConvertFlatten(hal::Operation* operation) {
  FLATTEN_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = ConvertOperand(input_operand);
  }
  auto reshape_name = GetOperatorName(output_operand);
  auto reshape_op = std::make_shared<hiai::op::Reshape>(reshape_name);
  std::vector<int32_t> shape;
  for (uint32_t i = 0; i < output_operand->type.dimension_count; i++) {
    shape.push_back(output_operand->type.dimensions[i]);
  }
  auto shape_operator = AddInt32ConstantOperator(shape);
  SET_INPUT(reshape_op, x, input_operator);
  SET_INPUT(reshape_op, shape, shape_operator);
  MAP_OUTPUT(reshape_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
