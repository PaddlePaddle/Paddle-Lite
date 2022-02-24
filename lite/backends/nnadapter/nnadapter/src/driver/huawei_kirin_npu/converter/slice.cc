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

#include "operation/slice.h"
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertSlice(Converter* converter, core::Operation* operation) {
  SLICE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (input_operator == nullptr) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto axes_operator = converter->GetMappedOperator(axes_operand);
  if (axes_operator == nullptr) {
    axes_operator = converter->ConvertOperand(axes_operand);
  }
  auto starts_operator = converter->GetMappedOperator(starts_operand);
  if (starts_operator == nullptr) {
    starts_operator = converter->ConvertOperand(starts_operand);
  }
  auto ends_operator = converter->GetMappedOperator(ends_operand);
  if (ends_operator == nullptr) {
    ends_operator = converter->ConvertOperand(ends_operand);
  }
  auto steps_operator = converter->GetMappedOperator(steps_operand);
  if (steps_operator == nullptr) {
    steps_operator = converter->ConvertOperand(steps_operand);
  }
  auto slice_op =
      converter->AddOperator<hiai::op::StridedSliceV2>(output_operand);
  SET_INPUT(slice_op, x, input_operator);
  SET_INPUT(slice_op, begin, starts_operator);
  SET_INPUT(slice_op, end, ends_operator);
  SET_INPUT(slice_op, axes, axes_operator);
  SET_INPUT(slice_op, strides, steps_operator);
  MAP_OUTPUT(slice_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
