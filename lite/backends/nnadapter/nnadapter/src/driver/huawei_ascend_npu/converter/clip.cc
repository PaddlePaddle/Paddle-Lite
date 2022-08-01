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

#include "operation/clip.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertClip(Converter* converter, core::Operation* operation) {
  CLIP_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto min_operator = converter->GetMappedOperator(min_operand);
  if (!min_operator) {
    min_operator = converter->ConvertOperand(min_operand);
  }
  auto max_operator = converter->GetMappedOperator(max_operand);
  if (!max_operator) {
    max_operator = converter->ConvertOperand(max_operand);
  }
  auto clip_op = converter->AddOperator<ge::op::ClipByValue>(output_operand);
  SET_INPUT(clip_op, x, input_operator);
  SET_INPUT(clip_op, clip_value_min, min_operator);
  SET_INPUT(clip_op, clip_value_max, max_operator);
  MAP_OUTPUT(clip_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
