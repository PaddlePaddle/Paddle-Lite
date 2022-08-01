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

#include "operation/layer_normalization.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertLayerNormalization(Converter* converter,
                              core::Operation* operation) {
  LAYER_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (input_operator == nullptr) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto scale_operator = converter->ConvertOperand(scale_operand);
  auto bias_operator = converter->ConvertOperand(bias_operand);

  // Layer normalization
  auto layer_norm_op =
      converter->AddOperator<ge::op::LayerNorm>(output_operand);
  layer_norm_op->set_attr_epsilon(epsilon);
  layer_norm_op->set_attr_begin_norm_axis(begin_norm_axis);
  layer_norm_op->set_attr_begin_params_axis(begin_norm_axis);
  SET_INPUT(layer_norm_op, x, input_operator);
  SET_INPUT(layer_norm_op, beta, bias_operator);
  SET_INPUT(layer_norm_op, gamma, scale_operator);
  MAP_OUTPUT(layer_norm_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
