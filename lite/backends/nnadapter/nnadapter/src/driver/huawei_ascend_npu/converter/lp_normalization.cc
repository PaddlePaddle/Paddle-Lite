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

#include "operation/lp_normalization.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertLpNormalization(Converter* converter, core::Operation* operation) {
  LP_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  ge::Operator::OpListInt axis;
  for (uint32_t i = 0; i < axis_count; i++) {
    axis.push_back(axis_data[i]);
  }
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  NNADAPTER_CHECK_EQ(p, 2) << "HUAWEI_ASCEND_NPU only support p = 2";
  auto l2_norm_op = converter->AddOperator<ge::op::L2Normalize>(output_operand);
  l2_norm_op->set_attr_axis(axis);
  l2_norm_op->set_attr_eps(epsilon);
  SET_INPUT(l2_norm_op, x, input_operator);
  MAP_OUTPUT(l2_norm_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
