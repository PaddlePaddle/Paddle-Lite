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
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertLpNormalization(Converter* converter, core::Operation* operation) {
  LP_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK_EQ(p, 2) << "L1 norm is not supported.";
  NNADAPTER_CHECK_GE(epsilon, 1e-4f) << "L2 norm only support epsilon >= 1e-4";
  if (axis_count > 1 && axis_data[0] != 1) {
    NNADAPTER_LOG(FATAL) << "L2 norm only support normalize the 1th dimension.";
  }

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto l2_norm_op =
      converter->AddOperator<hiai::op::L2Normalize>(output_operand);
  l2_norm_op->set_attr_eps(epsilon);
  SET_INPUT(l2_norm_op, x, input_operator);
  MAP_OUTPUT(l2_norm_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
