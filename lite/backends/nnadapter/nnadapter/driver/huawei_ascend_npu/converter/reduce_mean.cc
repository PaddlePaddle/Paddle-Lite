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

#include "core/operation/reduce_mean.h"
#include "driver/huawei_ascend_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int Program::ConvertReduceMean(hal::Operation* operation) {
  REDUCE_MEAN_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = ConvertOperand(input_operand);
  }
  auto axes_operator = ConvertOperand(axes_operand);
  auto reduce_mean_name = GetOperatorName(output_operand);
  auto reduce_mean_op = std::make_shared<ge::op::ReduceMean>(reduce_mean_name);
  reduce_mean_op->set_attr_keep_dims(keep_dim);
  SET_INPUT(reduce_mean_op, x, input_operator);
  SET_INPUT(reduce_mean_op, axes, axes_operator);
  MAP_OUTPUT(reduce_mean_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
