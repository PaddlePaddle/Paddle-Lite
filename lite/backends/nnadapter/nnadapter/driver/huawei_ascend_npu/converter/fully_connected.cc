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

#include "core/operation/fully_connected.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertFullyConnected(Converter* converter, hal::Operation* operation) {
  FULLY_CONNECTED_OPERATION_EXTRACT_INPUTS_OUTPUTS
  auto batch_size =
      ProductionOfDimensions(input_operand->type.dimensions.data,
                             input_operand->type.dimensions.count) /
      input_size;
  NNADAPTER_VLOG(5) << "batch_size: " << batch_size;

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  // Reshape the input operator to 2-D tensor {batch_size, input_size} if the
  // dimensions_count not equal 2
  if (input_operand->type.dimensions.count != 2) {
    auto reshape_op =
        converter->AddOperator<ge::op::Reshape>(input_operand, "reshape");
    auto shape_operator = converter->AddInt32ConstantOperator(
        std::vector<int32_t>({static_cast<int32_t>(batch_size), input_size}));
    SET_INPUT(reshape_op, x, input_operator);
    SET_INPUT(reshape_op, shape, shape_operator);
    input_operator = MAP_OUTPUT(reshape_op, y, output_operand);
  }
  auto weight_operator = converter->ConvertOperand(weight_operand);
  auto bias_operator = converter->ConvertOperand(bias_operand);
  // Use MatMul instead of FullyConnection to avoid outputing the 4-D tensor
  auto matmul_op = converter->AddOperator<ge::op::MatMul>(output_operand);
  matmul_op->set_attr_transpose_x1(false);
  matmul_op->set_attr_transpose_x2(
      true);  // {num_units, input_size} -> {input_size, num_units}
  SET_INPUT(matmul_op, x1, input_operator);
  SET_INPUT(matmul_op, x2, weight_operator);
  SET_INPUT(matmul_op, bias, bias_operator);
  MAP_OUTPUT(matmul_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
