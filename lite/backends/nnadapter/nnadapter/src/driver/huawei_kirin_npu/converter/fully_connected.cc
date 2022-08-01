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

#include "operation/fully_connected.h"
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertFullyConnected(Converter* converter, core::Operation* operation) {
  FULLY_CONNECTED_OPERATION_EXTRACT_INPUTS_OUTPUTS
  auto batch_size =
      ProductionOfDimensions(input_operand->type.dimensions.data,
                             input_operand->type.dimensions.count) /
      input_size;
  NNADAPTER_VLOG(5) << "batch_size: " << batch_size;

  // Convert to GE operators
  // Add input operator and reshape it to (batch_size, input_size, 1, 1)
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  // Reshape the input operator to 2-D tensor {batch_size, input_size} if the
  // dimensions_count not equal 2
  if (input_operand->type.dimensions.count != 2) {
    auto reshape_op = converter->AddOperator<hiai::op::Reshape>(input_operand);
    auto shape_operator = converter->AddInt32ConstantOperator(
        {static_cast<int32_t>(batch_size), input_size});
    SET_INPUT(reshape_op, x, input_operator);
    SET_INPUT(reshape_op, shape, shape_operator);
    input_operator = MAP_OUTPUT(reshape_op, y, input_operand);
  }
  auto weight_operator = converter->ConvertOperand(weight_operand);
  auto bias_operator = converter->ConvertOperand(bias_operand);
  // Use MatMul instead of FullyConnection to avoid outputing the 4-D tensor
  auto matmul_op = converter->AddOperator<hiai::op::MatMul>(output_operand);
  matmul_op->set_attr_transpose_x1(false);
  matmul_op->set_attr_transpose_x2(
      true);  // {num_units, input_size} -> {input_size, num_units}
  SET_INPUT(matmul_op, x1, input_operator);
  SET_INPUT(matmul_op, x2, weight_operator);
  // SET_INPUT(matmul_op, bias, bias_operator);
  auto matmul_operator = MAP_OUTPUT(matmul_op, y, output_operand);
  // Add a Add operator to support bias(HiAI GE MatMul doesn't support bias)
  auto add_op = converter->AddOperator<hiai::op::Add>(output_operand);
  SET_INPUT(add_op, x1, matmul_operator);
  SET_INPUT(add_op, x2, bias_operator);
  MAP_OUTPUT(add_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
