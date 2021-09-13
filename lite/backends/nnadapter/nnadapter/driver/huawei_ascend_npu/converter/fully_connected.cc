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

#include "driver/huawei_ascend_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int Program::ConvertFullyConnected(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 4);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  // Weight
  auto weight_operand = input_operands[1];
  NNADAPTER_VLOG(5) << "weight: " << OperandToString(weight_operand);
  NNADAPTER_CHECK_EQ(weight_operand->type.dimension_count, 2);
  auto num_units = weight_operand->type.dimensions[0];
  auto input_size = weight_operand->type.dimensions[1];
  auto batch_size =
      ProductionOfDimensions(input_operand->type.dimensions,
                             input_operand->type.dimension_count) /
      input_size;
  // Bias
  auto bias_operand = input_operands[2];
  NNADAPTER_VLOG(5) << "bias: " << OperandToString(bias_operand);
  NNADAPTER_CHECK_EQ(bias_operand->type.dimension_count, 1);
  NNADAPTER_CHECK_EQ(num_units, bias_operand->type.dimensions[0]);
  // Fuse code
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[3]->buffer);
  NNADAPTER_VLOG(5) << "fuse_code=" << fuse_code;
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = ConvertOperand(input_operand);
  }
  // Reshape the input operator to 2-D tensor {batch_size, input_size} if the
  // dimension_count not equal 2
  if (input_operand->type.dimension_count != 2) {
    auto reshape_name = GetOperatorName(input_operand) + "/reshape";
    auto reshape_op = std::make_shared<ge::op::Reshape>(reshape_name);
    auto shape_operator = AddInt32ConstantOperator(
        std::vector<int32_t>({static_cast<int32_t>(batch_size), input_size}));
    SET_INPUT(reshape_op, x, input_operator);
    SET_INPUT(reshape_op, shape, shape_operator);
    input_operator = MAP_OUTPUT(reshape_op, y, output_operand);
  }
  auto weight_operator = ConvertOperand(weight_operand);
  auto bias_operator = ConvertOperand(bias_operand);
  // Use MatMul instead of FullyConnection to avoid outputing the 4-D tensor
  auto matmul_name = GetOperatorName(output_operand) + "/matmul";
  auto matmul_op = std::make_shared<ge::op::MatMul>(matmul_name);
  matmul_op->set_attr_transpose_x1(false);
  matmul_op->set_attr_transpose_x2(
      true);  // {num_units, input_size} -> {input_size, num_units}
  SET_INPUT(matmul_op, x1, input_operator);
  SET_INPUT(matmul_op, x2, weight_operator);
  SET_INPUT(matmul_op, bias, bias_operator);
  auto matmul_operator = MAP_OUTPUT(matmul_op, y, output_operand);

  // fuse activations
  auto act_name = GetOperatorName(output_operand) + "/activation";
  switch (fuse_code) {
#define CONVERT_UNARY_ACTIVATION(type, class_name)                \
  case NNADAPTER_FUSED_##type: {                                  \
    auto act_op = std::make_shared<ge::op::class_name>(act_name); \
    SET_INPUT(act_op, x, matmul_operator);                        \
    MAP_OUTPUT(act_op, y, output_operand);                        \
  } break;
    CONVERT_UNARY_ACTIVATION(RELU, Relu);
    CONVERT_UNARY_ACTIVATION(RELU6, Relu6);
// TODO(lsy): support relu1.
#undef CONVERT_UNARY_ACTIVATION
    case NNADAPTER_FUSED_NONE:
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported fuse_code(" << fuse_code
                           << ") is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
