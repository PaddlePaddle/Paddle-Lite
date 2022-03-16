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

#include "operation/fully_connected.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertFullyConnected(Converter* converter, core::Operation* operation) {
  FULLY_CONNECTED_OPERATION_EXTRACT_INPUTS_OUTPUTS
  int64_t input_production = 1;
  for (uint32_t i = 0; i < input_operand->type.dimensions.count; i++) {
    auto dimension = input_operand->type.dimensions.data[i];
    if (dimension < 0) {
      input_production = -1;
      break;
    }
    input_production *= dimension;
  }
  int64_t batch_size;
  if (input_production < 0) {
    batch_size = -1;
  } else {
    batch_size = input_production / input_size;
  }
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

  std::shared_ptr<Operator> output_operator;
  auto input_precision = input_operand->type.precision;
  if (input_precision == NNADAPTER_FLOAT32) {
    // Use MatMul instead of FullyConnection to avoid outputing the 4-D tensor
    auto matmul_op = converter->AddOperator<ge::op::MatMul>(output_operand);
    matmul_op->set_attr_transpose_x1(false);
    // {num_units, input_size} -> {input_size, num_units}
    matmul_op->set_attr_transpose_x2(true);
    SET_INPUT(matmul_op, x1, input_operator);
    SET_INPUT(matmul_op, x2, weight_operator);
    SET_INPUT(matmul_op, bias, bias_operator);
    output_operator = MAP_OUTPUT(matmul_op, y, output_operand);
  } else if (input_precision == NNADAPTER_QUANT_INT8_SYMM_PER_LAYER) {
    // Only FullyConnection support int8
    auto fc_op =
        converter->AddOperator<ge::op::FullyConnection>(output_operand);
    fc_op->set_attr_transpose(false);
    fc_op->set_attr_num_output(1000);
    fc_op->set_attr_axis(1);
    SET_INPUT(fc_op, x, input_operator);
    SET_INPUT(fc_op, w, weight_operator);
    SET_INPUT(fc_op, b, bias_operator);
    auto fc_output_operator = MAP_OUTPUT(fc_op, y, output_operand);
    // reshape to 2D
    auto reshape_op = converter->AddOperator<ge::op::Reshape>(output_operand);
    auto shape_data = output_operand->type.dimensions.data;
    auto shape_operator = converter->AddInt32ConstantOperator(
        std::vector<int32_t>{static_cast<int32_t>(shape_data[0]),
                             static_cast<int32_t>(shape_data[1])});
    SET_INPUT(reshape_op, x, fc_output_operator);
    SET_INPUT(reshape_op, shape, shape_operator);
    output_operator = MAP_OUTPUT(reshape_op, y, output_operand);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported precision.";
  }

  // fuse activations
  switch (fuse_code) {
#define CONVERT_UNARY_ACTIVATION(type, class_name)                            \
  case NNADAPTER_FUSED_##type: {                                              \
    auto act_op = converter->AddOperator<ge::op::class_name>(output_operand); \
    SET_INPUT(act_op, x, output_operator);                                    \
    MAP_OUTPUT(act_op, y, output_operand);                                    \
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
