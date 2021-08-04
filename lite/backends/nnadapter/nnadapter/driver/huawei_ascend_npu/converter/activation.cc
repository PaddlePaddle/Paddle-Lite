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

namespace nnadapter {
namespace huawei_ascend_npu {

int Program::ConvertActivation(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 1);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = ConvertOperand(input_operand);
  }
  auto act_name = GetOperatorName(output_operand);
  switch (operation->type) {
#define CONVERT_UNARY_ACTIVATION(type, class_name)                \
  case NNADAPTER_##type: {                                        \
    auto act_op = std::make_shared<ge::op::class_name>(act_name); \
    SET_INPUT(act_op, x, input_operator);                         \
    MAP_OUTPUT(act_op, y, output_operand);                        \
  } break;
    CONVERT_UNARY_ACTIVATION(SIGMOID, Sigmoid);
    CONVERT_UNARY_ACTIVATION(RELU, Relu);
    CONVERT_UNARY_ACTIVATION(RELU6, Relu6);
    CONVERT_UNARY_ACTIVATION(TANH, Tanh);
    CONVERT_UNARY_ACTIVATION(ABS, Abs);

    case NNADAPTER_HARD_SWISH: {
      // Get operands and prepare param
      auto offset_operand = input_operands[1];
      auto threshold_operand = input_operands[2];
      float offset = *reinterpret_cast<float*>(offset_operand->buffer);
      float threshold = *reinterpret_cast<float*>(threshold_operand->buffer);
      float alpha = 1.0 / threshold;
      float beta = offset / threshold;
      float clip_min = 0.0;
      float clip_max = 1.0;
      int64_t data_cnt = 1;
      // Tensor info
      auto dimension_count = input_operand->type.dimension_count;
      std::vector<int> input_dim = {};
      for (int i = 0; i < dimension_count; i++) {
        input_dim.push_back(input_operand->type.dimensions[i]);
        data_cnt *= input_operand->type.dimensions[i];
      }
      // Prepare data
      std::vector<float> alpha_vec(data_cnt, alpha);
      std::vector<float> beta_vec(data_cnt, beta);
      std::vector<float> clipmin_vec(data_cnt, clip_min);
      std::vector<float> clipmax_vec(data_cnt, clip_max);
      // Scale op
      auto scale_op = std::make_shared<ge::op::Scale>("scale");
      auto alpha_operator = AddFloat32ConstantOperator(alpha_vec, input_dim);
      auto beta_operator = AddFloat32ConstantOperator(beta_vec, input_dim);
      scale_op->set_attr_axis(0);
      scale_op->set_attr_num_axes(-1);
      scale_op->set_attr_scale_from_blob(true);
      SET_INPUT(scale_op, x, input_operator);
      SET_INPUT(scale_op, scale, alpha_operator);
      SET_INPUT(scale_op, bias, beta_operator);
      std::shared_ptr<Operator> scale_operator =
          MAP_OUTPUT(scale_op, y, output_operand);
      // Clip op
      auto clip_op = std::make_shared<ge::op::ClipByValue>("clip");
      auto clip_min_operator =
          AddFloat32ConstantOperator(clipmin_vec, input_dim);
      auto clip_max_operator =
          AddFloat32ConstantOperator(clipmax_vec, input_dim);
      SET_INPUT(clip_op, x, scale_operator);
      SET_INPUT(clip_op, clip_value_min, clip_min_operator);
      SET_INPUT(clip_op, clip_value_max, clip_max_operator);
      std::shared_ptr<Operator> clip_operator =
          MAP_OUTPUT(clip_op, y, output_operand);
      // Mul op
      auto mul_op = std::make_shared<ge::op::Mul>("mul");
      SET_INPUT(mul_op, x1, input_operator);
      SET_INPUT(mul_op, x2, clip_operator);
      MAP_OUTPUT(mul_op, y, output_operand);

      return NNADAPTER_NO_ERROR;
    }
#undef CONVERT_UNARY_ACTIVATION
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported activation operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
