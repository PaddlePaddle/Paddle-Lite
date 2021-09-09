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

#include "core/operation/adaptive_pool2d.h"
#include "driver/huawei_ascend_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int Program::ConvertAdaptivePool2D(hal::Operation* operation) {
  ADAPTIVE_POOL_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = ConvertOperand(input_operand);
  }
  auto pool2d_name = GetOperatorName(output_operand);
  if (operation->type == NNADAPTER_ADAPTIVE_MAX_POOL_2D) {
    auto pool2d_op = std::make_shared<ge::op::AdaptiveMaxPool2d>(pool2d_name);
    pool2d_op->set_attr_output_size(
        ge::Operator::OpListInt({kernel_height, kernel_width}));
    SET_INPUT(pool2d_op, x, input_operator);
    auto adaptive_pool2d_op = MAP_OUTPUT(pool2d_op, y, output_operand);
    auto out_shape = ge::Shape();
    auto format = ge::FORMAT_NCHW;
    auto dtype = ge::DT_INT32;
    auto tensor_desc =
        std::make_shared<ge::TensorDesc>(out_shape, format, dtype);
    pool2d_op->update_output_desc_argmax(*tensor_desc);
    auto argmax_op =
        std::make_shared<Operator>(pool2d_op, tensor_desc, "argmax", -1);
    // Cast op
    auto cast_op = std::make_shared<ge::op::Cast>("dummy_cast");
    cast_op->set_attr_dst_type(ge::DT_FLOAT);
    SET_INPUT(cast_op, x, argmax_op);
    auto cast_operator = MAP_OUTPUT(cast_op, y, output_operand);
    // Sub op
    auto sub_op = std::make_shared<ge::op::Sub>("dummy_sub");
    SET_INPUT(sub_op, x1, cast_operator);
    SET_INPUT(sub_op, x2, cast_operator);
    auto sub_operator = MAP_OUTPUT(sub_op, y, output_operand);
    // Add op
    auto add_op = std::make_shared<ge::op::Add>("dummy_add");
    SET_INPUT(add_op, x1, adaptive_pool2d_op);
    SET_INPUT(add_op, x2, sub_operator);
    auto add_operator = MAP_OUTPUT(add_op, y, output_operand);
  } else {
    auto pool2d_op = std::make_shared<ge::op::AdaptiveAvgPool2d>(pool2d_name);
    pool2d_op->set_attr_output_size(
        ge::Operator::OpListInt({kernel_height, kernel_width}));
    SET_INPUT(pool2d_op, x, input_operator);
    MAP_OUTPUT(pool2d_op, y, output_operand);
  }

  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
