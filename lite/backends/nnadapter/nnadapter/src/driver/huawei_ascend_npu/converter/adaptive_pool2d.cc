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

#include "operation/adaptive_pool2d.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertAdaptivePool2D(Converter* converter, core::Operation* operation) {
  ADAPTIVE_POOL_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  if (operation->type == NNADAPTER_ADAPTIVE_MAX_POOL_2D) {
#if NNADAPTER_HUAWEI_ASCEND_NPU_CANN_VERSION_LESS_THAN(5, 1, 1)
    auto pool2d_op =
        converter->AddOperator<ge::op::AdaptiveMaxPool2d>(output_operand);
    pool2d_op->set_attr_output_size(
        ge::Operator::OpListInt({output_height, output_width}));
    SET_INPUT(pool2d_op, x, input_operator);
    auto adaptive_pool2d_operator = MAP_OUTPUT(pool2d_op, y, output_operand);
    auto tensor_desc = std::make_shared<ge::TensorDesc>(
        ge::Shape(), ge::FORMAT_NCHW, ge::DT_INT32);
    pool2d_op->update_output_desc_argmax(*tensor_desc);
    auto argmax_op =
        std::make_shared<Operator>(pool2d_op, tensor_desc, "argmax", -1);
    // Cast op
    auto dummy_cast_op =
        converter->AddOperator<ge::op::Cast>(output_operand, "dummy_cast");
    dummy_cast_op->set_attr_dst_type(ge::DT_FLOAT);
    SET_INPUT(dummy_cast_op, x, argmax_op);
    auto dummy_cast_operator = MAP_OUTPUT(dummy_cast_op, y, output_operand);
    // Sub op
    auto dummy_sub_op =
        converter->AddOperator<ge::op::Sub>(output_operand, "dummy_sub");
    SET_INPUT(dummy_sub_op, x1, dummy_cast_operator);
    SET_INPUT(dummy_sub_op, x2, dummy_cast_operator);
    auto dummy_sub_operator = MAP_OUTPUT(dummy_sub_op, y, output_operand);
    // Add op
    auto dummy_add_op =
        converter->AddOperator<ge::op::Add>(output_operand, "dummy_add");
    SET_INPUT(dummy_add_op, x1, adaptive_pool2d_operator);
    SET_INPUT(dummy_add_op, x2, dummy_sub_operator);
    MAP_OUTPUT(dummy_add_op, y, output_operand);
#else
    NNADAPTER_LOG(FATAL) << "AdaptiveMaxPool2d has bugs when CANN >= 5.1.rc1, "
                            "it will be fixed later";
#endif
  } else {
    auto pool2d_op =
        converter->AddOperator<ge::op::AdaptiveAvgPool2d>(output_operand);
    pool2d_op->set_attr_output_size(
        ge::Operator::OpListInt({output_height, output_width}));
    SET_INPUT(pool2d_op, x, input_operator);
    MAP_OUTPUT(pool2d_op, y, output_operand);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
