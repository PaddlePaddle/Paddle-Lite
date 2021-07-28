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

#include "driver/huawei_ascend_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int Program::ConvertResizeNearest(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 4);
  NNADAPTER_CHECK_EQ(output_count, 1);

  // input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  // shape
  auto shape_operand = input_operands[1];
  if (shape_operand != nullptr) {
    NNADAPTER_VLOG(5) << "shape: " << OperandToString(shape_operand);
  }
  // scales
  auto scales_operand = input_operands[2];
  if (scales_operand != nullptr) {
    NNADAPTER_VLOG(5) << "scales: " << OperandToString(scales_operand);
  }
  // align_corners
  auto align_corners_operand = input_operands[3];
  NNADAPTER_VLOG(5) << "align_corners: "
                    << OperandToString(align_corners_operand);
  bool align_corners =
      reinterpret_cast<bool*>(align_corners_operand->buffer)[0];

  // output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to GE operators
  auto resize_nearest_name = GetOperatorName(output_operand);
  auto resize_nearest_op =
      std::make_shared<ge::op::ResizeNearestNeighborV2>(resize_nearest_name);
  auto input_operator = GetMappedOperator(input_operand);
  if (input_operator == nullptr) {
    input_operator = ConvertOperand(input_operand);
  }
  SET_INPUT(resize_nearest_op, x, input_operator);
  if (shape_operand != nullptr) {
    auto shape_operator = GetMappedOperator(shape_operand);
    if (shape_operator == nullptr) {
      shape_operator = ConvertOperand(shape_operand);
    }
    SET_INPUT(resize_nearest_op, size, shape_operator);
  } else if (scales_operand != nullptr) {
    NNADAPTER_LOG(WARNING) << "Not support scales now.";
    return NNADAPTER_INVALID_PARAMETER;
  } else {
    NNADAPTER_LOG(WARNING) << "Either shape_operand or scales_operand should "
                              "be set.";
    return NNADAPTER_INVALID_PARAMETER;
  }
  resize_nearest_op->set_attr_align_corners(align_corners);
  MAP_OUTPUT(resize_nearest_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
