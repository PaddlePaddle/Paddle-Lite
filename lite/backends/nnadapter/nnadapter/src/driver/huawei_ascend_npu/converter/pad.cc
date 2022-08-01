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

#include "operation/pad.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertPad(Converter* converter, core::Operation* operation) {
  PAD_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  std::string pad_mode = ConvertPadModeCodeToGEPadMode(mode);
  int32_t value =
      static_cast<int32_t>(*reinterpret_cast<float*>(value_operand->buffer));
#if NNADAPTER_HUAWEI_ASCEND_NPU_CANN_VERSION_LESS_THAN(5, 0, 3)
  NNADAPTER_CHECK_EQ(pad_mode, "constant")
      << "Only support mode=constant right now, "
         "but received mode is "
      << pad_mode;
  NNADAPTER_CHECK_EQ(value, 0) << "Only support constant_values=0 right now, "
                                  "but received constant_value is "
                               << value;
#endif
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  if ((mode == NNADAPTER_PAD_MODE_REFLECT || mode == NNADAPTER_PAD_MODE_EDGE) &&
      input_operand->type.dimensions.count == 5 &&
      IsConstantOperand(pads_operand) &&
      !IsOperandWithDynamicShape(input_operand)) {
    uint32_t pads_size =
        pads_operand->length / static_cast<uint32_t>(sizeof(int32_t));
    auto pads_buffer = reinterpret_cast<int32_t*>(pads_operand->buffer);
    if (pads_size == 10 && pads_buffer[6] == 0 && pads_buffer[7] == 0 &&
        pads_buffer[8] == 0 && pads_buffer[9] == 0) {
      // Reshape to 4-dimensions
      std::vector<int32_t> shape_data(input_operand->type.dimensions.data,
                                      input_operand->type.dimensions.data + 3);
      shape_data.push_back(input_operand->type.dimensions.data[3] *
                           input_operand->type.dimensions.data[4]);
      auto start_shape_operator =
          converter->AddInt32ConstantOperator(shape_data);
      auto start_reshape_op =
          converter->AddOperator<ge::op::Reshape>(output_operand);
      SET_INPUT(start_reshape_op, x, input_operator);
      SET_INPUT(start_reshape_op, shape, start_shape_operator);
      auto reshape_output_operator =
          MAP_OUTPUT(start_reshape_op, y, output_operand);
      // Use pad2d op to compute
      std::vector<int32_t> pad_data(pads_buffer, pads_buffer + 8);
      auto pads_operator = converter->AddInt32ConstantOperator(pad_data);
      auto pad_op = converter->AddOperator<ge::op::PadV3>(output_operand);
      pad_op->set_attr_mode(pad_mode);
      SET_INPUT(pad_op, x, reshape_output_operator);
      SET_INPUT(pad_op, paddings, pads_operator);
      auto pad2d_output_operator = MAP_OUTPUT(pad_op, y, output_operand);
      // Reshape to 5-dimensions
      auto end_shape_operator = converter->AddInt32ConstantOperator(
          std::vector<int32_t>(output_operand->type.dimensions.data,
                               output_operand->type.dimensions.data +
                                   output_operand->type.dimensions.count));
      auto end_reshape_op =
          converter->AddOperator<ge::op::Reshape>(output_operand);
      SET_INPUT(end_reshape_op, x, pad2d_output_operator);
      SET_INPUT(end_reshape_op, shape, end_shape_operator);
      MAP_OUTPUT(end_reshape_op, y, output_operand);
      return NNADAPTER_NO_ERROR;
    }
  }
  auto pads_operator = converter->GetMappedOperator(pads_operand);
  if (!pads_operator) {
    pads_operator = converter->ConvertOperand(pads_operand);
  }
  auto value_operator = converter->AddInt32ConstantOperator(value);

  auto pad_op = converter->AddOperator<ge::op::PadV3>(output_operand);
  pad_op->set_attr_mode(pad_mode);
  SET_INPUT(pad_op, x, input_operator);
  SET_INPUT(pad_op, paddings, pads_operator);
  SET_INPUT(pad_op, constant_values, value_operator);
  MAP_OUTPUT(pad_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
