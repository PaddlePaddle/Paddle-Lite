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
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertPad(Converter* converter, core::Operation* operation) {
  PAD_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  std::string pad_mode = ConvertPadModeCodeToGEPadMode(mode);
  NNADAPTER_CHECK_NE(pad_mode, "edge") << "Not support mode=edge";
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto pads_cnt = pads_operand->length / sizeof(int32_t);
  std::vector<int32_t> pads_shape = {4, 2};
  auto pads_data = reinterpret_cast<int32_t*>(pads_operand->buffer);
  auto pads_operator = converter->AddInt32ConstantOperator(
      std::vector<int32_t>(pads_data, pads_data + pads_cnt), pads_shape);
  auto value_operator = converter->GetMappedOperator(value_operand);
  if (!value_operator) {
    value_operator = converter->ConvertOperand(value_operand);
  }
  if (pad_mode == "constant") {
    auto pad_op = converter->AddOperator<hiai::op::PadV2>(output_operand);
    SET_INPUT(pad_op, x, input_operator);
    SET_INPUT(pad_op, paddings, pads_operator);
    SET_INPUT(pad_op, constant_values, value_operator);
    MAP_OUTPUT(pad_op, y, output_operand);
  } else if (pad_mode == "reflect") {
    auto pad_op = converter->AddOperator<hiai::op::MirrorPad>(output_operand);
    pad_op->set_attr_mode("REFLECT");
    SET_INPUT(pad_op, x, input_operator);
    SET_INPUT(pad_op, paddings, pads_operator);
    MAP_OUTPUT(pad_op, y, output_operand);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupport pad mode: " << pad_mode;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
