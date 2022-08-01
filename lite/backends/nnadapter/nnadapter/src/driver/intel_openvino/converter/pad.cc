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
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertPad(Converter* converter, core::Operation* operation) {
  PAD_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto paddings_tensor = converter->GetMappedTensor(pads_operand);
  if (!paddings_tensor) {
    paddings_tensor = converter->ConvertOperand(pads_operand);
  }
  auto pad_mode = ConvertPadModeCodeToOVPadMode(mode);
  float value = *reinterpret_cast<float*>(value_operand->buffer);
  auto value_tensor = converter->AddConstantTensor<float>(value);

  // Set padding.
  std::vector<int> pads_begin;
  std::vector<int> pads_end;
  uint32_t pads_size =
      pads_operand->length / static_cast<uint32_t>(sizeof(int32_t));
  auto pads_buffer = reinterpret_cast<int32_t*>(pads_operand->buffer);
  for (uint32_t i = 0; i < pads_size; i = i + 2) {
    pads_begin.emplace_back(pads_buffer[i]);
    pads_end.emplace_back(pads_buffer[i + 1]);
  }
  auto padding_begin = converter->AddConstantTensor<int32_t>(pads_begin);
  auto padding_end = converter->AddConstantTensor<int32_t>(pads_end);
  std::shared_ptr<Operator> pad_op;
  if (pad_mode == PadMode::CONSTANT) {
    pad_op = std::make_shared<default_opset::Pad>(
        *input_tensor, *padding_begin, *padding_end, *value_tensor, pad_mode);
  } else {
    pad_op = std::make_shared<default_opset::Pad>(
        *input_tensor, *padding_begin, *padding_end, pad_mode);
  }
  MAP_OUTPUT(output_operand, pad_op, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
