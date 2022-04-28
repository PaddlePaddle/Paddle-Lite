// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/split.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertSplit(Converter* converter, core::Operation* operation) {
  SPLIT_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(IsConstantOperand(split_operand));

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto axis_tensor = converter->AddConstantTensor<int32_t>(axis);
  auto split_count = split.size();
  auto sections_tensor = converter->AddConstantTensor<int32_t>(split);
  auto split_op = std::make_shared<default_opset::VariadicSplit>(
      *input_tensor, *axis_tensor, *sections_tensor);
  for (uint32_t i = 0; i < split_count; i++) {
    MAP_OUTPUT(output_operands[i], split_op, i);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
