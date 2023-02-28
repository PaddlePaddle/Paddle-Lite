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

#include "operation/fill_like.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertFillLike(Converter* converter, core::Operation* operation) {
  FILL_LIKE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (input_tensor == nullptr) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto value_tensor = converter->GetMappedTensor(value_operand);
  if (value_tensor == nullptr) {
    value_tensor = converter->ConvertOperand(value_operand);
  }
  auto input_shape = std::make_shared<default_opset::ShapeOf>(
      *input_tensor, GetElementType<int32_t>());
  auto broadcast_op = std::make_shared<default_opset::Broadcast>(
      *value_tensor, input_shape->output(0));
  MAP_OUTPUT(output_operand, broadcast_op, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
