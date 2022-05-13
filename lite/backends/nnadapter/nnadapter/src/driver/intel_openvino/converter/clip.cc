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

#include "operation/clip.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertClip(Converter* converter, core::Operation* operation) {
  CLIP_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  NNADAPTER_CHECK(IsConstantOperand(min_operand))
      << "'min' should be a constant operand!";
  NNADAPTER_CHECK(IsConstantOperand(max_operand))
      << "'max' should be a constant operand!";
  auto min_value = *reinterpret_cast<float*>(min_operand->buffer);
  auto max_value = *reinterpret_cast<float*>(max_operand->buffer);
  auto clip_op = std::make_shared<default_opset::Clamp>(
      *input_tensor, min_value, max_value);
  MAP_OUTPUT(output_operand, clip_op, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
