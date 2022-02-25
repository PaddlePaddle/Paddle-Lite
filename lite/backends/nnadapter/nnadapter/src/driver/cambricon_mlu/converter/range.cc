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

#include "operation/range.h"
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertRange(Converter* converter, core::Operation* operation) {
  RANGE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto start_tensor = converter->GetMappedTensor(start_operand);
  if (!start_tensor) {
    start_tensor = converter->ConvertOperand(start_operand);
  }
  auto limit_tensor = converter->GetMappedTensor(limit_operand);
  if (!limit_tensor) {
    limit_tensor = converter->ConvertOperand(limit_operand);
  }
  auto delta_tensor = converter->GetMappedTensor(delta_operand);
  if (!delta_tensor) {
    delta_tensor = converter->ConvertOperand(delta_operand);
  }
  auto range_node = converter->network()->AddIRangeNode(
      start_tensor, limit_tensor, delta_tensor);
  NNADAPTER_CHECK(range_node) << "Failed to add range node.";
  auto output_tensor = range_node->GetOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
