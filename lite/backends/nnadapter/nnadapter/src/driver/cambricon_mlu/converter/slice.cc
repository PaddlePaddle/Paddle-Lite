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

#include "operation/slice.h"
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertSlice(Converter* converter, core::Operation* operation) {
  SLICE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (input_tensor == nullptr) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto axes_tensor = converter->GetMappedTensor(axes_operand);
  if (axes_tensor == nullptr) {
    axes_tensor = converter->ConvertOperand(axes_operand);
  }
  auto starts_tensor = converter->GetMappedTensor(starts_operand);
  if (starts_tensor == nullptr) {
    starts_tensor = converter->ConvertOperand(starts_operand);
  }
  auto ends_tensor = converter->GetMappedTensor(ends_operand);
  if (ends_tensor == nullptr) {
    ends_tensor = converter->ConvertOperand(ends_operand);
  }
  auto steps_tensor = converter->GetMappedTensor(steps_operand);
  if (steps_tensor == nullptr) {
    steps_tensor = converter->ConvertOperand(steps_operand);
  }
  auto strided_slice_node = converter->network()->AddIStridedSliceNode(
      input_tensor, starts_tensor, ends_tensor, steps_tensor, axes_tensor);
  NNADAPTER_CHECK(strided_slice_node) << "Failed to add strided_slice node.";
  auto output_tensor = strided_slice_node->GetOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);

  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
