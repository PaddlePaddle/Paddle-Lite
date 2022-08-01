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

#include "operation/gather.h"
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertGather(Converter* converter, core::Operation* operation) {
  GATHER_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto indices_tensor = converter->GetMappedTensor(indices_operand);
  if (!indices_tensor) {
    indices_tensor = converter->ConvertOperand(indices_operand);
  }
  auto axis_tensor = converter->GetMappedTensor(axis_operand);
  if (!axis_tensor) {
    axis_tensor = converter->ConvertOperand(axis_operand);
  }
  auto gather_node = converter->network()->AddIGatherNode(
      input_tensor, indices_tensor, axis_tensor);
  NNADAPTER_CHECK(gather_node) << "Failed to add gather node.";
  auto output_tensor = gather_node->GetOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
