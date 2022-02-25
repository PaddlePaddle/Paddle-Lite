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

#include "operation/lp_normalization.h"
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertLpNormalization(Converter* converter, core::Operation* operation) {
  LP_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto axis_tensor = converter->ConvertOperand(axis_operand);
  auto lp_normalization_node = converter->network()->AddINormalizeNode(
      input_tensor, axis_tensor, nullptr);
  NNADAPTER_CHECK(lp_normalization_node)
      << "Failed to add lp_normalization node.";
  lp_normalization_node->SetP(static_cast<float>(p));
  lp_normalization_node->SetEpsilon(epsilon);
  auto output_tensor = lp_normalization_node->GetOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
