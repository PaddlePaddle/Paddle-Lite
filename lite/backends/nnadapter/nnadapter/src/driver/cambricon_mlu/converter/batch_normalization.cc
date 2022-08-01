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

#include "operation/batch_normalization.h"
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertBatchNormalization(Converter* converter,
                              core::Operation* operation) {
  BATCH_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto scale_tensor = converter->ConvertOperand(scale_operand);
  auto offset_tensor = converter->ConvertOperand(bias_operand);
  auto mean_tensor = converter->ConvertOperand(mean_operand);
  auto variance_tensor = converter->ConvertOperand(variance_operand);
  auto batch_norm_node = converter->network()->AddIFusedBatchNormNode(
      input_tensor, mean_tensor, variance_tensor, scale_tensor, offset_tensor);
  NNADAPTER_CHECK(batch_norm_node) << "Failed to add batch_norm node.";
  batch_norm_node->SetEpsilon(epsilon);
  int64_t axis = ConvertToMagicMindAxis(input_operand->type.layout);
  batch_norm_node->SetAxis(axis);
  auto output_tensor = batch_norm_node->GetOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
