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

#include "operation/layer_normalization.h"
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertLayerNormalization(Converter* converter,
                              core::Operation* operation) {
  LAYER_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (input_tensor == nullptr) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto scale_tensor = converter->ConvertOperand(scale_operand);
  auto bias_tensor = converter->ConvertOperand(bias_operand);
  std::vector<int64_t> normalized_shape = {};
  auto input_dimension_count = input_operand->type.dimensions.count;
  NNADAPTER_CHECK_LT(begin_norm_axis, input_dimension_count);
  for (int i = begin_norm_axis; i < input_dimension_count; i++) {
    normalized_shape.push_back(
        static_cast<int64_t>(input_operand->type.dimensions.data[i]));
  }
  // Layer normalization
  auto layer_norm_node = converter->network()->AddILayerNormNode(
      input_tensor, scale_tensor, bias_tensor, normalized_shape, epsilon);
  NNADAPTER_CHECK(layer_norm_node) << "Failed to add layer_norm node.";
  auto output_tensor = layer_norm_node->GetOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
