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

#include "core/operation/fill_like.h"
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertFillLike(Converter* converter, hal::Operation* operation) {
  FILL_LIKE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (input_tensor == nullptr) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto shape_node = converter->network()->AddIShapeNode(input_tensor, nullptr);
  NNADAPTER_CHECK(shape_node) << "Failed to add shape node.";
  auto value_tensor = converter->GetMappedTensor(value_operand);
  if (value_tensor == nullptr) {
    value_tensor = converter->ConvertOperand(value_operand);
  }
  std::vector<int64_t> vec = {};
  value_tensor->SetDimension(magicmind::Dims(vec));
  auto fill_node = converter->network()->AddIFillNode(shape_node->GetOutput(0),
                                                      value_tensor);
  NNADAPTER_CHECK(fill_node) << "Failed to add fill node.";
  auto output_tensor = fill_node->GetOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
