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

#include "operation/top_k.h"
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertTopK(Converter* converter, core::Operation* operation) {
  TOP_K_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto k_tensor = converter->GetMappedTensor(k_operand);
  if (!k_tensor) {
    k_tensor = converter->ConvertOperand(k_operand);
  }
  auto top_k_node = converter->network()->AddITopKNode(input_tensor, k_tensor);
  NNADAPTER_CHECK(top_k_node) << "Failed to add topk node.";
  top_k_node->SetAxis(static_cast<int64_t>(axis));
  top_k_node->SetLargest(largest);
  top_k_node->SetSorted(sorted);
  auto output_tensor = top_k_node->GetOutput(0);
  converter->UpdateTensorMap(output_operands[0], output_tensor);
  auto indices_tensor = top_k_node->GetOutput(1);
  // topk currently only support indices_dtype is INT32.
  if (return_indices_dtype == NNADAPTER_INT64) {
    auto cast_node = converter->network()->AddICastNode(
        indices_tensor, magicmind::DataType::INT64);
    NNADAPTER_CHECK(cast_node) << "Failed to add cast node.";
    auto cast_out_tensor = cast_node->GetOutput(0);
    converter->UpdateTensorMap(output_operands[1], cast_out_tensor);
  } else {
    converter->UpdateTensorMap(output_operands[1], indices_tensor);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
