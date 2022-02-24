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

#include "operation/split.h"
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertSplit(Converter* converter, core::Operation* operation) {
  SPLIT_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto axis_tensor = converter->GetMappedTensor(axis_operand);
  if (!axis_tensor) {
    axis_tensor = converter->ConvertOperand(axis_operand);
  }
  auto split_size_tensor = converter->GetMappedTensor(split_operand);
  if (!split_size_tensor) {
    split_size_tensor = converter->ConvertOperand(split_operand);
  }
  auto split_node =
      converter->network()->AddISplitNode(input_tensor,
                                          split_size_tensor,
                                          axis_tensor,
                                          static_cast<int64_t>(output_count));
  NNADAPTER_CHECK(split_node) << "Failed to add split node.";
  int64_t output_num = split_node->GetSplitNum();
  NNADAPTER_CHECK_EQ(output_num, output_count);
  for (size_t i = 0; i < output_count; ++i) {
    converter->UpdateTensorMap(output_operands[i], split_node->GetOutput(i));
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
