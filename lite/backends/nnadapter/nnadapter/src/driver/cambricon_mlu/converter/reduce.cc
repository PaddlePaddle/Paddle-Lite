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

#include "operation/reduce.h"
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

const std::map<NNAdapterOperationType, magicmind::IReduce>*
ReduceOperationMap() {
  static auto* const m =
      new std::map<NNAdapterOperationType, magicmind::IReduce>{
          {NNADAPTER_REDUCE_MEAN, magicmind::IReduce::MEAN},
          {NNADAPTER_REDUCE_SUM, magicmind::IReduce::ADD},
      };
  return m;
}

int ConvertReduce(Converter* converter, core::Operation* operation) {
  REDUCE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto axes_tensor = converter->ConvertOperand(axes_operand);
  auto op_pair = ReduceOperationMap()->find(operation->type);
  auto reduce_node = converter->network()->AddIReduceNode(
      input_tensor, axes_tensor, op_pair->second, keep_dim);
  NNADAPTER_CHECK(reduce_node) << "Failed to add reduce node.";
  auto output_tensor = reduce_node->GetOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);

  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
