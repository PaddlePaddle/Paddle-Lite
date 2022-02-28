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

#include "operation/comparisons.h"
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

const std::map<NNAdapterOperationType, magicmind::ILogic>* LogicOperationMap() {
  static auto* const m =
      new std::map<NNAdapterOperationType, magicmind::ILogic>{
          {NNADAPTER_EQUAL, magicmind::ILogic::EQ},
          {NNADAPTER_NOT_EQUAL, magicmind::ILogic::NE},
          {NNADAPTER_GREATER, magicmind::ILogic::GT},
          {NNADAPTER_GREATER_EQUAL, magicmind::ILogic::GE},
          {NNADAPTER_LESS, magicmind::ILogic::LT},
          {NNADAPTER_LESS_EQUAL, magicmind::ILogic::LE},
      };
  return m;
}

int ConvertComparisons(Converter* converter, core::Operation* operation) {
  COMPARISONS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input0_tensor = converter->GetMappedTensor(input0_operand);
  if (!input0_tensor) {
    input0_tensor = converter->ConvertOperand(input0_operand);
  }
  auto input1_tensor = converter->GetMappedTensor(input1_operand);
  if (!input1_tensor) {
    input1_tensor = converter->ConvertOperand(input1_operand);
  }
  auto op_pair = LogicOperationMap()->find(operation->type);
  if (op_pair == LogicOperationMap()->end()) {
    NNADAPTER_VLOG(5) << "Unsupported logic op.";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  auto logic_node = converter->network()->AddILogicNode(
      input0_tensor, input1_tensor, op_pair->second);
  NNADAPTER_CHECK(logic_node) << "Failed to add logic node.";
  auto output_tensor = logic_node->GetOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
