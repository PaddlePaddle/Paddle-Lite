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

#include "operation/binary_logical_op.h"
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertBinaryLogicalOp(Converter* converter, core::Operation* operation) {
  BINARY_LOGICAL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input0_tensor = converter->GetMappedTensor(input0_operand);
  if (!input0_tensor) {
    input0_tensor = converter->ConvertOperand(input0_operand);
  }
  auto input1_tensor = converter->GetMappedTensor(input1_operand);
  if (!input1_tensor) {
    input1_tensor = converter->ConvertOperand(input1_operand);
  }
  switch (operation->type) {
#define CONVERT_BINARY_LOGICAL_OP(type, mm_type)                    \
  case NNADAPTER_##type: {                                          \
    auto binary_logical_node = converter->network()->AddILogicNode( \
        input0_tensor, input1_tensor, mm_type);                     \
    auto output_tensor = binary_logical_node->GetOutput(0);         \
    converter->UpdateTensorMap(output_operand, output_tensor);      \
  } break;
    CONVERT_BINARY_LOGICAL_OP(AND, magicmind::ILogic::AND);
    CONVERT_BINARY_LOGICAL_OP(OR, magicmind::ILogic::OR);
#undef CONVERT_BINARY_LOGICAL_OP
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported binary logical operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
