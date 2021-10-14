// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "core/operation/elementwise.h"
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace cambricon_mlu {

const std::map<NNAdapterOperationType, magicmind::IElementwise>*
ElementwiseOperationMap() {
  static auto* const m =
      new std::map<NNAdapterOperationType, magicmind::IElementwise> {
        {NNADAPTER_ADD, magicmind::IElementwise::ADD},
        {NNADAPTER_MUL, magicmind::IElementwise::MUL},
        {NNADAPTER_SUB, magicmind::IElementwise::SUB},
  };
  return m;
}

int ConvertElementwise(Converter* converter, hal::Operation* operation) {
  ELEMENTWISE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input0_tensor = converter->GetMappedTensor(input0_operand);
  if (!input0_tensor) {
    input0_tensor = converter->ConvertOperand(input0_operand);
  }
  auto input1_tensor = converter->GetMappedTensor(input1_operand);
  if (!input1_tensor) {
    input1_tensor = converter->ConvertOperand(input1_operand);
  }

  auto op_pair = ElementwiseOperationMap()->find(operation->type);
  if (op_pair == BinaryOperationMap()->end()) {
    NNADAPTER_VLOG(5) << "Unsupported binary op.";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  auto elementwise_node = converter->network()->AddIElementwiseNode(
      input0_tensor, input1_tensor, op_pair->second);
  if (elementwise_node == nullptr) {
    NNADAPTER_VLOG(5) << "Failed to add elementwise node.";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  auto output_tensor = elementwise_node->GetOutput(0);
  // fuse activations ?
  switch (fuse_code) {
    case NNADAPTER_FUSED_RELU:
      {
        auto activation_node = converter->network()->AddIActivationNode(output_tensor, magicmind::IActivation::RELU);
        auto fuse_out_tensor = activation_node->GetOutput(0);
        converter->UpdateTensorMap(output_operand, fuse_out_tensor);
        break;
      }
    case NNADAPTER_FUSED_RELU6:
      {
        auto activation_node = converter->network()->AddIActivationNode(output_tensor, magicmind::IActivation::RELU6);
        auto fuse_out_tensor = activation_node->GetOutput(0);
        converter->UpdateTensorMap(output_operand, fuse_out_tensor);
        break;
      }
    case NNADAPTER_FUSED_NONE:
      converter->UpdateTensorMap(output_operand, output_tensor);
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported fuse_code(" << fuse_code
                           << ") is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
