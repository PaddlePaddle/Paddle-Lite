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

#include "driver/cambricon_mlu/converter.h"
#include "operation/unary_activations.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

const std::map<NNAdapterOperationType, magicmind::IActivation>*
ActivationOperationMap() {
  static auto* const m =
      new std::map<NNAdapterOperationType, magicmind::IActivation>{
          {NNADAPTER_RELU, magicmind::IActivation::RELU},
          {NNADAPTER_RELU6, magicmind::IActivation::RELU6},
          {NNADAPTER_SIGMOID, magicmind::IActivation::SIGMOID},
          {NNADAPTER_TANH, magicmind::IActivation::TANH},
      };
  return m;
}

int ConvertActivations(Converter* converter, core::Operation* operation) {
  UNARY_ACTIVATIONS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto op_pair = ActivationOperationMap()->find(operation->type);
  if (op_pair == ActivationOperationMap()->end()) {
    NNADAPTER_VLOG(5) << "Unsupported activation op.";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  auto activation_node =
      converter->network()->AddIActivationNode(input_tensor, op_pair->second);
  NNADAPTER_CHECK(activation_node) << "Failed to add activation node.";
  auto output_tensor = activation_node->GetOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);

  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
