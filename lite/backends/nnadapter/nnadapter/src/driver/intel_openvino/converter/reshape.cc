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

#include "operation/reshape.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertReshape(Converter* converter, core::Operation* operation) {
  RESHAPE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert operand to Intel OpenVINO's OutputNode
  auto input_node = converter->GetMappedOutputNode(input_operand);
  if (!input_node) {
    input_node = converter->ConvertToOutputNode(input_operand);
  }
  // Create <Reshape> Node for Intel OpenVINO
  if (IsConstantOperand(shape_operand)) {
    auto shape_count = shape_operand->length / sizeof(int32_t);
    auto shape_data = reinterpret_cast<int32_t*>(shape_operand->buffer);
    for (uint32_t i = 0; i < shape_count; i++) {
      if (shape_data[i] == 0 &&
          input_operand->type.dimensions.data[i] != NNADAPTER_UNKNOWN) {
        shape_data[i] = input_operand->type.dimensions.data[i];
      }
    }
    auto shape_node = AddConstOutputNode<int32_t>(
        {shape_count},
        std::vector<int32_t>(shape_data, shape_data + shape_count));
    auto node = std::make_shared<default_opset::Reshape>(
        *input_node, *shape_node, true);
    auto output_node = std::make_shared<OutputNode>(node->output(0));
    converter->UpdateOutputNodeMap(output_operand, output_node);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported shape lifetime: "
                         << OperandLifetimeCodeToString(
                                shape_operand->type.lifetime);
    return NNADAPTER_INVALID_PARAMETER;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
