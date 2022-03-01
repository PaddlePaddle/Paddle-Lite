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

#include "operation/elementwise.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertElementwise(Converter* converter, core::Operation* operation) {
  ELEMENTWISE_OPERATION_EXTRACT_INPUTS_OUTPUTS
  
  // Convert operand to Intel OpenVINO's OutputNode
  auto input0_node = converter->GetMappedOutputNode(input0_operand);
  if (!input0_node) {
    input0_node = converter->ConvertToOutputNode(input0_operand);
  }
  auto input1_node = converter->GetMappedOutputNode(input1_operand);
  if (!input1_node) {
    input1_node = converter->ConvertToOutputNode(input1_operand);
  }
  // Create <ElementWise> Node for Intel OpenVINO
  std::shared_ptr<OutputNode> output_node{nullptr};
  switch (operation->type) {
#define CONVERT_ELEMENTWISE(type, class_name)                       \
  case NNADAPTER_##type: {                                          \
    std::shared_ptr<Node> node = std::make_shared<default_opset::class_name>(*input0_node, *input1_node); \
    output_node = std::make_shared<OutputNode>(node->output(0)); \
    converter->UpdateOutputNodeMap(output_operand, output_node); \
  } break;
    CONVERT_ELEMENTWISE(ADD, Add);
    CONVERT_ELEMENTWISE(SUB, Subtract);
    CONVERT_ELEMENTWISE(MUL, Multiply);
    CONVERT_ELEMENTWISE(DIV, Divide);
    CONVERT_ELEMENTWISE(MAX, Maximum);
    CONVERT_ELEMENTWISE(MIN, Minimum);
    CONVERT_ELEMENTWISE(POW, Power);
    CONVERT_ELEMENTWISE(EQUAL, Equal);
    CONVERT_ELEMENTWISE(GREATER_EQUAL, GreaterEqual);
#undef CONVERT_ELEMENTWISE
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported element-wise operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  // Fuse activation
  switch (fuse_code) {
#define CONVERT_UNARY_ACTIVATION(type, class_name)                            \
  case NNADAPTER_FUSED_##type: {                                              \
    std::shared_ptr<Node> act_node = std::make_shared<default_opset::class_name>(*output_node); \
    auto act_output_node = std::make_shared<OutputNode>(act_node->output(0)); \
    converter->UpdateOutputNodeMap(output_operand, act_output_node);                                    \
  } break;
    CONVERT_UNARY_ACTIVATION(RELU, Relu);
#undef CONVERT_UNARY_ACTIVATION
    case NNADAPTER_FUSED_NONE:
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported fuse_code(" << fuse_code
                           << ") is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
