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

#include "operation/mat_mul.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertMatMul(Converter* converter, core::Operation* operation) {
  MAT_MUL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert operand to Intel OpenVINO's OutputNode
  auto x_node = converter->GetMappedOutputNode(x_operand);
  if (!x_node) {
    x_node = converter->ConvertToOutputNode(x_operand);
  }
  auto y_node = converter->GetMappedOutputNode(y_operand);
  if (!y_node) {
    y_node = converter->ConvertToOutputNode(y_operand);
  }
  // Create <MatMul> Node for Intel OpenVINO
  std::shared_ptr<Node> node = std::make_shared<default_opset::MatMul>
    (*x_node, *y_node, transpose_x, transpose_y);
  auto output_node = std::make_shared<OutputNode>(node->output(0));
  converter->UpdateOutputNodeMap(output_operand, output_node);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
