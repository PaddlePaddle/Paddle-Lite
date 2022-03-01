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

#include "operation/batch_normalization.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertBatchNormalization(Converter* converter,
                              core::Operation* operation) {
  BATCH_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert operand to Intel OpenVINO's OutputNode
  auto input_node = converter->GetMappedOutputNode(input_operand);
  if (!input_node) {
    input_node = converter->ConvertToOutputNode(input_operand);
  }
  auto gamma_node = converter->ConvertToOutputNode(scale_operand);
  auto beta_node = converter->ConvertToOutputNode(bias_operand);
  auto mean_node = converter->ConvertToOutputNode(mean_operand);
  auto variance_node = converter->ConvertToOutputNode(variance_operand);
  // Create <BatchNormInference> Node for Intel OpenVINO
  std::shared_ptr<Node> node =
      std::make_shared<default_opset::BatchNormInference>(*input_node,
                                                          *gamma_node,
                                                          *beta_node,
                                                          *mean_node,
                                                          *variance_node,
                                                          epsilon);
  auto output_node = std::make_shared<OutputNode>(node->output(0));
  converter->UpdateOutputNodeMap(output_operand, output_node);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
