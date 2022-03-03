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

  // Convert operand to OpenVINO tensor
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto gamma_tensor = converter->ConvertOperand(scale_operand);
  auto beta_tensor = converter->ConvertOperand(bias_operand);
  auto mean_tensor = converter->ConvertOperand(mean_operand);
  auto variance_tensor = converter->ConvertOperand(variance_operand);
  auto batch_norm_op =
      std::make_shared<default_opset::BatchNormInference>(*input_tensor,
                                                          *gamma_tensor,
                                                          *beta_tensor,
                                                          *mean_tensor,
                                                          *variance_tensor,
                                                          epsilon);
  MAP_OUTPUT(output_operand, batch_norm_op, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
