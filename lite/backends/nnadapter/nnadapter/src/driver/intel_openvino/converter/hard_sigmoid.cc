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

#include "driver/intel_openvino/converter/converter.h"
#include "operation/hard_sigmoid_swish.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertHardSigmoid(Converter* converter, core::Operation* operation) {
  HARD_SIGMOID_SWISH_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto alpha_tensor = converter->AddConstantTensor<float>(alpha);
  auto beta_tensor = converter->AddConstantTensor<float>(beta);
  auto hard_sigmoid_op = std::make_shared<default_opset::HardSigmoid>(
      *input_tensor, *alpha_tensor, *beta_tensor);
  MAP_OUTPUT(output_operand, hard_sigmoid_op, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
