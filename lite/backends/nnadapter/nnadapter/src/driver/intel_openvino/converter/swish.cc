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

#include "driver/intel_openvino/converter/converter.h"
#include "operation/unary_activations.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertSwish(Converter* converter, core::Operation* operation) {
  UNARY_ACTIVATIONS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto beta_tensor = converter->AddConstantTensor<float>(1.0f);
  auto swish_op =
      std::make_shared<default_opset::Swish>(*input_tensor, *beta_tensor);
  MAP_OUTPUT(output_operand, swish_op, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
