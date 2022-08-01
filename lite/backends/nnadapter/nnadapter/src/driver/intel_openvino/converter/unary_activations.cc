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

#include "operation/unary_activations.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertUnaryActivations(Converter* converter, core::Operation* operation) {
  UNARY_ACTIVATIONS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert operand to OpenVINO tensor
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  switch (operation->type) {
#define CONVERT_UNARY_ACTIVATION(type, class_name)                            \
  case NNADAPTER_##type: {                                                    \
    auto act_op = std::make_shared<default_opset::class_name>(*input_tensor); \
    MAP_OUTPUT(output_operand, act_op, 0);                                    \
  } break;
    CONVERT_UNARY_ACTIVATION(SIGMOID, Sigmoid);
    CONVERT_UNARY_ACTIVATION(RELU, Relu);
    CONVERT_UNARY_ACTIVATION(TANH, Tanh);
    CONVERT_UNARY_ACTIVATION(LOG, Log);
    CONVERT_UNARY_ACTIVATION(ABS, Abs);
    CONVERT_UNARY_ACTIVATION(EXP, Exp);
    CONVERT_UNARY_ACTIVATION(FLOOR, Floor);
#undef CONVERT_UNARY_ACTIVATION
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported activation operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
