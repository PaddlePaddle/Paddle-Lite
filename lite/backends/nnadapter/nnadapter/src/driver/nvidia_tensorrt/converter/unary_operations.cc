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

#include "driver/nvidia_tensorrt/converter/converter.h"
#include "operation/unary_activations.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertUnaryOperations(Converter* converter, core::Operation* operation) {
  UNARY_ACTIVATIONS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  switch (operation->type) {
#define CONVERT_UNARY_OPERATION(type, unary_operation_type)             \
  case NNADAPTER_##type: {                                              \
    auto unary_layer = converter->network()->addUnary(                  \
        *input_tensor, nvinfer1::UnaryOperation::unary_operation_type); \
    auto output_tensor = unary_layer->getOutput(0);                     \
    converter->UpdateTensorMap(output_operand, output_tensor);          \
  } break;
    CONVERT_UNARY_OPERATION(EXP, kEXP);
    CONVERT_UNARY_OPERATION(LOG, kLOG);
#undef CONVERT_UNARY_OPERATION
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported unary operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
