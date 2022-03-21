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

typedef struct ActivationParam {
  nvinfer1::ActivationType type;
  float alpha;
  float beta;
} ActivationParam;

int ConvertActivations(Converter* converter, core::Operation* operation) {
  UNARY_ACTIVATIONS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  switch (operation->type) {
#define CONVERT_ACTIVATION(type, class_name, alpha, beta)        \
  case NNADAPTER_##type: {                                       \
    auto activation_layer = converter->network()->addActivation( \
        *input_tensor, nvinfer1::ActivationType::class_name);    \
    activation_layer->setAlpha(alpha);                           \
    activation_layer->setBeta(beta);                             \
    auto output_tensor = activation_layer->getOutput(0);         \
    converter->UpdateTensorMap(output_operand, output_tensor);   \
  } break;
    CONVERT_ACTIVATION(RELU, kRELU, 0, 0);
    CONVERT_ACTIVATION(RELU6, kCLIP, 0, 6);
    CONVERT_ACTIVATION(SIGMOID, kSIGMOID, 0, 0);
    CONVERT_ACTIVATION(TANH, kTANH, 0, 0);
#undef CONVERT_ACTIVATION
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported activation operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
