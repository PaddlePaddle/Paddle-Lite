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

#include "operation/elementwise.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertElementwise(Converter* converter, core::Operation* operation) {
  ELEMENTWISE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert operand to OpenVINO tensor
  auto input0_tensor = converter->GetMappedTensor(input0_operand);
  if (!input0_tensor) {
    input0_tensor = converter->ConvertOperand(input0_operand);
  }
  auto input1_tensor = converter->GetMappedTensor(input1_operand);
  if (!input1_tensor) {
    input1_tensor = converter->ConvertOperand(input1_operand);
  }
  std::shared_ptr<Tensor> output_tensor{nullptr};
  switch (operation->type) {
#define CONVERT_ELEMENTWISE(type, class_name)                           \
  case NNADAPTER_##type: {                                              \
    auto element_wise_op = std::make_shared<default_opset::class_name>( \
        *input0_tensor, *input1_tensor);                                \
    output_tensor = MAP_OUTPUT(output_operand, element_wise_op, 0);     \
  } break;
    CONVERT_ELEMENTWISE(ADD, Add);
    CONVERT_ELEMENTWISE(SUB, Subtract);
    CONVERT_ELEMENTWISE(MUL, Multiply);
    CONVERT_ELEMENTWISE(DIV, Divide);
    CONVERT_ELEMENTWISE(MAX, Maximum);
    CONVERT_ELEMENTWISE(MIN, Minimum);
    CONVERT_ELEMENTWISE(POW, Power);
#undef CONVERT_ELEMENTWISE
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported element-wise operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  // Fuse activation
  switch (fuse_code) {
#define CONVERT_UNARY_ACTIVATION(type, class_name)                             \
  case NNADAPTER_FUSED_##type: {                                               \
    auto act_op = std::make_shared<default_opset::class_name>(*output_tensor); \
    MAP_OUTPUT(output_operand, act_op, 0);                                     \
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
