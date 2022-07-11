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

#include "operation/comparisons.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertComparisons(Converter* converter, core::Operation* operation) {
  COMPARISONS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input0_tensor = converter->GetMappedTensor(input0_operand);
  if (!input0_tensor) {
    input0_tensor = converter->ConvertOperand(input0_operand);
  }
  auto input1_tensor = converter->GetMappedTensor(input1_operand);
  if (!input1_tensor) {
    input1_tensor = converter->ConvertOperand(input1_operand);
  }
  switch (operation->type) {
#define CONVERT_COMPARISON(type, class_name)                    \
  case NNADAPTER_##type: {                                      \
    auto comp_op = std::make_shared<default_opset::class_name>( \
        *input0_tensor, *input1_tensor);                        \
    MAP_OUTPUT(output_operand, comp_op, 0);                     \
  } break;
    CONVERT_COMPARISON(EQUAL, Equal);
    CONVERT_COMPARISON(NOT_EQUAL, NotEqual);
    CONVERT_COMPARISON(GREATER_EQUAL, GreaterEqual);
#undef CONVERT_COMPARISON
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported comparison operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
