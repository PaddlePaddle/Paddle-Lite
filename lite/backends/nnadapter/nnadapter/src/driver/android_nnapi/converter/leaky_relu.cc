// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/leaky_relu.h"
#include <memory>
#include "driver/android_nnapi/converter/converter.h"
#include "driver/android_nnapi/converter/validator.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace android_nnapi {

bool ValidateLeakyRelu(Validator* validator, const core::Operation* operation) {
  return true;
}

int ConvertLeakyRelu(Converter* converter, core::Operation* operation) {
  LEAKY_RELU_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to NNAPI operands and operations
  auto input_index = converter->GetMappedIndex(input_operand);
  if (input_index == INVALID_INDEX) {
    input_index = converter->ConvertOperand(input_operand);
  }
  uint32_t alpha_index = INVALID_INDEX;
  auto input_precision = input_operand->type.precision;
  switch (input_precision) {
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER: {
      float scale = alpha;
      int32_t zero_point = 0;
      alpha_index = converter->AddQuant8ConstantOperand(
          std::vector<uint8_t>({1}), scale, zero_point);
    } break;
    case NNADAPTER_FLOAT32:
      alpha_index =
          converter->AddFloat32ConstantOperand(std::vector<float>({alpha}));
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported precision code "
                           << OperandPrecisionCodeToString(input_precision)
                           << " is found.";
      break;
  }
  auto output_index = converter->ConvertOperand(output_operand);
  NNADAPTER_CHECK_EQ(
      converter->AddOperation(
          ANEURALNETWORKS_PRELU, {input_index, alpha_index}, {output_index}),
      ANEURALNETWORKS_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

}  // namespace android_nnapi
}  // namespace nnadapter
