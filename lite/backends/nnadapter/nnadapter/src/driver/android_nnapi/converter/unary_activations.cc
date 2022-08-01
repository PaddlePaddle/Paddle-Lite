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

#include "operation/unary_activations.h"
#include "driver/android_nnapi/converter/converter.h"
#include "driver/android_nnapi/converter/validator.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace android_nnapi {

bool ValidateUnaryActivations(Validator* validator,
                              const core::Operation* operation) {
  return true;
}

int ConvertUnaryActivations(Converter* converter, core::Operation* operation) {
  UNARY_ACTIVATIONS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to NNAPI operands and operations
  auto input_index = converter->GetMappedIndex(input_operand);
  if (input_index == INVALID_INDEX) {
    input_index = converter->ConvertOperand(input_operand);
  }
  auto output_index = converter->ConvertOperand(output_operand);
  ANeuralNetworksOperationType op_type;
  if (operation->type == NNADAPTER_SIGMOID) {
    op_type = ANEURALNETWORKS_LOGISTIC;
  } else if (operation->type == NNADAPTER_RELU) {
    op_type = ANEURALNETWORKS_RELU;
  } else if (operation->type == NNADAPTER_RELU6) {
    op_type = ANEURALNETWORKS_RELU6;
  } else if (operation->type == NNADAPTER_TANH) {
    op_type = ANEURALNETWORKS_TANH;
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported activation operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  NNADAPTER_CHECK_EQ(
      converter->AddOperation(op_type, {input_index}, {output_index}),
      ANEURALNETWORKS_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

}  // namespace android_nnapi
}  // namespace nnadapter
