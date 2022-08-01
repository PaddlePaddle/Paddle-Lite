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

#include "operation/elementwise.h"
#include "driver/android_nnapi/converter/converter.h"
#include "driver/android_nnapi/converter/validator.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace android_nnapi {

bool ValidateElementwise(Validator* validator,
                         const core::Operation* operation) {
  return true;
}

int ConvertElementwise(Converter* converter, core::Operation* operation) {
  ELEMENTWISE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to NNAPI operands and operations
  auto input0_index = converter->GetMappedIndex(input0_operand);
  if (input0_index == INVALID_INDEX) {
    input0_index = converter->ConvertOperand(input0_operand);
  }
  auto input1_index = converter->GetMappedIndex(input1_operand);
  if (input1_index == INVALID_INDEX) {
    input1_index = converter->ConvertOperand(input1_operand);
  }
  auto fuse_code_index = converter->AddInt32ConstantOperand(
      ConvertFuseCodeToNNFuseCode(fuse_code));
  auto output_index = converter->ConvertOperand(output_operand);
  ANeuralNetworksOperationType op_type;
  if (operation->type == NNADAPTER_ADD) {
    op_type = ANEURALNETWORKS_ADD;
  } else if (operation->type == NNADAPTER_SUB) {
    op_type = ANEURALNETWORKS_SUB;
  } else if (operation->type == NNADAPTER_MUL) {
    op_type = ANEURALNETWORKS_MUL;
  } else if (operation->type == NNADAPTER_DIV) {
    op_type = ANEURALNETWORKS_DIV;
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported element-wise operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  NNADAPTER_CHECK_EQ(
      converter->AddOperation(op_type,
                              {input0_index, input1_index, fuse_code_index},
                              {output_index}),
      ANEURALNETWORKS_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

}  // namespace android_nnapi
}  // namespace nnadapter
