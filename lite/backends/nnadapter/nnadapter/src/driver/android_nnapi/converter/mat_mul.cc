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

#include "operation/mat_mul.h"
#include "driver/android_nnapi/converter/converter.h"
#include "driver/android_nnapi/converter/validator.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace android_nnapi {

bool ValidateMatmul(Validator* validator, const core::Operation* operation) {
  return true;
}

int ConvertMatmul(Converter* converter, core::Operation* operation) {
  MAT_MUL_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // TODO(zhupengyang): support by reshape or squeeze
  NNADAPTER_CHECK_NE(x_operand->type.dimensions.count, 1);
  NNADAPTER_CHECK_NE(y_operand->type.dimensions.count, 1);

  // Convert to NNAPI operands and operations
  auto x_index = converter->GetMappedIndex(x_operand);
  if (x_index == INVALID_INDEX) {
    x_index = converter->ConvertOperand(x_operand);
  }
  auto y_index = converter->GetMappedIndex(y_operand);
  if (y_index == INVALID_INDEX) {
    y_index = converter->ConvertOperand(y_operand);
  }
  auto output_index = converter->ConvertOperand(output_operand);
  NNADAPTER_CHECK_EQ(
      converter->AddOperation(
          ANEURALNETWORKS_BATCH_MATMUL, {x_index, y_index}, {output_index}),
      ANEURALNETWORKS_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

}  // namespace android_nnapi
}  // namespace nnadapter
