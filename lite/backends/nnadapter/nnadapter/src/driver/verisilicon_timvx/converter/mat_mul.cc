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
#include "driver/verisilicon_timvx/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace verisilicon_timvx {

int ConvertMatMul(Converter* converter, core::Operation* operation) {
  MAT_MUL_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK_NE(x_operand->type.dimensions.count, 1);
  NNADAPTER_CHECK_NE(y_operand->type.dimensions.count, 1);

  // Convert to tim-vx tensors and operators
  auto x_tensor = converter->GetMappedTensor(x_operand);
  if (!x_tensor) {
    x_tensor = converter->ConvertOperand(x_operand);
  }

  auto y_tensor = converter->GetMappedTensor(y_operand);
  if (!y_tensor) {
    y_tensor = converter->ConvertOperand(y_operand);
  }
  auto output_tensor = converter->ConvertOperand(output_operand);

  auto mat_mul_op = converter->graph()->CreateOperation<tim::vx::ops::Matmul>(
      transpose_x, transpose_y, false, false);
  mat_mul_op->BindInputs({x_tensor, y_tensor});
  mat_mul_op->BindOutputs({output_tensor});
  return NNADAPTER_NO_ERROR;
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
