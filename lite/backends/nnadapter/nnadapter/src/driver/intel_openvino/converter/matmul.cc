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

#include "driver/intel_openvino/converter/converter.h"
#include "operation/mat_mul.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertMatMul(Converter* converter, core::Operation* operation) {
  MAT_MUL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert operand to OpenVINO tensor
  auto x_tensor = converter->GetMappedTensor(x_operand);
  if (!x_tensor) {
    x_tensor = converter->ConvertOperand(x_operand);
  }
  auto y_tensor = converter->GetMappedTensor(y_operand);
  if (!y_tensor) {
    y_tensor = converter->ConvertOperand(y_operand);
  }
  auto matmul_op = std::make_shared<default_opset::MatMul>(
      *x_tensor, *y_tensor, transpose_x, transpose_y);
  MAP_OUTPUT(output_operand, matmul_op, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
