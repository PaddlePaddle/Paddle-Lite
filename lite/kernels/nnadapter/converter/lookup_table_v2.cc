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

#include "lite/kernels/nnadapter/converter/converter.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

int ConvertLookupTableV2(Converter* converter, OpInfo* op, Scope* scope) {
  // X operand
  NNAdapterOperand* x_operand = nullptr;
  auto x_name = op->Input("X").front();
  auto x_tensor = scope->FindTensor(x_name);
  if (x_tensor->persistable()) {
    x_operand = converter->AddConstantOperand(*x_tensor);
  } else {
    x_operand = converter->GetMappedOperand(x_name);
  }

  // Y operand
  NNAdapterOperand* y_operand = nullptr;
  auto y_name = op->Input("Y").front();
  auto y_tensor = scope->FindTensor(y_name);
  if (y_tensor->persistable()) {
    y_operand = converter->AddConstantOperand(*y_tensor);
  } else {
    y_operand = converter->GetMappedOperand(y_name);
  }

  // Transpose_x operand
  bool transpose_x = op->GetAttr<bool>("trans_x");
  auto transpose_x_operand = converter->AddConstantOperand(transpose_x);

  // Transpose_y operand
  bool transpose_y = op->GetAttr<bool>("trans_y");
  auto transpose_y_operand = converter->AddConstantOperand(transpose_y);

  // Output operand
  auto out_name = op->Output("Out").front();
  auto output_operand = converter->AddOutputOperand(out_name);

  // Mat_mul operation
  converter->AddOperation(
      NNADAPTER_MAT_MUL,
      {x_operand, y_operand, transpose_x_operand, transpose_y_operand},
      {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
