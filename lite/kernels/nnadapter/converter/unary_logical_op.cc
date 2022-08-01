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

int ConvertUnaryLogicalOp(Converter* converter, OpInfo* op, Scope* scope) {
  // Extract op attributes
  auto x_name = op->Input("X").front();
  auto output_name = op->Output("Out").front();

  // Convert to NNAdapter operands and operation
  auto input_operand = converter->AddInputOperand(scope, x_name);
  auto output_operand = converter->AddOutputOperand(output_name);
  // Logic operation
  converter->AddOperation(NNADAPTER_NOT, {input_operand}, {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
