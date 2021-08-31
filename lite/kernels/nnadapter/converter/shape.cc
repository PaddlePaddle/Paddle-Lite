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

int ConvertShape(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto x_name = op->Input("Input").front();
  auto* input_operand = converter->GetMappedOperand(x_name);

  // Dtype operand
  auto dtype_operand = converter->AddConstantOperand(
      static_cast<int32_t>(NNADAPTER_TENSOR_INT32));

  // Shape operand
  auto out_name = op->Output("Out").front();
  auto shape_operand = converter->AddShapeOperand(out_name);

  // Shape operation
  std::vector<NNAdapterOperand*> shape_input_operands = {input_operand,
                                                         dtype_operand};
  std::vector<NNAdapterOperand*> shape_output_operands = {shape_operand};
  converter->AddOperation(
      NNADAPTER_SHAPE, &shape_input_operands, &shape_output_operands);
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
