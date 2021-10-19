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

int ConvertTopK(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto x_name = op->Input("X").front();
  auto input_operand = converter->AddInputOperand(scope, x_name);

  // K operand
  NNAdapterOperand* k_operand = nullptr;
  if (HasInput(op, scope, "K")) {
    auto k_name = op->Input("K").front();
    k_operand = converter->AddInputOperand(scope, k_name);
  } else {
    int k = op->GetAttr<int>("k");
    k_operand = converter->AddConstantOperand(k);
  }

  // Axis operand
  int axis = op->HasAttr("axis") ? op->GetAttr<int>("axis") : -1;
  auto axis_operand = converter->AddConstantOperand(axis);

  // largest operand
  bool largest = op->HasAttr("largest") ? op->GetAttr<bool>("largest") : true;
  auto largest_operand =
      converter->AddConstantOperand(static_cast<int8_t>(largest));

  // sorted operand
  bool sorted = op->HasAttr("sorted") ? op->GetAttr<bool>("sorted") : true;
  auto sorted_operand =
      converter->AddConstantOperand(static_cast<int8_t>(sorted));

  // return_indices_dtype operand
  int return_indices_dtype = NNADAPTER_INT64;
  auto return_indices_dtype_operand =
      converter->AddConstantOperand(return_indices_dtype);

  // Output operand
  auto out_name = op->Output("Out").front();
  auto output_operand = converter->AddOutputOperand(out_name);

  // Indices operand
  auto indices_name = op->Output("Indices").front();
  auto indices_operand = converter->AddOutputOperand(indices_name);

  // TopK operation
  converter->AddOperation(NNADAPTER_TOP_K,
                          {input_operand,
                           k_operand,
                           axis_operand,
                           largest_operand,
                           sorted_operand,
                           return_indices_dtype_operand},
                          {output_operand, indices_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
