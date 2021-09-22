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

int ConvertCumsum(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto x_tensor = scope->FindTensor(x_name);
  auto input_operand =
      converter->AddInputOperand(x_name, *x_tensor, {}, true, x_scales);

  // Axis operand
  int axis = op->GetAttr<int>("axis");
  auto axis_operand = converter->AddConstantOperand(axis);

  // Exclusive operand
  bool exclusive = op->GetAttr<bool>("exclusive");
  auto exclusive_operand = converter->AddConstantOperand(exclusive);

  // Reverse operand
  bool reverse = op->GetAttr<bool>("reverse");
  auto reverse_operand = converter->AddConstantOperand(reverse);

  // Output operand
  auto out_name = op->Output("Out").front();
  auto output_operand = converter->AddOutputOperand(out_name);

  // Cumsum operation
  converter->AddOperation(
      NNADAPTER_CUM_SUM,
      {input_operand, axis_operand, exclusive_operand, reverse_operand},
      {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
