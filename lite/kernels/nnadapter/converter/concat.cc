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

int ConvertConcat(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto x_names = op->Input("X");
  std::vector<NNAdapterOperand*> input_operands;
  for (size_t i = 0; i < x_names.size(); i++) {
    auto x_name = x_names[i];
    auto x_scale_name = "X" + paddle::lite::to_string(i) + "_scale";
    std::vector<float> x_scales;
    if (op->HasInputScale(x_scale_name, true)) {
      x_scales = op->GetInputScale(x_scale_name, true);
    }
    auto input_operand =
        converter->AddInputOperand(scope, x_name, {}, x_scales);
    input_operands.push_back(input_operand);
  }

  // Axis operand
  int axis = op->GetAttr<int>("axis");
  auto axis_operand = converter->AddConstantOperand(axis);
  input_operands.push_back(axis_operand);

  // Output operand
  auto output_name = op->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  std::vector<float> out_scales;
  if (op->HasOutputScale(out_scale_name, true)) {
    out_scales = op->GetOutputScale(out_scale_name, true);
  }
  auto output_operand = converter->AddOutputOperand(output_name, out_scales);

  // Concat operation
  converter->AddOperation(NNADAPTER_CONCAT, input_operands, {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
