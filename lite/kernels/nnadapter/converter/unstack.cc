// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

int ConvertUnstack(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);
  auto input_dimensions = converter->GetOperandType(input_operand)->dimensions;

  // Axis operand
  int axis = op->GetAttr<int>("axis");
  if (axis < 0) {
    axis += input_dimensions.count;
  }
  auto axis_operand = converter->AddConstantOperand(axis);
  // Num operand
  int num = op->GetAttr<int>("num");
  CHECK_GE(num, 0);
  auto num_operand = converter->AddConstantOperand(num);
  // Output operand
  std::vector<NNAdapterOperand*> output_operands;
  auto out_names = op->Output("Y");
  for (size_t i = 0; i < out_names.size(); i++) {
    auto out_name = out_names[i];
    auto out_scale_name = "Y" + std::to_string(i) + "_scale";
    std::vector<float> out_scales;
    if (op->HasOutputScale(out_scale_name, true)) {
      out_scales = op->GetOutputScale(out_scale_name, true);
    }
    output_operands.push_back(
        converter->AddOutputOperand(out_name, out_scales));
  }

  // Unstack operation
  converter->AddOperation(NNADAPTER_UNSTACK,
                          {input_operand, axis_operand, num_operand},
                          output_operands);
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
