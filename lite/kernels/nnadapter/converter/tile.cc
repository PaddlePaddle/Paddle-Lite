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

int ConvertTile(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);

  // Output operand
  auto out_name = op->Output("Out").front();
  auto output_operand = converter->AddOutputOperand(out_name);

  // Repeats operand
  NNAdapterOperand* repeats_operand = nullptr;
  if (HasInput(op, scope, "RepeatTimes") && !op->Input("RepeatTimes").empty()) {
    auto repeats_name = op->Input("RepeatTimes").front();
    repeats_operand = converter->AddInputOperand(scope, repeats_name);
  } else if (HasInput(op, scope, "repeat_times_tensor") &&
             !op->Input("repeat_times_tensor").empty()) {
    std::vector<NNAdapterOperand*> repeat_operands;
    for (auto repeat_tensor_name : op->Input("repeat_times_tensor")) {
      auto repeat_operand =
          converter->AddInputOperand(scope, repeat_tensor_name);
      repeat_operands.push_back(repeat_operand);
    }
    auto axis_operand = converter->AddConstantOperand(0);
    repeat_operands.push_back(axis_operand);
    repeats_operand = converter->AddOutputOperand(out_name + "/concat");
    converter->AddOperation(
        NNADAPTER_CONCAT, repeat_operands, {repeats_operand});
  } else {
    auto repeats = op->GetAttr<std::vector<int>>("repeat_times");
    repeats_operand = converter->AddConstantOperand(repeats);
  }

  // Tile operation
  converter->AddOperation(
      NNADAPTER_TILE, {input_operand, repeats_operand}, {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
