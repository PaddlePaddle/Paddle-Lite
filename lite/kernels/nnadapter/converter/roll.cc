// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

int ConvertRoll(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);
  // Shifts
  NNAdapterOperand* shifts_operand = nullptr;
  if (HasInput(op, scope, "ShiftsTensor")) {
    auto shifts_name = op->Input("ShiftsTensor").front();
    shifts_operand = converter->AddInputOperand(scope, shifts_name);
  } else {
    std::vector<int64_t> shifts = op->GetAttr<std::vector<int64_t>>("shifts");
    shifts_operand = converter->AddConstantOperand(
        std::vector<int32_t>(shifts.begin(), shifts.end()));
  }
  // Axes
  std::vector<int64_t> axis = op->GetAttr<std::vector<int64_t>>("axis");
  auto axes_operand = converter->AddConstantOperand(
      std::vector<int32_t>(axis.begin(), axis.end()));
  // Output
  auto out_name = op->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  std::vector<float> out_scales;
  if (op->HasOutputScale(out_scale_name, true)) {
    out_scales = op->GetOutputScale(out_scale_name, true);
  }
  auto output_operand = converter->AddOutputOperand(out_name, out_scales);
  // Roll operation
  converter->AddOperation(NNADAPTER_ROLL,
                          {input_operand, shifts_operand, axes_operand},
                          {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
