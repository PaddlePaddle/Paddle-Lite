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

int ConvertSlice(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto input_name = op->Input("Input").front();
  auto input_scale_name = "Input0_scale";
  std::vector<float> input_scales;
  if (op->HasInputScale(input_scale_name, true)) {
    input_scales = op->GetInputScale(input_scale_name, true);
  }
  auto input_tensor = scope->FindTensor(input_name);
  auto input_dims = input_tensor->dims();
  auto input_operand = converter->AddInputOperand(
      input_name, *input_tensor, {}, true, input_scales);
  // Axes operand
  auto axes = op->GetAttr<std::vector<int>>("axes");
  auto axes_size = static_cast<int>(axes.size());
  auto axes_operand = converter->AddConstantOperand(axes);
  // Starts operand
  auto starts = op->GetAttr<std::vector<int>>("starts");
  auto starts_operand = converter->AddConstantOperand(starts);
  // Ends operand
  auto ends_ori = op->GetAttr<std::vector<int>>("ends");
  std::vector<int> ends(axes_size, 0);
  for (int i = 0; i < axes_size; i++) {
    ends[i] =
        ends_ori[i] > input_dims[axes[i]] ? input_dims[axes[i]] : ends_ori[i];
  }
  auto ends_operand = converter->AddConstantOperand(ends);
  // Steps operand
  std::vector<int> steps(axes_size, 1);
  auto steps_operand = converter->AddConstantOperand(steps);
  // Output operand
  std::vector<int> decrease_axis;
  if (op->HasAttr("decrease_axis")) {
    decrease_axis = op->GetAttr<std::vector<int>>("decrease_axis");
  }
  auto out_name = op->Output("Out").front();
  auto out_tensor = scope->FindTensor(out_name);
  auto out_dims = out_tensor->dims();
  std::string out_name_slice =
      decrease_axis.empty() ? out_name : out_name + "_squeeze_in";
  auto output_operand = converter->AddOutputOperand(out_name_slice);

  // Slice operation
  converter->AddOperation(NNADAPTER_SLICE,
                          {input_operand,
                           axes_operand,
                           starts_operand,
                           ends_operand,
                           steps_operand},
                          {output_operand});

  // Use squeeze to process decrease_axis(attr)
  if (!decrease_axis.empty()) {
    auto squeeze_axes_operand = converter->AddConstantOperand(decrease_axis);
    auto squeeze_output_operand = converter->AddOutputOperand(out_name);
    // Squeeze operation
    converter->AddOperation(NNADAPTER_SQUEEZE,
                            {output_operand, squeeze_axes_operand},
                            {squeeze_output_operand});
  }

  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
