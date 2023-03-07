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
  auto input_operand =
      converter->AddInputOperand(scope, input_name, {}, input_scales);
  CHECK(input_operand);
  auto input_type = converter->GetOperandType(input_operand);
  // Axes
  auto axes = op->GetAttr<std::vector<int>>("axes");
  NNAdapterOperand* axes_operand = converter->AddConstantOperand(axes);
  auto axes_size = static_cast<int>(axes.size());
  // Starts
  NNAdapterOperand* starts_operand = nullptr;
  if (HasInput(op, scope, "StartsTensor")) {
    auto name = op->Input("StartsTensor").front();
    starts_operand = converter->AddInputOperand(scope, name);
  } else if (HasInput(op, scope, "StartsTensorList")) {
    auto names = op->Input("StartsTensorList");
    if (names.size() == 1) {
      starts_operand = converter->AddInputOperand(scope, names[0]);
    } else {
      std::vector<NNAdapterOperand*> concat_input_operands;
      for (auto name : names) {
        auto sub_start_operand = converter->AddInputOperand(scope, name);
        concat_input_operands.push_back(sub_start_operand);
      }
      NNAdapterOperand* concat_axis_operand =
          converter->AddConstantOperand<int>(0);
      concat_input_operands.push_back(concat_axis_operand);
      starts_operand = converter->AddOutputOperand();
      converter->AddOperation(
          NNADAPTER_CONCAT, concat_input_operands, {starts_operand});
    }
  } else {
    CHECK(op->HasAttr("starts")) << "One of 'StartsTensor', "
                                    "'StartsTensorList', 'starts'(attr) must "
                                    "be exist.";
    auto starts = op->GetAttr<std::vector<int>>("starts");
    starts_operand = converter->AddConstantOperand(starts);
  }
  // Ends
  NNAdapterOperand* ends_operand = nullptr;
  if (HasInput(op, scope, "EndsTensor")) {
    auto name = op->Input("EndsTensor").front();
    ends_operand = converter->AddInputOperand(scope, name);
  } else if (HasInput(op, scope, "EndsTensorList")) {
    auto names = op->Input("EndsTensorList");
    if (names.size() == 1) {
      ends_operand = converter->AddInputOperand(scope, names[0]);
    } else {
      std::vector<NNAdapterOperand*> concat_input_operands;
      for (auto name : names) {
        auto sub_end_operand = converter->AddInputOperand(scope, name);
        concat_input_operands.push_back(sub_end_operand);
      }
      NNAdapterOperand* concat_axis_operand =
          converter->AddConstantOperand<int>(0);
      concat_input_operands.push_back(concat_axis_operand);
      ends_operand = converter->AddOutputOperand();
      converter->AddOperation(
          NNADAPTER_CONCAT, concat_input_operands, {ends_operand});
    }
  } else {
    CHECK(op->HasAttr("ends"))
        << "One of 'EndsTensor', 'EndsTensorList', 'ends'(attr) must be exist.";
    auto ends = op->GetAttr<std::vector<int>>("ends");
    ends_operand = converter->AddConstantOperand(ends);
  }
  // Steps
  std::vector<int> steps(axes_size, 1);
  auto steps_operand = converter->AddConstantOperand(steps);
  // Decrease axis
  std::vector<int> decrease_axis;
  if (op->HasAttr("decrease_axis")) {
    decrease_axis = op->GetAttr<std::vector<int>>("decrease_axis");
  }
  // Output
  auto out_name = op->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  std::vector<float> out_scales;
  if (op->HasOutputScale(out_scale_name, true)) {
    out_scales = op->GetOutputScale(out_scale_name, true);
  }
  auto output_operand = converter->AddOutputOperand(out_name, out_scales);

  // Slice operation
  converter->AddOperation(NNADAPTER_SLICE,
                          {input_operand,
                           axes_operand,
                           starts_operand,
                           ends_operand,
                           steps_operand},
                          {output_operand});

  // Use squeeze to process decrease_axis(attr)
  if (!decrease_axis.empty() &&
      decrease_axis.size() != input_type->dimensions.count) {
    // Squeeze operation
    converter->AddSqueezeOperation(
        output_operand, decrease_axis, out_name, out_scales);
  }
  if (decrease_axis.size() == input_type->dimensions.count &&
      decrease_axis.size() > 1) {
    std::vector<int> shape = {1};
    converter->AddReshapeOperation(output_operand, shape, out_name, out_scales);
  }
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
