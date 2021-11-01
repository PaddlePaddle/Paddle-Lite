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

int ConvertSplit(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);

  // Axis operand
  // Priority: AxisTensor > axis(attr)
  NNAdapterOperand* axis_operand = nullptr;
  if (HasInput(op, scope, "AxisTensor")) {
    auto axis_name = op->Input("AxisTensor").front();
    axis_operand = converter->AddInputOperand(scope, axis_name);
  } else if (op->HasAttr("axis")) {
    int axis = op->GetAttr<int>("axis");
    axis_operand = converter->AddConstantOperand(axis);
  } else {
    LOG(FATAL) << "Either AxisTenor or axis(attr) should be exist.";
  }

  // Split operand
  // Priority: num(attr) > SectionsTensorList > sections(attr)
  NNAdapterOperand* split_operand = nullptr;
  int num = op->GetAttr<int>("num");
  if (num > 0) {
    // Not support dynamic shape now.
    int axis;
    if (HasInput(op, scope, "AxisTensor")) {
      auto axis_tensor = scope->FindTensor(op->Input("AxisTensor").front());
      CHECK(axis_tensor->persistable());
      axis = axis_tensor->data<int>()[0];
    } else {
      axis = op->GetAttr<int>("axis");
    }
    auto input_dimensions =
        converter->GetOperandType(input_operand)->dimensions;
    if (axis < 0) {
      axis += input_dimensions.count;
    }
    int size = input_dimensions.data[axis];
    CHECK_GE(size, 0);
    CHECK_EQ(size % num, 0);
    std::vector<int> sections(num, size / num);
    split_operand = converter->AddConstantOperand(sections);
  } else if (HasInput(op, scope, "SectionsTensorList")) {
    std::vector<int> sections;
    auto names = op->Input("SectionsTensorList");
    for (auto name : names) {
      auto section_tensor = scope->FindTensor(name);
      CHECK(section_tensor->persistable());
      sections.push_back(section_tensor->data<int>()[0]);
    }
    split_operand = converter->AddConstantOperand(sections);
  } else {
    std::vector<int> sections = op->GetAttr<std::vector<int>>("sections");
    split_operand = converter->AddConstantOperand(sections);
  }

  // Output operand
  std::vector<NNAdapterOperand*> output_operands;
  auto out_names = op->Output("Out");
  for (size_t i = 0; i < out_names.size(); i++) {
    auto out_name = out_names[i];
    auto out_scale_name = "Out" + std::to_string(i) + "_scale";
    std::vector<float> out_scales;
    if (op->HasOutputScale(out_scale_name, true)) {
      out_scales = op->GetOutputScale(out_scale_name, true);
    }
    output_operands.push_back(
        converter->AddOutputOperand(out_name, out_scales));
  }

  // Split operation
  converter->AddOperation(NNADAPTER_SPLIT,
                          {input_operand, axis_operand, split_operand},
                          output_operands);
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
