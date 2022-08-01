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

int ConvertExpandV2(Converter* converter, OpInfo* op, Scope* scope) {
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

  // Shape operand
  NNAdapterOperand* shape_operand = nullptr;
  if (HasInput(op, scope, "Shape")) {
    auto shape_name = op->Input("Shape").front();
    shape_operand = converter->AddInputOperand(scope, shape_name);
  } else if (HasInput(op, scope, "expand_shapes_tensor") &&
             !op->Input("expand_shapes_tensor").empty()) {
    std::vector<NNAdapterOperand*> expand_shapes_operands;
    for (auto expand_shapes_tensor_name : op->Input("expand_shapes_tensor")) {
      auto expand_shapes_tensor_scale_name =
          expand_shapes_tensor_name + "_scale";
      std::vector<float> expand_shapes_tensor_scales;
      if (op->HasInputScale(expand_shapes_tensor_scale_name, true)) {
        expand_shapes_tensor_scales =
            op->GetInputScale(expand_shapes_tensor_scale_name, true);
      }
      auto expand_shapes_operand = converter->AddInputOperand(
          scope, expand_shapes_tensor_name, {}, expand_shapes_tensor_scales);
      expand_shapes_operands.push_back(expand_shapes_operand);
    }
    auto axis_operand = converter->AddConstantOperand(0);
    expand_shapes_operands.push_back(axis_operand);
    shape_operand = converter->AddOutputOperand(out_name + "/concat");
    // Concat operation
    converter->AddOperation(
        NNADAPTER_CONCAT, expand_shapes_operands, {shape_operand});
  } else {
    auto shape = op->GetAttr<std::vector<int>>("shape");
    shape_operand = converter->AddConstantOperand(shape);
  }

  // Expand operation
  converter->AddOperation(
      NNADAPTER_EXPAND, {input_operand, shape_operand}, {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
