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

int ConvertFlatten(Converter* converter, OpInfo* op, Scope* scope) {
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto out_name = op->Output("Out").front();
  auto axis = op->GetAttr<int>("axis");

  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);
  auto input_type = converter->GetOperandType(input_operand);
  axis = axis < 0 ? axis + input_type->dimensions.count : axis;
  NNAdapterOperand* output_operand = nullptr;
  if (axis == 0) {
    // Directly convert to reshape with shape[1,-1],
    auto shape_operand =
        converter->AddConstantOperand(std::vector<int32_t>{1, -1});
    output_operand = converter->AddOutputOperand(out_name);
    converter->AddOperation(
        NNADAPTER_RESHAPE, {input_operand, shape_operand}, {output_operand});
  } else if (axis == input_type->dimensions.count - 1) {
    converter->AddFlattenOperation(input_operand, 0, axis - 1, out_name);
  } else {
    // step1: flatten [0, axis)
    output_operand = converter->AddFlattenOperation(
        input_operand, 0, axis - 1, out_name + "/flatten_0_axis");
    // step2: flatten [axis, -1)
    int32_t start_axis = axis == 1 ? axis : axis - 1;
    converter->AddFlattenOperation(output_operand, start_axis, -1, out_name);
  }
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
