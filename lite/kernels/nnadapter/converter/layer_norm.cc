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

int ConvertLayerNorm(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);
  CHECK(input_operand);
  auto input_type = converter->GetOperandType(input_operand);
  // Begin norm axis operand
  auto begin_norm_axis = op->GetAttr<int>("begin_norm_axis");
  auto begin_norm_axis_operand = converter->AddConstantOperand(begin_norm_axis);
  if (begin_norm_axis < 0) {
    begin_norm_axis += input_type->dimensions.count;
  }
  // Bias operand
  std::vector<int64_t> scale_bias_shape;
  uint32_t scale_bias_count = 1;
  for (int i = begin_norm_axis; i < input_type->dimensions.count; i++) {
    auto dim = input_type->dimensions.data[i];
    CHECK(dim != NNADAPTER_UNKNOWN);
    scale_bias_shape.push_back(dim);
    scale_bias_count *= dim;
  }
  NNAdapterOperand* bias_operand = nullptr;
  if (op->HasInput("Bias")) {
    auto bias_name = op->Input("Bias").front();
    auto bias_tensor = scope->FindMutableTensor(bias_name);
    CHECK(bias_tensor->persistable());
    bias_operand =
        converter->AddConstantOperand(*bias_tensor, DDim(scale_bias_shape));
  } else {
    bias_operand = converter->AddConstantOperand(
        std::vector<float>(scale_bias_count, 0), DDim(scale_bias_shape));
  }
  // Scale operand
  NNAdapterOperand* scale_operand = nullptr;
  if (op->HasInput("Scale")) {
    auto scale_name = op->Input("Scale").front();
    auto scale_tensor = scope->FindMutableTensor(scale_name);
    CHECK(scale_tensor->persistable());
    scale_operand =
        converter->AddConstantOperand(*scale_tensor, DDim(scale_bias_shape));
  } else {
    scale_operand = converter->AddConstantOperand(
        std::vector<float>(scale_bias_count, 1), DDim(scale_bias_shape));
  }
  // Epsilon operand
  auto epsilon = op->GetAttr<float>("epsilon");
  auto epsilon_operand = converter->AddConstantOperand(epsilon);
  // Output operand
  auto out_name = op->Output("Y").front();
  auto out_scale_name = "Y0_scale";
  std::vector<float> out_scales;
  if (op->HasOutputScale(out_scale_name, true)) {
    out_scales = op->GetOutputScale(out_scale_name, true);
  }
  auto output_operand = converter->AddOutputOperand(out_name, out_scales);
  // LayerNorm operand
  converter->AddOperation(NNADAPTER_LAYER_NORMALIZATION,
                          {input_operand,
                           scale_operand,
                           bias_operand,
                           begin_norm_axis_operand,
                           epsilon_operand},
                          {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
