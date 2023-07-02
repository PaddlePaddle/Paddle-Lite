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

int ConvertLrn(Converter* converter, OpInfo* op, Scope* scope) {
  // Extract the inputs, outputs and attributes
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto out_name = op->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  std::vector<float> out_scales;
  if (op->HasOutputScale(out_scale_name, true)) {
    out_scales = op->GetOutputScale(out_scale_name, true);
  }
  int size = 5;
  if (op->HasAttr("n")) {
    size = op->GetAttr<int>("n");
  }
  float bias = 1.f;
  if (op->HasAttr("k")) {
    bias = op->GetAttr<float>("k");
  }
  float alpha = 0.0001f;
  if (op->HasAttr("alpha")) {
    alpha = op->GetAttr<float>("alpha");
  }
  float beta = 0.75f;
  if (op->HasAttr("beta")) {
    beta = op->GetAttr<float>("beta");
  }
  // Check quantization mode
  bool is_quant_mode = IsValidSymmPerLayerQuantParams(out_scales);
  if (is_quant_mode) {
    CHECK(IsValidSymmPerLayerQuantParams(x_scales))
        << "Missing the quant params '" << x_scale_name << "' for the input '"
        << x_name << "'";
  }
  // Convert to NNAdapter operands and operation
  // Input operand
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);
  CHECK(input_operand);
  auto input_type = converter->GetOperandType(input_operand);
  if (is_quant_mode) {
    CHECK(IsNNInt8SymmPerLayerQuantType(*input_type));
    std::vector<float> quant_scales;
    CHECK(GetNNSymmQuantParams(*input_type, &quant_scales));
    CHECK(IsSameSymmQuantParams(x_scales, quant_scales));
  }
  // Size operand
  auto size_operand = converter->AddConstantOperand(size);
  // Bias operand
  auto bias_operand = converter->AddConstantOperand(bias);
  // Alpha operand
  auto alpha_operand = converter->AddConstantOperand(alpha);
  // Beta operand
  auto beta_operand = converter->AddConstantOperand(beta);
  // Output operand
  auto output_operand = converter->AddOutputOperand(out_name, out_scales);
  // Lrn operation
  converter->AddOperation(
      NNADAPTER_LRN,
      {input_operand, size_operand, bias_operand, alpha_operand, beta_operand},
      {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
