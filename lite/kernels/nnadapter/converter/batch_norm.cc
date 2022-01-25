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

int ConvertBatchNorm(Converter* converter, OpInfo* op, Scope* scope) {
  // Extract op attributes
  // Input
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  // Scale
  auto scale_name = op->Input("Scale").front();
  // Bias
  auto bias_name = op->Input("Bias").front();
  // Mean
  auto mean_name = op->Input("Mean").front();
  // Variance
  auto variance_name = op->Input("Variance").front();
  // Epsilon
  float epsilon = op->GetAttr<float>("epsilon");
  // Output
  auto output_name = op->Output("Y").front();
  auto output_scale_name = "Y0_scale";
  std::vector<float> output_scales;
  if (op->HasOutputScale(output_scale_name, true)) {
    output_scales = op->GetOutputScale(output_scale_name, true);
  }

  // Convert to NNAdapter operands and operation
  // Input operand
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);
  // Scale operand
  auto scale_operand = converter->AddInputOperand(scope, scale_name);
  // Bias operand
  auto bias_operand = converter->AddInputOperand(scope, bias_name);
  // Mean operand
  auto mean_operand = converter->AddInputOperand(scope, mean_name);
  // Variance operand
  auto variance_operand = converter->AddInputOperand(scope, variance_name);
  // epsilon operand
  auto epsilon_operand = converter->AddConstantOperand(epsilon);
  // output
  auto output_operand = converter->AddOutputOperand(output_name, output_scales);
  // BatchNorm operation
  converter->AddOperation(NNADAPTER_BATCH_NORMALIZATION,
                          {input_operand,
                           scale_operand,
                           bias_operand,
                           mean_operand,
                           variance_operand,
                           epsilon_operand},
                          {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
