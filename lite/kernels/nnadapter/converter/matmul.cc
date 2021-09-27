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

#include <cmath>
#include "lite/kernels/nnadapter/converter/converter.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

int ConvertMatmul(Converter* converter, OpInfo* op, Scope* scope) {
  // X operand
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto x_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);

  // Y operand
  auto y_name = op->Input("Y").front();
  auto y_scale_name = "Y0_scale";
  std::vector<float> y_scales;
  if (op->HasInputScale(y_scale_name, true)) {
    y_scales = op->GetInputScale(y_scale_name, true);
  }
  auto y_operand = converter->AddInputOperand(scope, y_name, {}, y_scales);

  // Transpose_x operand
  bool transpose_x = op->GetAttr<bool>("transpose_X");
  auto transpose_x_operand = converter->AddConstantOperand(transpose_x);

  // Transpose_y operand
  bool transpose_y = op->GetAttr<bool>("transpose_Y");
  auto transpose_y_operand = converter->AddConstantOperand(transpose_y);

  // Output operand
  auto out_name = op->Output("Out").front();
  auto output_operand = converter->AddOutputOperand(out_name);

  // Mat_mul operation
  converter->AddOperation(
      NNADAPTER_MAT_MUL,
      {x_operand, y_operand, transpose_x_operand, transpose_y_operand},
      {output_operand});

  // Use elementwise_mul to calculate alpha
  float alpha = op->GetAttr<float>("alpha");
  if (std::abs(alpha - 1.f) > 1e-5) {
    auto add_y_operand = converter->AddConstantOperand(alpha);
    auto add_fuse_code_operand = converter->AddConstantOperand(
        static_cast<int32_t>(NNADAPTER_FUSED_NONE));
    auto add_out_operand = converter->AddOutputOperand(out_name);
    converter->AddOperation(
        NNADAPTER_MUL,
        {output_operand, add_y_operand, add_fuse_code_operand},
        {add_out_operand});
  }
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
