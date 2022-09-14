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

int ConvertMatmulV2(Converter* converter, OpInfo* op, Scope* scope) {
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
    if (!IsValidSymmPerChannelQuantParams(y_scales)) {
      y_scales = {y_scales[0]};
    }
  }
  auto y_operand = converter->AddInputOperand(scope, y_name, {}, y_scales);

  // Transpose_x operand
  bool transpose_x = op->GetAttr<bool>("trans_x");
  auto transpose_x_operand = converter->AddConstantOperand(transpose_x);

  // Transpose_y operand
  bool transpose_y = op->GetAttr<bool>("trans_y");
  auto transpose_y_operand = converter->AddConstantOperand(transpose_y);

  // Output operand
  auto out_name = op->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  std::vector<float> out_scales;
  if (op->HasOutputScale(out_scale_name, true)) {
    out_scales = op->GetOutputScale(out_scale_name, true);
  }
  auto output_operand = converter->AddOutputOperand(out_name, out_scales);

  // Mat_mul operation
  converter->AddOperation(
      NNADAPTER_MAT_MUL,
      {x_operand, y_operand, transpose_x_operand, transpose_y_operand},
      {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
