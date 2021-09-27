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

int ConvertPRelu(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);

  // Slope operand
  NNAdapterOperand* slope_operand = nullptr;
  auto slope_name = op->Input("Alpha").front();
  auto slope_tensor = scope->FindTensor(slope_name);
  auto mode = op->GetAttr<std::string>("mode");
  if (mode == "all" || mode == "channel") {
    slope_operand = converter->AddConstantOperand(
        *slope_tensor, DDim({slope_tensor->numel()}));
  } else if (mode == "element") {
    slope_operand = converter->AddConstantOperand(*slope_tensor);
  } else {
    LOG(ERROR) << "Not support prelu mode: " << mode;
    return PARAMETER_ERROR;
  }

  // Output operand
  auto out_name = op->Output("Out").front();
  auto output_operand = converter->AddOutputOperand(out_name);

  // PRelu operation
  if (mode == "all" || mode == "channel") {
    converter->AddOperation(
        NNADAPTER_PRELU, {input_operand, slope_operand}, {output_operand});
  } else {
    // TODO(zhupengyang): support by max/mul/add/cast
    LOG(ERROR) << "Not support prelu mode: " << mode;
  }
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
