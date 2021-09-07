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

int ConvertUnsqueeze(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto x_name = op->Input("X").front();
  auto input_operand = converter->GetMappedOperand(x_name);

  // Axes operand
  NNAdapterOperand* axes_operand = nullptr;
  if (HasInput(op, scope, "AxesTensorList")) {
    LOG(WARNING) << "Not support AxesTensorList.";
    return UNSUPPORTED_FEATURE;
  } else if (HasInput(op, scope, "AxesTensor")) {
    LOG(WARNING) << "Not support AxesTensor.";
    return UNSUPPORTED_FEATURE;
  } else {
    std::vector<int> axes = op->GetAttr<std::vector<int>>("axes");
    axes_operand = converter->AddConstantOperand(axes);
  }

  // Output operand
  auto out_name = op->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  std::vector<float> out_scales;
  if (op->HasOutputScale(out_scale_name, true)) {
    out_scales = op->GetOutputScale(out_scale_name, true);
  }
  auto output_operand = converter->AddOutputOperand(out_name, out_scales);

  // Unsqueeze operation
  converter->AddOperation(
      NNADAPTER_UNSQUEEZE, {input_operand, axes_operand}, {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
