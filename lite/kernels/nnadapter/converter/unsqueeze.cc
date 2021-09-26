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
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);
  // Axes
  std::vector<int32_t> axes;
  if (HasInput(op, scope, "AxesTensorList")) {
    LOG(WARNING) << "Not support AxesTensorList.";
    return UNSUPPORTED_FEATURE;
  } else if (HasInput(op, scope, "AxesTensor")) {
    LOG(WARNING) << "Not support AxesTensor.";
    return UNSUPPORTED_FEATURE;
  } else {
    axes = op->GetAttr<std::vector<int>>("axes");
  }
  // Output
  auto out_name = op->Output("Out").front();
  // Add unsqueeze operation
  converter->AddUnsqueezeOperation(input_operand, axes, out_name);
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
