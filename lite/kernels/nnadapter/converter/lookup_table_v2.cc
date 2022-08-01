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

int ConvertLookupTableV2(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto input_name = op->Input("W").front();
  auto input_scale_name = "W0_scale";
  std::vector<float> input_scales;
  if (op->HasInputScale(input_scale_name, true)) {
    input_scales = op->GetInputScale(input_scale_name, true);
  }
  auto input_operand =
      converter->AddInputOperand(scope, input_name, {}, input_scales);

  // Indices operand
  auto indices_name = op->Input("Ids").front();
  auto indices_scale_name = "Ids0_scale";
  std::vector<float> indices_scales;
  if (op->HasInputScale(indices_scale_name, true)) {
    indices_scales = op->GetInputScale(indices_scale_name, true);
  }
  auto indices_operand =
      converter->AddInputOperand(scope, indices_name, {}, indices_scales);

  // Axis operand
  auto axis_operand = converter->AddConstantOperand<int>(0);

  // Padding_idx
  if (op->HasAttr("padding_idx")) {
    auto padding_idx = op->GetAttr<int64_t>("padding_idx");
    // TODO(zhupengyang): support padding_idx later.
    if (padding_idx != -1 && padding_idx != 0) {
      LOG(FATAL) << "Only support padding_idx = -1 or 0";
    }
  }

  // Output operand
  auto out_name = op->Output("Out").front();
  auto output_operand = converter->AddOutputOperand(out_name);

  // Gather operation
  converter->AddOperation(NNADAPTER_GATHER,
                          {input_operand, indices_operand, axis_operand},
                          {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
