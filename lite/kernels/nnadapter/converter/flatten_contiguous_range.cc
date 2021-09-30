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

int ConvertFlattenContiguousRange(Converter* converter,
                                  OpInfo* op,
                                  Scope* scope) {
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto out_name = op->Output("Out").front();
  auto start_axis = op->GetAttr<int>("start_axis");
  auto end_axis = op->GetAttr<int>("stop_axis");

  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);
  NNAdapterOperand* output_operand = converter->AddOutputOperand(out_name);
  auto start_axis_operand =
      converter->AddConstantOperand(static_cast<int32_t>(start_axis));
  auto end_axis_operand =
      converter->AddConstantOperand(static_cast<int32_t>(end_axis));
  converter->AddOperation(NNADAPTER_FLATTEN,
                          {input_operand, start_axis_operand, end_axis_operand},
                          {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
