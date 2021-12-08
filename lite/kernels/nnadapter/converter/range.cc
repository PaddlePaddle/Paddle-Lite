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

int ConvertRange(Converter* converter, OpInfo* op, Scope* scope) {
  // Start operand
  auto start_name = op->Input("Start").front();
  auto start_scale_name = "Start0_scale";
  std::vector<float> start_scales;
  if (op->HasInputScale(start_scale_name, true)) {
    start_scales = op->GetInputScale(start_scale_name, true);
  }
  auto start_operand =
      converter->AddInputOperand(scope, start_name, {}, start_scales);

  // End operand
  auto end_name = op->Input("End").front();
  auto end_scale_name = "End0_scale";
  std::vector<float> end_scales;
  if (op->HasInputScale(end_scale_name, true)) {
    end_scales = op->GetInputScale(end_scale_name, true);
  }
  auto end_operand =
      converter->AddInputOperand(scope, end_name, {}, end_scales);

  // Step operand
  auto step_name = op->Input("Step").front();
  auto step_scale_name = "Step0_scale";
  std::vector<float> step_scales;
  if (op->HasInputScale(step_scale_name, true)) {
    step_scales = op->GetInputScale(step_scale_name, true);
  }
  auto step_operand =
      converter->AddInputOperand(scope, step_name, {}, step_scales);

  // Output operand
  auto out_name = op->Output("Out").front();
  auto output_operand = converter->AddOutputOperand(out_name);

  // Range operation
  converter->AddOperation(NNADAPTER_RANGE,
                          {start_operand, end_operand, step_operand},
                          {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
