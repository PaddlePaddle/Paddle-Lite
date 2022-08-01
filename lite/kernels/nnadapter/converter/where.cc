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

int ConvertWhere(Converter* converter, OpInfo* op, Scope* scope) {
  // Condition operand
  auto condition_name = op->Input("Condition").front();
  auto condition_operand = converter->AddInputOperand(scope, condition_name);
  // Input0 operand
  auto x_name = op->Input("X").front();
  auto x_tensor = scope->FindTensor(x_name);
  auto x_persistable = x_tensor->persistable();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto input0_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);
  // Input1 operand
  auto y_name = op->Input("Y").front();
  auto y_tensor = scope->FindTensor(y_name);
  auto y_persistable = y_tensor->persistable();
  auto y_scale_name = "Y0_scale";
  std::vector<float> y_scales;
  if (op->HasInputScale(y_scale_name, true)) {
    y_scales = op->GetInputScale(y_scale_name, true);
  }
  auto input1_operand = converter->AddInputOperand(scope, y_name, {}, y_scales);

  uint32_t condition_rank =
      converter->GetOperandType(condition_operand)->dimensions.count;
  uint32_t x_rank = converter->GetOperandType(input0_operand)->dimensions.count;
  uint32_t y_rank = converter->GetOperandType(input1_operand)->dimensions.count;
  if (condition_rank != x_rank && condition_rank != y_rank) {
    LOG(FATAL) << "The rank of condition needs to be the same as that of x or "
                  "y, but the condition rank is "
               << condition_rank;
  }

  uint32_t max_rank = std::max(x_rank, y_rank);
  // Prepare unsqueeze axes
  std::vector<int32_t> axes;
  int32_t remain = max_rank - static_cast<int32_t>(std::min(x_rank, y_rank));
  for (int32_t i = 0; i < remain; i++) {
    axes.push_back(max_rank - remain + i);
  }
  // If persistable, set matcjed dims.
  // If not persistable, add unsqueeze to match shape.
  if (x_rank > y_rank) {
    if (y_persistable) {
      std::vector<int64_t> shape = y_tensor->dims().Vectorize();
      shape.insert(shape.end(), max_rank - shape.size(), 1);
      input1_operand = converter->AddConstantOperand(
          *y_tensor, DDim(shape), false, y_scales);
    } else {
      CHECK(!axes.empty());
      input1_operand = converter->AddUnsqueezeOperation(input1_operand, axes);
    }
  } else if (y_rank > x_rank) {
    if (x_persistable) {
      std::vector<int64_t> shape = x_tensor->dims().Vectorize();
      shape.insert(shape.end(), max_rank - shape.size(), 1);
      input0_operand = converter->AddConstantOperand(
          *x_tensor, DDim(shape), false, x_scales);
    } else {
      CHECK(!axes.empty());
      input0_operand = converter->AddUnsqueezeOperation(input0_operand, axes);
    }
  }

  // Output operand
  auto output_name = op->Output("Out").front();
  auto output_scale_name = "Out0_scale";
  std::vector<float> output_scales;
  if (op->HasOutputScale(output_scale_name, true)) {
    output_scales = op->GetOutputScale(output_scale_name, true);
  }
  auto output_operand = converter->AddOutputOperand(output_name, output_scales);
  // Where operation
  converter->AddOperation(NNADAPTER_WHERE,
                          {condition_operand, input0_operand, input1_operand},
                          {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
