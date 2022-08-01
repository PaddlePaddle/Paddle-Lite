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

int ConvertReduce(Converter* converter, OpInfo* op, Scope* scope) {
  // Extract op attributes
  // Input
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  // Axes
  bool reduce_all = false;
  if (op->HasAttr("reduce_all")) {
    reduce_all = op->GetAttr<bool>("reduce_all");
  }
  std::vector<int> dim = op->GetAttr<std::vector<int>>("dim");
  // Keep_dim
  bool keep_dim =
      op->HasAttr("keep_dim") ? op->GetAttr<bool>("keep_dim") : true;
  // Output
  auto output_name = op->Output("Out").front();
  auto output_scale_name = "Out0_scale";
  std::vector<float> output_scales;
  if (op->HasOutputScale(output_scale_name, true)) {
    output_scales = op->GetOutputScale(output_scale_name, true);
  }

  // Convert to NNAdapter operands and operation
  // Input operand
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);
  // Axes operand
  auto input_type = converter->GetOperandType(input_operand);
  if (reduce_all) {
    dim.clear();
    for (int i = 0; i < input_type->dimensions.count; i++) {
      dim.push_back(i);
    }
  }
  auto axes_operand = converter->AddConstantOperand(dim);
  // Keep_dim operand: keep_dim: default 1
  auto keep_dim_operand = converter->AddConstantOperand(keep_dim);
  // Output operand
  auto output_operand = converter->AddOutputOperand(output_name, output_scales);
  // Reduce operation
  NNAdapterOperationType reduce_operation_type;
  auto op_type = op->Type();
  if (op_type == "reduce_mean") {
    reduce_operation_type = NNADAPTER_REDUCE_MEAN;
  } else if (op_type == "reduce_max") {
    reduce_operation_type = NNADAPTER_REDUCE_MAX;
  } else if (op_type == "reduce_sum") {
    reduce_operation_type = NNADAPTER_REDUCE_SUM;
  } else {
    LOG(WARNING) << "Unsupported reduce operation type: " << op_type;
    return UNSUPPORTED_FEATURE;
  }
  converter->AddOperation(reduce_operation_type,
                          {input_operand, axes_operand, keep_dim_operand},
                          {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
