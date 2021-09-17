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
  NNAdapterOperand* input_operand = nullptr;
  auto input_name = op->Input("W").front();
  auto input_tensor = scope->FindTensor(input_name);
  if (input_tensor->persistable()) {
    input_operand = converter->AddConstantOperand(*input_tensor);
  } else {
    input_operand = converter->GetMappedOperand(input_name);
  }

  // Indices operand
  NNAdapterOperand* indices_operand = nullptr;
  auto indices_name = op->Input("Ids").front();
  auto indices_tensor = scope->FindTensor(indices_name);
  if (indices_tensor->persistable()) {
    indices_operand = converter->AddConstantOperand(*indices_tensor);
  } else {
    indices_operand = converter->GetMappedOperand(indices_name);
  }

  // Axis operand
  auto axis_operand = converter->AddConstantOperand<int>(0);

  // Padding_idx
  if (op->HasAttr("padding_idx")) {
    // TODO(zhupengyang): support padding_idx later.
    CHECK_EQ(op->GetAttr<int64_t>("padding_idx"), -1L)
        << "Only support padding_idx = -1";
  }

  // Output operand
  auto out_name = op->Output("Out").front();
  auto output_operand = converter->AddOutputOperand(out_name);

  // Mat_mul operation
  converter->AddOperation(NNADAPTER_GATHER,
                          {input_operand, indices_operand, axis_operand},
                          {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
