// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

int PowConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();

  auto out_name = op_info->Output("Out").front();
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();

  // Input0 operand
  NNAdapterOperand* input_operand_0 = nullptr;
  if (converter->HasOperand(x_name)) {
    input_operand_0 = converter->GetOperand(x_name);
  } else {
    input_operand_0 = converter->AddFloat32VariableOperand(x_dims, x_name);
  }

  // Input1 operand
  auto factor = op_info->GetAttr<float>("factor");
  NNAdapterOperand* input_operand_1 = nullptr;
  input_operand_1 = converter->AddFloat32ConstantOperand(
      &factor, DDim({static_cast<int64_t>(1)}));

  // Output operand
  NNAdapterOperand* output_operand = nullptr;
  output_operand = converter->AddFloat32VariableOperand(out_dims, out_name);

  // Activation operation
  std::vector<NNAdapterOperand*> input_operands = {input_operand_0,
                                                   input_operand_1};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  NNAdapterOperation* pow_operation = converter->AddOperation(NNADAPTER_POW);

  converter->SetOperation(pow_operation, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(pow,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::PowConverter);
