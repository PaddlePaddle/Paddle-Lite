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

int ConcatConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

  // Get input and output vars and op attributes
  auto x_names = op_info->Input("X");
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto has_out_scale = op_info->HasOutputScale(out_scale_name, true);
  auto out_scale =
      has_out_scale ? op_info->GetOutputScale(out_scale_name, true)[0] : 0.f;
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();
  auto out_rank = out_dims.size();
  auto axis = op_info->GetAttr<int>("axis");
  if (axis < 0) {
    axis += out_rank;
  }

  // Input operands
  int num = x_names.size();
  std::vector<NNAdapterOperand*> input_operands;
  for (int i = 0; i < num; i++) {
    auto x_name = x_names[i];
    auto x_scale_name = "X" + paddle::lite::to_string(i) + "_scale";
    auto has_x_scale = op_info->HasInputScale(x_scale_name, true);
    auto x_scale =
        has_x_scale ? op_info->GetInputScale(x_scale_name, true)[0] : 0.f;
    auto x = scope->FindMutableTensor(x_name);
    auto x_dims = x->dims();
    NNAdapterOperand* input_operand = nullptr;
    if (converter->HasOperand(x_name)) {
      input_operand = converter->GetOperand(x_name);
    } else {
      if (has_x_scale) {
        input_operand =
            converter->AddQuant8VariableOperand(x_dims, x_scale, x_name);
      } else {
        input_operand = converter->AddFloat32VariableOperand(x_dims, x_name);
      }
    }
    input_operands.push_back(input_operand);
  }

  // Axis operand
  auto axis_operand = converter->AddInt32ConstantOperand(axis);
  input_operands.push_back(axis_operand);

  // Output operand
  NNAdapterOperand* output_operand = nullptr;
  if (has_out_scale) {
    output_operand =
        converter->AddQuant8VariableOperand(out_dims, out_scale, out_name);
  } else {
    output_operand = converter->AddFloat32VariableOperand(out_dims, out_name);
  }

  // Concat operation
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  auto concat_operation = converter->AddOperation(NNADAPTER_CONCAT);
  converter->SetOperation(concat_operation, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(concat,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::ConcatConverter);
