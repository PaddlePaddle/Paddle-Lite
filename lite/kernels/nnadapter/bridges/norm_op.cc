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

#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

int NormConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

  // Get input, output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_scale_name = "X0_scale";
  auto has_x_scale = op_info->HasInputScale(x_scale_name, true);
  auto x_scale =
      has_x_scale ? op_info->GetInputScale(x_scale_name, true)[0] : 0.f;
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();

  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto has_out_scale = op_info->HasOutputScale(out_scale_name, true);
  auto out_scale =
      has_out_scale ? op_info->GetOutputScale(out_scale_name, true)[0] : 0.f;
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();

  float porder;
  bool keepdim;
  if (op_type == "p_norm") {
    porder = op_info->GetAttr<float>("porder");
    keepdim = op_info->GetAttr<bool>("keepdim");
  } else {
    porder = 2;
    keepdim = true;
  }
  auto axis = op_info->GetAttr<int>("axis");
  auto epsilon = op_info->GetAttr<float>("epsilon");
  if (axis < 0) axis = x_dims.size() + axis;
  std::vector<int32_t> axis_data({axis});

  // Input operand
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

  // Output operand
  NNAdapterOperand* output_operand = nullptr;
  if (has_out_scale) {
    output_operand =
        converter->AddQuant8VariableOperand(out_dims, out_scale, out_name);
  } else {
    output_operand = converter->AddFloat32VariableOperand(out_dims, out_name);
  }

  // P, axis, epsilon and keepdim
  int p = 0;
  if (porder == INFINITY) {
    p = INT_MAX;
  } else if (porder == -INFINITY) {
    p = INT_MIN;
  } else {
    p = static_cast<int>(porder);
  }
  auto p_operand = converter->AddInt32ConstantOperand(p);
  auto axis_operand = converter->AddInt32ConstantOperand(
      &axis_data[0], DDim({static_cast<int64_t>(axis_data.size())}));
  auto epsilon_operand = converter->AddFloat32ConstantOperand(epsilon);
  auto keepdim_operand = converter->AddBool8ConstantOperand(keepdim);

  // Norm operation
  std::vector<NNAdapterOperand*> input_operands = {
      input_operand, axis_operand, p_operand, epsilon_operand, keepdim_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  converter->AddOperation(
      NNADAPTER_LP_NORMALIZATION, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(norm,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::NormConverter);
REGISTER_SUBGRAPH_BRIDGE(p_norm,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::NormConverter);
