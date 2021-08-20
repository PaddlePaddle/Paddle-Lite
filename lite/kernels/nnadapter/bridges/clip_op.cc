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

int ClipConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

  // Input
  auto x_name = op_info->Input("X").front();
  auto x_scale_name = "X0_scale";
  auto has_x_scale = op_info->HasInputScale(x_scale_name, true);
  auto x_scale =
      has_x_scale ? op_info->GetInputScale(x_scale_name, true)[0] : 0.f;
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  // Output
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto has_out_scale = op_info->HasOutputScale(out_scale_name, true);
  auto out_scale =
      has_out_scale ? op_info->GetOutputScale(out_scale_name, true)[0] : 0.f;
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();

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
  // Min operand
  NNAdapterOperand* min_operand = nullptr;
  if (op_info->HasInput("Min") && op_info->Input("Min").size() > 0) {
    auto min_name = op_info->Input("Min").front();
    auto min_tensor = scope->FindMutableTensor(min_name);
    if (converter->HasOperand(min_name)) {
      min_operand = converter->GetOperand(min_name);
    } else {
      min_operand = converter->AddOperand(min_tensor, min_name);
    }
  } else {
    float min_value =
        op_info->HasAttr("min") ? op_info->GetAttr<float>("min") : 0.0f;
    min_operand = converter->AddFloat32ConstantOperand(
        &min_value, DDim({static_cast<int64_t>(1)}));
  }
  // Max operand
  NNAdapterOperand* max_operand = nullptr;
  if (op_info->HasInput("Max") && op_info->Input("Max").size() > 0) {
    auto max_name = op_info->Input("Max").front();
    auto max_tensor = scope->FindMutableTensor(max_name);
    if (converter->HasOperand(max_name)) {
      max_operand = converter->GetOperand(max_name);
    } else {
      max_operand = converter->AddOperand(max_tensor, max_name);
    }
  } else {
    float max_value =
        op_info->HasAttr("max") ? op_info->GetAttr<float>("max") : 0.0f;
    max_operand = converter->AddFloat32ConstantOperand(
        &max_value, DDim({static_cast<int64_t>(1)}));
  }
  // Output operand
  NNAdapterOperand* output_operand = nullptr;
  if (has_out_scale) {
    output_operand =
        converter->AddQuant8VariableOperand(out_dims, out_scale, out_name);
  } else {
    output_operand = converter->AddFloat32VariableOperand(out_dims, out_name);
  }

  // Clip operation
  std::vector<NNAdapterOperand*> input_operands = {
      input_operand, min_operand, max_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  converter->AddOperation(NNADAPTER_CLIP, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(clip,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::ClipConverter);
