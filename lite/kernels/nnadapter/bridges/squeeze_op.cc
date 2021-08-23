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

#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

int SqueezeConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

  // Input operand
  NNAdapterOperand* input_operand = nullptr;
  auto x_name = op_info->Input("X").front();
  auto x_scale_name = "X0_scale";
  auto has_x_scale = op_info->HasInputScale(x_scale_name, true);
  auto x_scale =
      has_x_scale ? op_info->GetInputScale(x_scale_name, true)[0] : 0.f;
  auto x = scope->FindTensor(x_name);
  auto x_dims = x->dims();
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

  // Axes operand
  NNAdapterOperand* axes_operand = nullptr;
  std::vector<int> axes;
  if (op_info->HasAttr("axes")) {
    axes = op_info->GetAttr<std::vector<int>>("axes");
  }
  if (!axes.empty()) {
    axes_operand = converter->AddInt32ConstantOperand(
        axes.data(), DDim({static_cast<int64_t>(axes.size())}));
  }

  // Output operand
  NNAdapterOperand* output_operand = nullptr;
  auto out_scale_name = "Out0_scale";
  auto has_out_scale = op_info->HasOutputScale(out_scale_name, true);
  auto out_scale =
      has_out_scale ? op_info->GetOutputScale(out_scale_name, true)[0] : 0.f;
  auto out_name = op_info->Output("Out").front();
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();
  if (has_out_scale) {
    output_operand =
        converter->AddQuant8VariableOperand(out_dims, out_scale, out_name);
  } else {
    output_operand = converter->AddFloat32VariableOperand(out_dims, out_name);
  }

  // Squeeze operation
  std::vector<NNAdapterOperand*> input_operands = {input_operand, axes_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  converter->AddOperation(NNADAPTER_SQUEEZE, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(squeeze,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::SqueezeConverter);
REGISTER_SUBGRAPH_BRIDGE(squeeze2,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::SqueezeConverter);
