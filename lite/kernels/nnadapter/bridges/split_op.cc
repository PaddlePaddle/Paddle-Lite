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

int SplitConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_scale_name = "X0_scale";
  auto has_x_scale = op_info->HasInputScale(x_scale_name, true);
  auto x_scale =
      has_x_scale ? op_info->GetInputScale(x_scale_name, true)[0] : 0.f;
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto x_rank = x_dims.size();
  auto out_names = op_info->Output("Out");
  auto axis = op_info->GetAttr<int>("axis");
  if (axis < 0) {
    axis += x_rank;
  }
  auto num = op_info->GetAttr<int>("num");
  if (num > 0) {
    CHECK_EQ(num, out_names.size()) << "The attribute 'num' should be equal to "
                                       "the number of output tensors.";
  } else {
    num = out_names.size();
  }
  std::vector<int> sections = op_info->GetAttr<std::vector<int>>("sections");
  if (sections.size() > 0) {
    CHECK_EQ(sections.size(), num) << "The size of sections should be equal to "
                                      "the number of output tensors.";
  } else {
    auto axis_dim_size = x_dims[axis];
    CHECK_EQ(axis_dim_size % num, 0);
    for (uint32_t i = 0; i < num; i++) {
      sections.push_back(axis_dim_size / num);
    }
  }

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

  // Axis operand
  auto axis_operand = converter->AddInt32ConstantOperand(axis);

  // Split operand
  auto split_operand = converter->AddInt32ConstantOperand(
      &sections[0], DDim({static_cast<int64_t>(sections.size())}));

  // Output operands
  std::vector<NNAdapterOperand*> output_operands;
  for (size_t i = 0; i < sections.size(); i++) {
    auto out_name = out_names[i];
    auto out_scale_name = "Out" + paddle::lite::to_string(i) + "_scale";
    auto has_out_scale = op_info->HasOutputScale(out_scale_name, true);
    auto out_scale =
        has_out_scale ? op_info->GetOutputScale(out_scale_name, true)[0] : 0.f;
    auto out = scope->FindMutableTensor(out_name);
    auto out_dims = out->dims();
    NNAdapterOperand* output_operand = nullptr;
    if (has_out_scale) {
      output_operand =
          converter->AddQuant8VariableOperand(out_dims, out_scale, out_name);
    } else {
      output_operand = converter->AddFloat32VariableOperand(out_dims, out_name);
    }
    output_operands.push_back(output_operand);
  }

  // Split operation
  std::vector<NNAdapterOperand*> input_operands = {
      input_operand, axis_operand, split_operand};
  converter->AddOperation(NNADAPTER_SPLIT, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(split,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::SplitConverter);
