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

#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/xpu/bridges/graph.h"
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

int LayerNormConverter(void* ctx, OpLite* op) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Get input vars and op attributes
  auto x_var_name = op_info->Input("X").front();

  auto scale_var_name = op_info->Input("Scale").front();
  auto* scale = scope->FindMutableTensor(scale_var_name);
  auto bias_var_name = op_info->Input("Bias").front();
  auto* bias = scope->FindMutableTensor(bias_var_name);

  auto y_var_name = op_info->Output("Y").front();
  auto epsilon = op_info->GetAttr<float>("epsilon");
  auto axis = op_info->GetAttr<int>("begin_norm_axis");

  // Create scale, bias nodes
  auto scale_const_node = graph->AddNode(scale_var_name, *scale);
  auto bias_const_node = graph->AddNode(bias_var_name, *bias);

  // Create node and set params from op
  auto layer_norm_node =
      graph->builder_.CreateLayerNorm(*graph->GetNode(x_var_name),
                                      *scale_const_node,
                                      *bias_const_node,
                                      axis,
                                      epsilon,
                                      true,
                                      true);
  graph->AddNode(y_var_name, graph->builder_.GetField(layer_norm_node, 0));
  return SUCCESS;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(XPU,
                         layer_norm,
                         paddle::lite::subgraph::xpu::LayerNormConverter);
