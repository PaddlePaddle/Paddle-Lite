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

int LayerNormConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_type = kernel->GetInputDeclType("X");
  CHECK(x_type->precision() == PRECISION(kFloat));
  CHECK(x_type->layout() == DATALAYOUT(kNCHW));
  auto x = scope->FindMutableTensor(input_name);
  auto x_dims = x->dims();
  auto scale_name = op_info->Input("Scale").front();
  auto scale_type = kernel->GetInputDeclType("Scale");
  CHECK(scale_type->precision() == PRECISION(kFloat));
  CHECK(scale_type->layout() == DATALAYOUT(kNCHW));
  auto scale = scope->FindMutableTensor(scale_name);
  auto bias_name = op_info->Input("Bias").front();
  auto bias_type = kernel->GetInputDeclType("Bias");
  CHECK(bias_type->precision() == PRECISION(kFloat));
  CHECK(bias_type->layout() == DATALAYOUT(kNCHW));
  auto bias = scope->FindMutableTensor(bias_name);
  auto y_name = op_info->Output("Y").front();
  auto y_type = kernel->GetOutputDeclType("Y");
  CHECK(y_type->precision() == PRECISION(kFloat));
  CHECK(y_type->layout() == DATALAYOUT(kNCHW));
  auto epsilon = op_info->GetAttr<float>("epsilon");
  auto axis = op_info->GetAttr<int>("begin_norm_axis");

  // X node
  std::shared_ptr<xtcl::xExpr> x_node = nullptr;
  if (graph->HasNode(x_name)) {
    x_node = graph->GetNode(x_name);
  } else {
    x_node = graph->AddNode(x_name, x_dims);
  }

  // Scale and Bias node
  auto scale_const_node = graph->AddNode(scale_name, *scale);
  auto bias_const_node = graph->AddNode(bias_name, *bias);

  // Layer Norm node
  auto layer_norm_node = graph->builder_.CreateLayerNorm(
      *x_node, *scale_const_node, *bias_const_node, axis, epsilon, true, true);
  graph->AddNode(y_name, graph->builder_.GetField(layer_norm_node, 0));
  return SUCCESS;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(XPU,
                         layer_norm,
                         paddle::lite::subgraph::xpu::LayerNormConverter);
