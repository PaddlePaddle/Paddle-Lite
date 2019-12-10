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

#include "lite/core/mir/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/xpu/bridges/context.h"
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

int BatchNormConverter(void* ctx, OpLite* op) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph_ctx = static_cast<Context*>(ctx);
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
  auto mean_var_name = op_info->Input("Mean").front();
  auto* mean = scope->FindMutableTensor(mean_var_name);
  auto variance_var_name = op_info->Input("Variance").front();
  auto* variance = scope->FindMutableTensor(variance_var_name);
  auto y_var_name = op_info->Output("Y").front();
  auto epsilon = op_info->GetAttr<float>("epsilon");

  // Create scale, bias, mean, variance nodes
  auto scale_const_node = graph_ctx->AddNode(scale_var_name, *scale);
  auto bias_const_node = graph_ctx->AddNode(bias_var_name, *bias);
  auto mean_const_node = graph_ctx->AddNode(mean_var_name, *mean);
  auto variance_const_node = graph_ctx->AddNode(variance_var_name, *variance);

  // Create batch_norm node and set params from op
  auto batch_norm_node =
      graph_ctx->builder_.CreateBatchNorm(*graph_ctx->GetNode(x_var_name),
                                          *scale_const_node,
                                          *bias_const_node,
                                          *mean_const_node,
                                          *variance_const_node,
                                          1,
                                          epsilon);
  graph_ctx->AddNode(y_var_name,
                     graph_ctx->builder_.GetField(batch_norm_node, 0));
  return SUCCESS;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(XPU,
                         batch_norm,
                         paddle::lite::subgraph::xpu::BatchNormConverter);
