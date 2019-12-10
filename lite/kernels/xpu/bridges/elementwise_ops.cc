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

int ElementwiseConverter(void* ctx, OpLite* op) {
  CHECK(op != nullptr);
  CHECK(ctx != nullptr);
  auto graph_ctx = static_cast<Context*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Get input, and attributes
  auto x_var_name = op_info->Input("X").front();
  auto y_var_name = op_info->Input("Y").front();
  auto out_var_name = op_info->Output("Out").front();
  auto axis = op_info->GetAttr<int>("axis");
  auto x = scope->FindMutableTensor(x_var_name);
  auto y = scope->FindMutableTensor(y_var_name);
  auto x_dims = x->dims();
  auto y_dims = y->dims();

  // Create x and y node
  std::shared_ptr<xtcl::xExpr> x_node = nullptr;
  if (graph_ctx->HasNode(x_var_name)) {
    x_node = graph_ctx->GetNode(x_var_name);
  } else {
    x_node = graph_ctx->AddNode(x_var_name, *x);
  }

  std::shared_ptr<xtcl::xExpr> y_node = nullptr;
  if (graph_ctx->HasNode(y_var_name)) {
    y_node = graph_ctx->GetNode(y_var_name);
  } else {
    y_node = graph_ctx->AddNode(y_var_name, *y);
  }

  // Create elementwise node and set input, attributes
  std::shared_ptr<xtcl::xExpr> elementwise_node = nullptr;
  if (y_dims.size() == 1) {
    elementwise_node = graph_ctx->AddNode(
        out_var_name,
        graph_ctx->builder_.CreateBiasAdd(*x_node, axis, *y_node));
  } else if (x_dims.size() == y_dims.size()) {
    elementwise_node = graph_ctx->AddNode(
        out_var_name,
        graph_ctx->builder_.CreateBinaryOp("add", *x_node, *y_node));
  } else {
    LOG(WARNING)
        << "[XPU] elementwise_add only support y of one dimension, or x "
           "and y of the same dimension. But recieved x's dimension: "
        << x_dims << ", y's dimension: " << y_dims << ", axis: " << axis;
    return FAILED;
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(XPU,
                         elementwise_add,
                         paddle::lite::subgraph::xpu::ElementwiseConverter);
