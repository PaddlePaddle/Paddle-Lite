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
#include "lite/kernels/xpu/bridges/graph.h"
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

int ElementwiseConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(op != nullptr);
  CHECK(ctx != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto y_name = op_info->Input("Y").front();
  auto y = scope->FindMutableTensor(y_name);
  auto y_dims = y->dims();
  auto out_name = op_info->Output("Out").front();
  auto axis = op_info->GetAttr<int>("axis");

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // Y node
  std::shared_ptr<Node> y_node = nullptr;
  if (graph->Has(y_name)) {
    y_node = graph->Get(y_name);
  } else {
    y_node = graph->Add(y_name, *y);
  }

  // Elementwise node
  std::shared_ptr<Node> elt_node = nullptr;
  if (y_dims.size() == 1) {
    elt_node = graph->Add(
        out_name,
        graph->builder_.CreateBiasAdd(*x_node->data(), axis, *y_node->data()));
  } else if (x_dims.size() == y_dims.size()) {
    elt_node = graph->Add(out_name,
                          graph->builder_.CreateBinaryOp(
                              "add", *x_node->data(), *y_node->data()));
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

REGISTER_SUBGRAPH_BRIDGE(elementwise_add,
                         kXPU,
                         paddle::lite::subgraph::xpu::ElementwiseConverter);
