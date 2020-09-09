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

int MulConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
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
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();
  auto x_num_col_dims = op_info->GetAttr<int>("x_num_col_dims");
  auto x_matrix_dims = x_dims.Flatten2D(x_num_col_dims);
  auto y_num_col_dims = op_info->GetAttr<int>("y_num_col_dims");
  auto y_matrix_dims = y_dims.Flatten2D(y_num_col_dims);
  CHECK_EQ(x_matrix_dims[1], y_matrix_dims[0]);

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }
  // Flatten X node
  if (x_dims.size() != 2) {
    x_node = graph->Add(
        x_name + "/reshape",
        graph->builder_.CreateReshape(
            *x_node->data(), {-1, static_cast<int>(x_matrix_dims[1])}));
  }

  // Y node
  std::shared_ptr<Node> y_node = nullptr;
  if (graph->Has(y_name)) {
    y_node = graph->Get(y_name);
  } else {
    y_node = graph->Add(y_name, *y);
  }
  // Flatten Y node
  if (y_dims.size() != 2) {
    y_node = graph->Add(
        y_name + "/reshape",
        graph->builder_.CreateReshape(
            *y_node->data(), {static_cast<int>(y_matrix_dims[0]), -1}));
  }

  // Reshape the matmul node with the inferred shape as the output node
  auto matmul_node = graph->Add(
      out_name,
      graph->builder_.CreateMatmul2D(*x_node->data(), *y_node->data(), false));
  if (out_dims.size() != 2) {
    graph->Add(out_name,
               graph->builder_.CreateReshape(
                   *matmul_node->data(), CvtShape<xtcl::Integer>(out_dims)));
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}  // namespace xpu

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(mul, kXPU, paddle::lite::subgraph::xpu::MulConverter);
