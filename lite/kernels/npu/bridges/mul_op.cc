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
#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

// Note: all of the input weight vars should be handled in this converter
int MulConverter(void* ctx, OpLite* op) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  auto x_var_name = op_info->Input("X").front();
  auto y_var_name = op_info->Input("Y").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto y = scope->FindVar(y_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  auto out_var_name = op_info->Output("Out").front();
  int x_num_col_dims = op_info->GetAttr<int>("x_num_col_dims");
  int y_num_col_dims = op_info->GetAttr<int>("y_num_col_dims");
  int m = x_dims.Slice(0, x_num_col_dims).production();
  int k = x_dims.Slice(x_num_col_dims, x_dims.size()).production();
  CHECK_EQ(k, y_dims.Slice(0, y_num_col_dims).production())
      << "[NPU] columns of X must be equal with rows of Y";
  int n = y_dims.Slice(y_num_col_dims, y_dims.size()).production();
  VLOG(3) << "m:" << m << ",n:" << n << ",k:" << k;
  VLOG(3) << "x_var_name:" << x_var_name
          << ", is data: " << graph->HasNode(x_var_name);
  VLOG(3) << "y_var_name:" << y_var_name
          << ", is data: " << graph->HasNode(y_var_name);
  CHECK(graph->HasNode(x_var_name))
      << "[NPU] MatMul in HiAI DDK only support X is data, Y is const yet.";

  auto mul_node = graph->AddNode<ge::op::MatMul>(out_var_name);
  // Add input x node which supports persistable and non-persistable tensor, and
  // reshape to (m, k)
  if (graph->HasNode(x_var_name)) {
    auto reshaped_x_node =
        graph->AddNode<ge::op::Reshape>(x_var_name + "/reshape");
    reshaped_x_node->set_input_tensor(*graph->GetNode(x_var_name));
    reshaped_x_node->set_attr_shape({m, k});
    reshaped_x_node->set_attr_axis(0);
    mul_node->set_input_x1(*reshaped_x_node);
  } else {
    auto x_const_node = graph->AddNode(x_var_name, *x, {m, k});
    mul_node->set_input_x1(*x_const_node);
  }
  // Add input y node which only supports persistable tensor, and reshape to
  // (k,n)
  if (graph->HasNode(y_var_name)) {
    auto reshaped_y_node =
        graph->AddNode<ge::op::Reshape>(y_var_name + "/reshape");
    reshaped_y_node->set_input_tensor(*graph->GetNode(y_var_name));
    reshaped_y_node->set_attr_shape({k, n});
    reshaped_y_node->set_attr_axis(0);
    mul_node->set_input_x2(*reshaped_y_node);
  } else {
    auto y_const_node = graph->AddNode(y_var_name, *y, {k, n});
    mul_node->set_input_x2(*y_const_node);
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(NPU, mul, paddle::lite::subgraph::npu::MulConverter);
