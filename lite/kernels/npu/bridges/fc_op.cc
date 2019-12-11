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

#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int FCConverter(void* ctx, OpLite* op) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  auto x_var_name = op_info->Input("Input").front();
  auto w_var_name = op_info->Input("W").front();
  auto out_var_name = op_info->Output("Out").front();

  int in_num_col_dims = op_info->GetAttr<int>("in_num_col_dims");
  auto x = scope->FindVar(x_var_name)->GetMutable<Tensor>();
  auto w = scope->FindVar(w_var_name)->GetMutable<Tensor>();
  auto x_dims = x->dims();
  auto w_dims = w->dims();

  CHECK_GE(x_dims.size(), 2UL);
  CHECK_EQ(w_dims.size(), 2UL);

  int m = x_dims.Slice(0, in_num_col_dims).production();
  int k = x_dims.Slice(in_num_col_dims, x_dims.size()).production();
  int n = w_dims[1];
  CHECK_EQ(k * n, w_dims.production());
  VLOG(3) << "[NPU] x dims: " << x_dims << " w dims: " << w_dims << " m: " << m
          << " k: " << k << " n: " << n;

  auto fc_node = graph->AddNode<ge::op::FullConnection>(out_var_name + "/fc");
  CHECK(!graph->HasNode(w_var_name));

  // Reshape x to (m, k, 1, 1)
  auto reshaped_x_node =
      graph->AddNode<ge::op::Reshape>(x_var_name + "/reshape");
  reshaped_x_node->set_input_tensor(*graph->GetNode(x_var_name));
  reshaped_x_node->set_attr_shape({m, k, 1, 1});
  reshaped_x_node->set_attr_axis(0);
  fc_node->set_input_x(*reshaped_x_node);

  // Create w const node, set its shape to (n, k, 1, 1) and fill with
  // the transposed w tensor
  Tensor transpose_w;
  transpose_w.Resize({n, k, 1, 1});
  auto transpose_w_data = transpose_w.mutable_data<float>();
  auto w_data = w->mutable_data<float>();
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      transpose_w_data[j * k + i] = w_data[i * n + j];
    }
  }
  auto w_const_node = graph->AddNode(w_var_name, transpose_w);
  fc_node->set_input_w(*w_const_node);

  // Add bias node if bias tensor exists
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_var_name = op_info->Input("Bias").front();
    auto bias = scope->FindVar(bias_var_name)->GetMutable<lite::Tensor>();
    auto bias_dims = bias->dims();
    CHECK(!graph->HasNode(bias_var_name));
    CHECK_EQ(bias_dims.production(), n);

    auto bias_const_node = graph->AddNode(bias_var_name, *bias, {1, n, 1, 1});
    fc_node->set_input_b(*bias_const_node);
  }

  // Reshape output of fc_node from (m, n, 1, 1) to (m, n)
  auto reshaped_fc_node = graph->AddNode<ge::op::Reshape>(out_var_name);
  reshaped_fc_node->set_input_tensor(*fc_node);
  reshaped_fc_node->set_attr_shape({m, n});
  reshaped_fc_node->set_attr_axis(0);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(NPU, fc, paddle::lite::subgraph::npu::FCConverter);
