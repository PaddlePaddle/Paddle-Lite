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

// Note: all of the input weight vars should be handled in this converter
int MulConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_type = kernel->GetInputDeclType("X");
  CHECK(x_type->precision() == PRECISION(kFloat));
  CHECK(x_type->layout() == DATALAYOUT(kNCHW));
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto y_name = op_info->Input("Y").front();
  auto y_type = kernel->GetInputDeclType("Y");
  CHECK(y_type->precision() == PRECISION(kFloat));
  CHECK(y_type->layout() == DATALAYOUT(kNCHW));
  auto y = scope->FindMutableTensor(y_name);
  auto y_dims = y->dims();
  auto out_name = op_info->Output("Out").front();
  auto out_type = kernel->GetOutputDeclType("Out");
  CHECK(out_type->precision() == PRECISION(kFloat));
  CHECK(out_type->layout() == DATALAYOUT(kNCHW));
  int x_num_col_dims = op_info->GetAttr<int>("x_num_col_dims");
  int y_num_col_dims = op_info->GetAttr<int>("y_num_col_dims");
  int m = x_dims.Slice(0, x_num_col_dims).production();
  int k = x_dims.Slice(x_num_col_dims, x_dims.size()).production();
  CHECK_EQ(k, y_dims.Slice(0, y_num_col_dims).production())
      << "[NPU] columns of X must be equal with rows of Y";
  int n = y_dims.Slice(y_num_col_dims, y_dims.size()).production();
  VLOG(3) << "m:" << m << ",n:" << n << ",k:" << k;
  VLOG(3) << "x_name:" << x_name << ", is data: " << graph->HasNode(x_name);
  VLOG(3) << "y_name:" << y_name << ", is data: " << graph->HasNode(y_name);
  CHECK(graph->HasNode(x_name))
      << "[NPU] MatMul in HiAI DDK only support X is data, Y is const yet.";

  // X node which supports persistable and non-persistable tensor, and
  // reshape to (m, k)
  std::shared_ptr<ge::Operator> x_node = nullptr;
  if (graph->HasNode(x_name)) {
    x_node = graph->GetNode(x_name);
    auto reshaped_x_node = graph->AddNode<ge::op::Reshape>(x_name + "/reshape");
    reshaped_x_node->set_input_tensor(*x_node);
    reshaped_x_node->set_attr_shape({m, k});
    reshaped_x_node->set_attr_axis(0);
    x_node = reshaped_x_node;
  } else {
    auto x_const_node = graph->AddNode(x_name, *x, {m, k});
    x_node = x_const_node;
  }

  // Y node which only supports persistable tensor, and reshape to
  // (k,n)
  std::shared_ptr<ge::Operator> y_node = nullptr;
  if (graph->HasNode(y_name)) {
    y_node = graph->GetNode(y_name);
    auto reshaped_y_node = graph->AddNode<ge::op::Reshape>(y_name + "/reshape");
    reshaped_y_node->set_input_tensor(*y_node);
    reshaped_y_node->set_attr_shape({k, n});
    reshaped_y_node->set_attr_axis(0);
    y_node = reshaped_y_node;
  } else {
    auto y_const_node = graph->AddNode(y_name, *y, {k, n});
    y_node = y_const_node;
  }

  // Matmul node
  auto mul_node = graph->AddNode<ge::op::MatMul>(out_name);
  mul_node->set_input_x1(*x_node);
  mul_node->set_input_x2(*y_node);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(NPU, mul, paddle::lite::subgraph::npu::MulConverter);
