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
#include "lite/kernels/npu/bridges/graph.h"
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
  auto x = scope->FindTensor(x_name);
  auto x_dims = x->dims();

  auto y_name = op_info->Input("Y").front();
  auto y = scope->FindTensor(y_name);
  auto y_dims = y->dims();

  auto out_name = op_info->Output("Out").front();
  auto out = scope->FindTensor(out_name);
  auto out_dims = out->dims();
  if (out_dims.size() > 4) {
    LOG(WARNING) << "[NPU] not supported above 4-D.";
    return FAILED;
  }

  int x_num_col_dims = op_info->GetAttr<int>("x_num_col_dims");
  int y_num_col_dims = op_info->GetAttr<int>("y_num_col_dims");
  int m = x_dims.Slice(0, x_num_col_dims).production();
  int k = x_dims.Slice(x_num_col_dims, x_dims.size()).production();
  CHECK_EQ(k, y_dims.Slice(0, y_num_col_dims).production())
      << "[NPU] columns of X must be equal with rows of Y";
  int n = y_dims.Slice(y_num_col_dims, y_dims.size()).production();
  VLOG(3) << "m:" << m << ",n:" << n << ",k:" << k;
  VLOG(3) << "x_name:" << x_name << ", is data: " << graph->Has(x_name);
  VLOG(3) << "y_name:" << y_name << ", is data: " << graph->Has(y_name);

  // X node which supports persistable and non-persistable tensor, and
  // reshape to (m, k)
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
    if (x_dims.size() != 2) {
      auto reshaped_x_node = graph->Add<ge::op::Reshape>(x_name + "/reshape");
      auto reshaped_x_op = reshaped_x_node->data<ge::op::Reshape>();
      reshaped_x_op->set_input_tensor(*x_node->data());
      reshaped_x_op->set_attr_shape({m, k});
      reshaped_x_op->set_attr_axis(0);
      x_node = reshaped_x_node;
    }
  } else {
    x_node = graph->Add(x_name, *x, {m, k});
  }

  // Y node which only supports persistable tensor, and reshape to
  // (k,n)
  std::shared_ptr<Node> y_node = nullptr;
  if (graph->Has(y_name)) {
    y_node = graph->Get(y_name);
    if (y_dims.size() != 2) {
      auto reshaped_y_node = graph->Add<ge::op::Reshape>(y_name + "/reshape");
      auto reshaped_y_op = reshaped_y_node->data<ge::op::Reshape>();
      reshaped_y_op->set_input_tensor(*y_node->data());
      reshaped_y_op->set_attr_shape({k, n});
      reshaped_y_op->set_attr_axis(0);
      y_node = reshaped_y_node;
    }
  } else {
    y_node = graph->Add(y_name, *y, {k, n});
  }

  // Matmul node
  auto mul_node = graph->Add<ge::op::MatMul>(out_name);
  auto mul_op = mul_node->data<ge::op::MatMul>();
  mul_op->set_input_x1(*x_node->data());
  mul_op->set_input_x2(*y_node->data());

  if (out_dims.size() != 2) {
    auto reshaped_out_node = graph->Add<ge::op::Reshape>(out_name);
    auto reshaped_out_op = reshaped_out_node->data<ge::op::Reshape>();
    reshaped_out_op->set_input_tensor(*mul_node->data());
    auto out_shape = out_dims.Vectorize();
    reshaped_out_op->set_attr_shape(
        ge::AttrValue::LIST_INT(out_shape.begin(), out_shape.end()));
    reshaped_out_op->set_attr_axis(0);
  }

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(mul, kNPU, paddle::lite::subgraph::npu::MulConverter);
