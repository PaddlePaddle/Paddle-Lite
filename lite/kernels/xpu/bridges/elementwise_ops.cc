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

#include "lite/backends/xpu/builder.h"
#include "lite/kernels/xpu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {
namespace bridges {

node_map_type ElementwiseConverter(const std::shared_ptr<lite::OpLite> op,
                                   graph_ctx_type* graph_ctx,
                                   const node_map_type& input_nodes) {
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = lite::xpu::UniqueName(op_type);
  LOG(INFO) << "[XPU] Converting " + op_type + "...";

  // check context
  CHECK(graph_ctx != nullptr);
  CHECK(graph_ctx->builder != nullptr);
  CHECK(graph_ctx->params != nullptr);

  // get input, and attributes
  auto x_var_name = op_info->Input("X").front();
  auto y_var_name = op_info->Input("Y").front();
  auto axis = op_info->GetAttr<int>("axis");
  auto x_tensor = scope->FindMutableTensor(x_var_name);
  auto y_tensor = scope->FindMutableTensor(y_var_name);
  auto x_dims = x_tensor->dims();
  auto y_dims = y_tensor->dims();

  // create x and y node
  std::shared_ptr<xtcl::xExpr> x_node = nullptr;
  if (input_nodes.count(x_var_name)) {
    x_node = input_nodes.at(x_var_name);
  } else {
    x_node = std::make_shared<xtcl::xExpr>(graph_ctx->builder->CreateTensor(
        x_var_name, lite::xpu::CvtShape(x_dims), ::xtcl::Float(32)));
    auto x_const_tensor = lite::xpu::CvtTensor(x_tensor);
    graph_ctx->params->emplace(std::make_pair(x_var_name, *x_const_tensor));
  }

  std::shared_ptr<xtcl::xExpr> y_node = nullptr;
  if (input_nodes.count(y_var_name)) {
    y_node = input_nodes.at(y_var_name);
  } else {
    y_node = std::make_shared<xtcl::xExpr>(graph_ctx->builder->CreateTensor(
        y_var_name, lite::xpu::CvtShape(y_dims), ::xtcl::Float(32)));
    auto y_const_tensor = lite::xpu::CvtTensor(y_tensor);
    graph_ctx->params->emplace(std::make_pair(y_var_name, *y_const_tensor));
  }

  // create elementwise node and set input, attributes
  std::shared_ptr<xtcl::xExpr> elementwise_node = nullptr;
  if (y_dims.size() == 1) {
    elementwise_node = std::make_shared<xtcl::xExpr>(
        graph_ctx->builder->CreateBiasAdd(*x_node, axis, *y_node));
  } else if (x_dims.size() == y_dims.size()) {
    elementwise_node = std::make_shared<xtcl::xExpr>(
        graph_ctx->builder->CreateBinaryOp("add", *x_node, *y_node));
  } else {
    LOG(ERROR) << "XPU elementwise_add only support y of one dimension, or x "
                  "and y of the same dimension. But recieved x's dimension: "
               << x_dims << ", y's dimension: " << y_dims << ", axis: " << axis;
  }
  graph_ctx->builder->SetLayer(unique_op_type);

  // output converted nodes
  node_map_type output_nodes;
  output_nodes[op_info->Output("Out").front()] = elementwise_node;
  return output_nodes;
}

}  // namespace bridges
}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_XPU_BRIDGE(elementwise_add,
                    paddle::lite::kernels::xpu::bridges::ElementwiseConverter);
