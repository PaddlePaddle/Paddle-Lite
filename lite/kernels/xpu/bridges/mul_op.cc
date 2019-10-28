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

node_map_type MulConverter(const std::shared_ptr<lite::OpLite> op,
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
  auto y_tensor = scope->FindMutableTensor(y_var_name);
  auto y_dims = y_tensor->dims();
  CHECK_EQ(y_dims.size(), 2) << "xpu now only support y_dims.size() == 2";

  auto x_num_col_dims = op_info->GetAttr<int>("x_num_col_dims");
  CHECK_EQ(x_num_col_dims, 1) << "xpu now only support x_num_col_dims == 1";
  auto y_num_col_dims = op_info->GetAttr<int>("x_num_col_dims");
  CHECK_EQ(y_num_col_dims, 1) << "xpu now only support y_num_col_dims == 1";

  // create x node
  std::shared_ptr<xtcl::xExpr> x_node = nullptr;
  x_node = std::make_shared<xtcl::xExpr>(
      graph_ctx->builder->CreateBatchFlatten(*input_nodes.at(x_var_name)));
  graph_ctx->builder->SetLayer(unique_op_type + "/X");

  // transpose y
  DDimLite y_dims_t(std::vector<int64_t>{1, 1});
  y_dims_t[0] = y_dims[1];
  y_dims_t[1] = y_dims[0];
  auto y_var_name_t = unique_op_type + "/Y";
  auto y_tensor_t = scope->NewTensor(y_var_name_t);
  y_tensor_t->Resize(y_dims_t);
  auto y_data_t = y_tensor_t->mutable_data<float>();
  auto y_data = y_tensor->mutable_data<float>();
  for (int i = 0; i < y_dims_t[0]; i++) {
    for (int j = 0; j < y_dims_t[1]; j++) {
      y_data_t[i * y_dims_t[1] + j] = y_data[j * y_dims_t[0] + i];
    }
  }

  // create y node
  std::shared_ptr<xtcl::xExpr> y_const_node = nullptr;
  y_const_node = std::make_shared<xtcl::xExpr>(graph_ctx->builder->CreateTensor(
      y_var_name_t, lite::xpu::CvtShape(y_dims_t), ::xtcl::Float(32)));
  auto y_const_tensor = lite::xpu::CvtTensor(y_tensor_t);
  graph_ctx->params->emplace(std::make_pair(y_var_name_t, *y_const_tensor));

  // create mul node and set params from op
  std::shared_ptr<xtcl::xExpr> mul_node = nullptr;
  mul_node = std::make_shared<xtcl::xExpr>(graph_ctx->builder->CreateDense(
      *x_node, *y_const_node, static_cast<int>(y_dims[1])));
  graph_ctx->builder->SetLayer(unique_op_type);

  // output converted nodes
  node_map_type output_nodes;
  output_nodes[op_info->Output("Out").front()] = mul_node;
  return output_nodes;
}

}  // namespace bridges
}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_XPU_BRIDGE(mul, paddle::lite::kernels::xpu::bridges::MulConverter);
