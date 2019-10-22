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

node_map_type ActConverter(const std::shared_ptr<lite::OpLite> op,
                           graph_ctx_type* graph_ctx,
                           const node_map_type& input_nodes) {
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = lite::xpu::UniqueName(op_type);
  LOG(INFO) << "[XPU] Converting " + op_type + "...";

  // check context
  CHECK(graph_ctx != nullptr);
  CHECK(graph_ctx->builder != nullptr);
  CHECK(graph_ctx->params != nullptr);

  // create act node and set params from op
  auto x_var_name = op_info->Input("X").front();
  CHECK(input_nodes.count(x_var_name));
  std::shared_ptr<xtcl::xExpr> act_node = nullptr;
  if (op_type == "relu") {
    act_node = std::make_shared<xtcl::xExpr>(
        graph_ctx->builder->CreateRelu(*input_nodes.at(x_var_name)));
  } else {
    // TODO(hong19860320) supports more activation ops
    LOG(FATAL) << "[XPU] Unsupported activation type " << op_type;
  }
  graph_ctx->builder->SetLayer(unique_op_type);

  // output converted nodes
  node_map_type output_nodes;
  output_nodes[op_info->Output("Out").front()] = act_node;
  return output_nodes;
}

}  // namespace bridges
}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_XPU_BRIDGE(relu, paddle::lite::kernels::xpu::bridges::ActConverter);
