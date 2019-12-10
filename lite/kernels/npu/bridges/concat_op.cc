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
#include "lite/kernels/npu/bridges/context.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int ConcatConverter(void* ctx, OpLite* op) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph_ctx = static_cast<Context*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " << op_type << " ... ";

  auto x_var_names = op_info->Input("X");
  auto out_var_name = op_info->Output("Out").front();
  auto axis = op_info->GetAttr<int>("axis");
  auto num = x_var_names.size();
  auto concat_node = graph_ctx->AddNode<ge::op::Concat>(out_var_name);
  concat_node->set_attr_axis(axis);
  concat_node->set_attr_N(num);
  concat_node->create_dynamic_input_x(num);
  int idx = 1;
  for (auto& x_var_name : x_var_names) {
    if (graph_ctx->HasNode(x_var_name)) {
      concat_node->set_dynamic_input_x(idx, *graph_ctx->GetNode(x_var_name));
    } else {
      auto x = scope->FindVar(x_var_name)->GetMutable<Tensor>();
      auto x_const_node = graph_ctx->AddNode(x_var_name, *x);
      concat_node->set_dynamic_input_x(idx, *x_const_node);
    }
    idx++;
  }
  return SUCCESS;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(NPU,
                         concat,
                         paddle::lite::subgraph::npu::ConcatConverter);
