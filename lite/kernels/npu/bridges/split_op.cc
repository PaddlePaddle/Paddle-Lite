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

int SplitConverter(void* ctx, OpLite* op) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph_ctx = static_cast<Context*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " << op_type << " ... ";

  auto x_var_name = op_info->Input("X").front();
  auto out_var_names = op_info->Output("Out");
  auto axis = op_info->GetAttr<int>("axis");
  auto num = op_info->GetAttr<int>("num");
  auto sections = op_info->GetAttr<std::vector<int>>("sections");
  int64_t sections_num = static_cast<int64_t>(sections.size());

  auto split_node =
      graph_ctx->AddNode<ge::op::Split>(op_type + "/" + x_var_name);
  split_node->set_input_x(*graph_ctx->GetNode(x_var_name));
  split_node->set_attr_axis(static_cast<int64_t>(axis));
  if (num > 0) {
    split_node->set_attr_output_num(static_cast<int64_t>(num));
  } else {
    split_node->set_attr_output_num(sections_num);
    auto size_split = ge::AttrValue::LIST_INT(sections.begin(), sections.end());
    split_node->set_attr_size_split(size_split);
  }

  split_node->create_dynamic_output_y(out_var_names.size());
  int idx = 1;
  for (auto& out_var_name : out_var_names) {
    auto zero_const_node =
        graph_ctx->AddNode(out_var_name + "/zero" + std::to_string(idx), 0);
    auto add_node = graph_ctx->AddNode<ge::op::Add>(out_var_name);
    add_node->set_input_x1(*split_node, "y" + std::to_string(idx));
    add_node->set_input_x2(*zero_const_node);
    idx++;
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(NPU,
                         split,
                         paddle::lite::subgraph::npu::SplitConverter);
