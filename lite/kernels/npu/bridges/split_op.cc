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

int SplitConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " << op_type << " ... ";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto out_names = op_info->Output("Out");
  auto axis = op_info->GetAttr<int>("axis");
  auto num = op_info->GetAttr<int>("num");
  auto sections = op_info->GetAttr<std::vector<int>>("sections");
  int64_t sections_num = static_cast<int64_t>(sections.size());

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // Split node
  auto split_node = graph->Add<ge::op::Split>(op_type + "/" + x_name);
  auto split_op = split_node->data<ge::op::Split>();
  split_op->set_input_x(*x_node->data());
  split_op->set_attr_axis(static_cast<int64_t>(axis));
  if (num > 0) {
    split_op->set_attr_output_num(static_cast<int64_t>(num));
  } else {
    split_op->set_attr_output_num(sections_num);
    auto size_split = ge::AttrValue::LIST_INT(sections.begin(), sections.end());
    split_op->set_attr_size_split(size_split);
  }

  split_op->create_dynamic_output_y(out_names.size());
  int idx = 1;
  for (auto& out_name : out_names) {
    auto zero_node =
        graph->Add(out_name + "/zero" + paddle::lite::to_string(idx), 0);
    auto add_node = graph->Add<ge::op::Add>(out_name);
    auto add_op = add_node->data<ge::op::Add>();
    add_op->set_input_x1(*split_node->data(),
                         "y" + paddle::lite::to_string(idx));
    add_op->set_input_x2(*zero_node->data());
    idx++;
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(split,
                         kNPU,
                         paddle::lite::subgraph::npu::SplitConverter);
