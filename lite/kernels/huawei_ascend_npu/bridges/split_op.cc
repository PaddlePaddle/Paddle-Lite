// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/kernels/huawei_ascend_npu/bridges/graph.h"
#include "lite/kernels/huawei_ascend_npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace huawei_ascend_npu {

int SplitConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  // Get op attributes
  auto axis = op_info->GetAttr<int>("axis");
  int32_t num = op_info->GetAttr<int>("num");
  auto sections = op_info->GetAttr<std::vector<int>>("sections");
  int32_t sections_num = static_cast<int32_t>(sections.size());

  // X node
  auto x_name = op_info->Input("X").front();
  auto x_tensor = scope->FindMutableTensor(x_name);
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x_tensor);
  }

  // Split node
  auto out_names = op_info->Output("Out");
  std::shared_ptr<Node> split_node = nullptr;
  auto split_dim_node = graph->Add<int32_t>(x_name + "/split_dim", axis);
  if (num > 0) {
    auto split_node = graph->Add<ge::op::Split>(op_type + "/" + x_name);
    auto split_op = split_node->data<ge::op::Split>();
    split_op->set_input_x(*x_node->data());
    split_op->set_input_split_dim(*split_dim_node->data());
    split_op->set_attr_num_split(num);
    split_op->create_dynamic_output_y(num);
  } else {
    auto split_node = graph->Add<ge::op::SplitV>(op_type + "/" + x_name);
    auto split_op = split_node->data<ge::op::SplitV>();
    auto size_splits_node =
        graph->Add<int32_t>(x_name + "/size_splits", sections);
    split_op->set_input_x(*x_node->data());
    split_op->set_input_size_splits(*size_splits_node->data());
    split_op->set_attr_num_split(sections_num);
    split_op->create_dynamic_output_y(sections_num);
  }

  int idx = 1;
  for (auto& out_name : out_names) {
    auto zero_node =
        graph->Add(out_name + "/zero" + paddle::lite::to_string(idx), 0);
    auto add_node = graph->Add<ge::op::Add>(out_name);
    auto add_op = add_node->data<ge::op::Add>();
    add_op->set_input_x1_by_name(*split_node->data(),
                                 ("y" + paddle::lite::to_string(idx)).c_str());
    add_op->set_input_x2(*zero_node->data());
    idx++;
  }

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    split,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::SplitConverter);
