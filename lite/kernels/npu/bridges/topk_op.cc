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

#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int TopkConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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

  auto out_name = op_info->Output("Out").front();

  int k = op_info->GetAttr<int>("k");

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // k node
  std::shared_ptr<Node> k_node = graph->Add<int>(out_name + "/k", k);

  // topk node
  auto topk_node = graph->Add<ge::op::TopK>(out_name);
  auto topk_op = topk_node->data<ge::op::TopK>();
  topk_op->set_input_x(*x_node->data());
  topk_op->set_input_k(*k_node->data());
  topk_op->set_attr_format(0);

  return SUCCESS;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(top_k,
                         kNPU,
                         paddle::lite::subgraph::npu::TopkConverter);
