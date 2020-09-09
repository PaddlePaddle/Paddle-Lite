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

int GatherConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Get input, output and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindTensor(x_name);

  auto index_name = op_info->Input("Index").front();
  auto index = scope->FindTensor(index_name);
  auto index_dims = index->dims();
  CHECK(index_dims.size() == 1 ||
        (index_dims.size() == 2 && index_dims[1] == 1))
      << "index dims unmatch";

  auto out_name = op_info->Output("Out").front();

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // Index node
  std::shared_ptr<Node> index_node = nullptr;
  if (graph->Has(index_name)) {
    index_node = graph->Get(index_name);
  } else {
    index_node = graph->Add(index_name, *index);
  }

  // Gather node
  auto gather_node = graph->Add<ge::op::Gather>(out_name);
  auto gather_op = gather_node->data<ge::op::Gather>();
  gather_op->set_input_params(*x_node->data());
  gather_op->set_input_indices(*index_node->data());

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(gather,
                         kNPU,
                         paddle::lite::subgraph::npu::GatherConverter);
