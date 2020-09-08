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
#include "lite/kernels/xpu/bridges/graph.h"
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

int GatherConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto index_name = op_info->Input("Index").front();
  auto index = scope->FindMutableTensor(index_name);
  auto index_dims = index->dims();
  CHECK(index_dims.size() == 1 ||
        (index_dims.size() == 2 && index_dims[1] == 1));
  auto out_name = op_info->Output("Out").front();
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();

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
  // Flatten index node
  if (index_dims.size() != 1) {
    index_node =
        graph->Add(index_name + "/reshape",
                   graph->builder_.CreateReshape(*index_node->data(), {-1}),
                   index_node->precision(),
                   index_node->layout());
  }

  // Reshape the gather node with the inferred shape as the output node
  auto gather_node =
      graph->Add(out_name,
                 graph->builder_.CreateGather(
                     *x_node->data(), *index_node->data(), /* axis= */ 0),
                 x_node->precision(),
                 x_node->layout());
  if (out_dims.size() != 2) {
    graph->Add(out_name,
               graph->builder_.CreateReshape(*gather_node->data(),
                                             CvtShape<xtcl::Integer>(out_dims)),
               gather_node->precision(),
               gather_node->layout());
  }
  return SUCCESS;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(gather,
                         kXPU,
                         paddle::lite::subgraph::xpu::GatherConverter);
