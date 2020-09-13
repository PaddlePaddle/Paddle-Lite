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

int LookupTableConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto ids_name = op_info->Input("Ids").front();
  auto ids = scope->FindMutableTensor(ids_name);
  auto ids_dims = ids->dims();
  auto w_name = op_info->Input("W").front();
  auto w = scope->FindMutableTensor(w_name);
  auto w_dims = w->dims();
  CHECK_EQ(w_dims.size(), 2);
  auto out_name = op_info->Output("Out").front();
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();
  auto padding_idx = op_info->GetAttr<int64_t>("padding_idx");
  if (padding_idx != -1) {
    LOG(WARNING) << "[XPU] Only padding_idx=-1 is supported.";
    return FAILED;
  }

  // Ids node
  std::shared_ptr<Node> ids_node = nullptr;
  if (graph->Has(ids_name)) {
    ids_node = graph->Get(ids_name);
  } else {
    ids_node = graph->Add(ids_name, *ids);
  }
  // Flatten Ids node
  if (ids_dims.size() != 1) {
    ids_node =
        graph->Add(ids_name + "/reshape",
                   graph->builder_.CreateReshape(*ids_node->data(), {-1}),
                   ids_node->precision(),
                   ids_node->layout());
  }

  // W node
  auto w_node = graph->Add(w_name, *w);

  // Reshape the gather node with the inferred shape as the output node
  auto gather_node =
      graph->Add(out_name,
                 graph->builder_.CreateGather(
                     *w_node->data(), *ids_node->data(), /* axis= */ 0),
                 w_node->precision(),
                 w_node->layout());
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

REGISTER_SUBGRAPH_BRIDGE(lookup_table,
                         kXPU,
                         paddle::lite::subgraph::xpu::LookupTableConverter);
