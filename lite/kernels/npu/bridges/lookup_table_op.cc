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

int LookupTableConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Get input, output and op attributes
  auto w_name = op_info->Input("W").front();
  auto w = scope->FindTensor(w_name);

  auto index_name = op_info->Input("Ids").front();
  auto index = scope->FindTensor(index_name);

  auto out_name = op_info->Output("Out").front();
  auto out = scope->FindTensor(out_name);
  auto out_shape = out->dims().Vectorize();

  // W node
  std::shared_ptr<Node> w_node = nullptr;
  if (graph->Has(w_name)) {
    w_node = graph->Get(w_name);
  } else {
    w_node = graph->Add(w_name, *w);
  }

  // Index node
  std::shared_ptr<Node> index_node = nullptr;
  if (graph->Has(index_name)) {
    index_node = graph->Get(index_name);
  } else {
    index_node = graph->Add(index_name, *index);
  }

  // reshape ids
  auto reshaped_index_node =
      graph->Add<ge::op::Reshape>(index_name + "/reshape");
  auto reshaped_index_op = reshaped_index_node->data<ge::op::Reshape>();
  reshaped_index_op->set_input_tensor(*index_node->data());
  reshaped_index_op->set_attr_shape(ge::AttrValue::LIST_INT({index->numel()}));
  reshaped_index_op->set_attr_axis(0);
  index_node = reshaped_index_node;

  // Gather node
  auto gather_node = graph->Add<ge::op::Gather>(out_name);
  auto gather_op = gather_node->data<ge::op::Gather>();
  gather_op->set_input_params(*w_node->data());
  gather_op->set_input_indices(*index_node->data());

  // reshape out
  auto reshaped_gather_node = graph->Add<ge::op::Reshape>(out_name);
  auto reshaped_gather_op = reshaped_gather_node->data<ge::op::Reshape>();
  reshaped_gather_op->set_input_tensor(*gather_node->data());
  reshaped_gather_op->set_attr_shape(
      ge::AttrValue::LIST_INT(out_shape.begin(), out_shape.end()));
  reshaped_gather_op->set_attr_axis(0);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(lookup_table,
                         kNPU,
                         paddle::lite::subgraph::npu::LookupTableConverter);
