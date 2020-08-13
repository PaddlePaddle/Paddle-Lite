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

#include "lite/kernels/huawei_ascend_npu/bridges/graph.h"
#include "lite/kernels/huawei_ascend_npu/bridges/utility.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace huawei_ascend_npu {

int LookupTableConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

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
  std::vector<int64_t> index_new_shape{index->numel()};
  auto shape_node =
      graph->Add<int64_t>(index_name + "/index_shape", index_new_shape);
  auto reshaped_index_node =
      graph->Add<ge::op::Reshape>(index_name + "/reshape");
  auto reshaped_index_op = reshaped_index_node->data<ge::op::Reshape>();
  reshaped_index_op->set_input_x(*index_node->data());
  reshaped_index_op->set_input_shape(*shape_node->data());
  reshaped_index_op->set_attr_axis(0);
  INPUT_UPDATE(reshaped_index_op, x, index_node);
  INPUT_UPDATE(reshaped_index_op, shape, shape_node);
  OUTPUT_UPDATE(reshaped_index_op, y, reshaped_index_node);

  // Gather node
  auto gather_node = graph->Add<ge::op::Gather>(out_name + "/gather");
  auto gather_op = gather_node->data<ge::op::Gather>();
  gather_op->set_input_x(*w_node->data());
  gather_op->set_input_indices(*reshaped_index_node->data());
  INPUT_UPDATE(gather_op, x, w_node);
  INPUT_UPDATE(gather_op, indices, reshaped_index_node);
  OUTPUT_UPDATE(gather_op, y, gather_node);

  // reshape out
  auto out_shape_node = graph->Add<int64_t>(out_name + "/out_shape", out_shape);
  auto reshaped_gather_node = graph->Add<ge::op::Reshape>(out_name);
  auto reshaped_gather_op = reshaped_gather_node->data<ge::op::Reshape>();
  reshaped_gather_op->set_input_x(*gather_node->data());
  reshaped_gather_op->set_input_shape(*out_shape_node->data());
  reshaped_gather_op->set_attr_axis(0);
  INPUT_UPDATE(reshaped_gather_op, x, gather_node);
  INPUT_UPDATE(reshaped_gather_op, shape, out_shape_node);
  OUTPUT_UPDATE(reshaped_gather_op, y, reshaped_gather_node);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    lookup_table,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::LookupTableConverter);
