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

#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int UnsqueezeConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " << op_type << "... ";

  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();

  auto out_name = op_info->Output("Out").front();
  auto out_shape = scope->FindTensor(out_name)->dims().Vectorize();
  CHECK(op_info->HasAttr("axes"))
      << "[NPU] unsqueeze not support axes from tensor now";

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // Unsqueeze node
  auto unsqueeze_node = graph->Add<ge::op::Reshape>(out_name);
  auto unsqueeze_op = unsqueeze_node->data<ge::op::Reshape>();
  unsqueeze_op->set_input_tensor(*x_node->data());
  unsqueeze_op->set_attr_shape(
      ge::AttrValue::LIST_INT(out_shape.begin(), out_shape.end()));
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(unsqueeze,
                         kNPU,
                         paddle::lite::subgraph::npu::UnsqueezeConverter);
REGISTER_SUBGRAPH_BRIDGE(unsqueeze2,
                         kNPU,
                         paddle::lite::subgraph::npu::UnsqueezeConverter);
