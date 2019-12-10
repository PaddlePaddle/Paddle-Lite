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
#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int UnsqueezeConverter(void* ctx, OpLite* op) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " << op_type << "... ";

  auto x_var_name = op_info->Input("X").front();
  auto out_var_name = op_info->Output("Out").front();
  auto out_shape = scope->FindTensor(out_var_name)->dims().Vectorize();
  CHECK(op_info->HasAttr("axes"))
      << "[NPU] unsqueeze not support axes from tensor now";

  auto unsqueeze_node = graph->AddNode<ge::op::Reshape>(out_var_name);
  unsqueeze_node->set_input_tensor(*graph->GetNode(x_var_name));
  unsqueeze_node->set_attr_shape(
      ge::AttrValue::LIST_INT(out_shape.begin(), out_shape.end()));
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(NPU,
                         unsqueeze,
                         paddle::lite::subgraph::npu::UnsqueezeConverter);
REGISTER_SUBGRAPH_BRIDGE(NPU,
                         unsqueeze2,
                         paddle::lite::subgraph::npu::UnsqueezeConverter);
