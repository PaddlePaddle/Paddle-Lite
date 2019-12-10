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
#include "lite/kernels/npu/bridges/context.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int SoftmaxConverter(void* ctx, OpLite* op) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph_ctx = static_cast<Context*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  auto x_var_name = op_info->Input("X").front();
  auto out_var_name = op_info->Output("Out").front();
  auto x_dims = scope->FindVar(x_var_name)->GetMutable<Tensor>()->dims();
  auto axis = op_info->GetAttr<int>("axis");
  if (x_dims.size() > 3) {
    CHECK(!(axis == 2 && x_dims[3] > 1))
        << "[NPU] Unsupported softmax params: axis = " << axis
        << "  :x_w = " << x_dims[3];
  }

  auto softmax_node = graph_ctx->AddNode<ge::op::Softmax>(out_var_name);
  softmax_node->set_input_x(*graph_ctx->GetNode(x_var_name));
  softmax_node->set_attr_axis(axis);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(NPU,
                         softmax,
                         paddle::lite::subgraph::npu::SoftmaxConverter);
