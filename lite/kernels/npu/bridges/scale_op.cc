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

int ScaleConverter(void* ctx, OpLite* op) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Get input, output and op attributes
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = x->dims().Vectorize();
  CHECK_GE(x_dims.size(), 2);
  auto out_var_name = op_info->Output("Out").front();
  std::vector<int64_t> scale_bias_shape = {x_dims[1]};
  float scale = op_info->GetAttr<float>("scale");
  float bias = op_info->GetAttr<float>("bias");
  bool bias_after_scale = op_info->GetAttr<bool>("bias_after_scale");
  if (!bias_after_scale) {
    bias *= scale;
  }

  // Create scale node and set input node from inputs_map
  auto scale_node = graph->AddNode<ge::op::Scale>(out_var_name);
  scale_node->set_input_x(*graph->GetNode(x_var_name));

  // Add filter node(fill with scale)
  auto filter_const_node =
      graph->AddNode(out_var_name + "/filter", scale, scale_bias_shape);
  scale_node->set_input_filter(*filter_const_node);

  // Add bias node(fill with bias)
  if (fabs(bias) > 1e-6f) {
    auto bias_const_node =
        graph->AddNode(out_var_name + "/bias", bias, scale_bias_shape);
    scale_node->set_input_bias(*bias_const_node);
    scale_node->set_attr_has_bias_value(true);
  }
  scale_node->set_attr_axis(1);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(NPU,
                         scale,
                         paddle::lite::subgraph::npu::ScaleConverter);
