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

int LayerNormConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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
  auto y_name = op_info->Output("Y").front();
  auto y = scope->FindMutableTensor(y_name);
  auto y_dims = y->dims();
  auto epsilon = op_info->GetAttr<float>("epsilon");
  auto axis = op_info->GetAttr<int>("begin_norm_axis");
  auto x_rank = static_cast<int>(x_dims.size());
  axis = axis < 0 ? (x_rank + axis) : axis;
  bool reshape = axis != (x_rank - 1);  // XPU only support the last dimension
  auto x_inner_size = x_dims.Slice(axis, x_rank).production();

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }
  if (reshape) {
    auto reshaped_x_dims = x_dims.Slice(0, axis).Vectorize();
    reshaped_x_dims.push_back(x_inner_size);
    x_node = graph->Add(
        x_name + "/reshape",
        graph->builder_.CreateReshape(
            *x_node->data(), CvtShape<xtcl::Integer>(reshaped_x_dims)));
  }

  // Scale node
  std::shared_ptr<Node> scale_node = nullptr;
  if (HasInputArg(op_info, scope, "Scale")) {
    auto scale_name = op_info->Input("Scale").front();
    auto scale = scope->FindMutableTensor(scale_name);
    auto scale_dims = scale->dims();
    CHECK_EQ(scale_dims.size(), 1);
    CHECK_EQ(scale_dims.production(), x_inner_size);
    scale_node = graph->Add(scale_name, *scale);
  } else {
    scale_node = graph->Add(y_name + "/scale_one", 1.0f, {x_inner_size});
  }

  // Bias node
  std::shared_ptr<Node> bias_node = nullptr;
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_name = op_info->Input("Bias").front();
    auto bias = scope->FindMutableTensor(bias_name);
    auto bias_dims = bias->dims();
    CHECK_EQ(bias_dims.size(), 1);
    CHECK_EQ(bias_dims.production(), x_inner_size);
    bias_node = graph->Add(bias_name, *bias);
  } else {
    bias_node = graph->Add(y_name + "/bias_zero", 0.0f, {x_inner_size});
  }

  // Layer Norm node
  auto layer_norm_node =
      graph->Add(y_name,
                 graph->builder_.CreateLayerNorm(*x_node->data(),
                                                 *scale_node->data(),
                                                 *bias_node->data(),
                                                 axis,
                                                 epsilon,
                                                 true,
                                                 true));
  if (reshape) {
    graph->Add(y_name,
               graph->builder_.CreateReshape(*layer_norm_node->data(),
                                             CvtShape<xtcl::Integer>(y_dims)));
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(layer_norm,
                         kXPU,
                         paddle::lite::subgraph::xpu::LayerNormConverter);
