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

int LayerNormConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto padded_x_shape = CvtShape(x_dims);
  auto x_rank = static_cast<int>(x_dims.size());
  CHECK(x_rank >= 2 && x_rank <= 4);

  auto y_name = op_info->Output("Y").front();
  auto y = scope->FindMutableTensor(y_name);
  auto y_dims = y->dims();
  auto padded_y_shape = CvtShape(y_dims);

  auto epsilon = op_info->GetAttr<float>("epsilon");
  auto begin_norm_axis = op_info->GetAttr<int>("begin_norm_axis");
  if (begin_norm_axis < 0) {
    begin_norm_axis += x_rank;
  }
  CHECK(begin_norm_axis >= 1 && begin_norm_axis < x_rank);
  auto x_mat_dims = x_dims.Flatten2D(begin_norm_axis);
  auto left = x_mat_dims[0];
  auto right = x_mat_dims[1];

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x, padded_x_shape);
  }

  // Reshaped X node if needs
  bool reshape = false;
  if (!(x_rank == 4 && begin_norm_axis == 1)) {
    reshape = true;
    // Only the input shape 4-D(n, c, h, w) and axis=1 is supported
    // by HiAI DDK, So the input shape need to be padded to 4-D if it is less
    // than 4 or axis!=1. For example:
    // (1) (n, c, h, w), axis=1 -> no need
    // (2) (n, c, h, w), axis=2 -> (n * c, h, w, 1)
    // (3) (n, c, h, w), axis=3 -> (n * c * h, w, 1)
    // (4) (n, h, w), axis=1 -> (n, h, w, 1)
    // (5) (n, h, w), axis=2 -> (n * h, w, 1, 1)
    // (6) (h, w), axis=1 -> (h, w, 1, 1)
    padded_x_shape = {left};
    for (int i = begin_norm_axis; i < x_rank; i++) {
      padded_x_shape.push_back(x_dims[i]);
    }
    auto remain = 4 - padded_x_shape.size();
    for (int i = 0; i < remain; i++) {
      padded_x_shape.push_back(1);
    }
    auto reshaped_x_node = graph->Add<ge::op::Reshape>(
        x_name + "/reshape", x_node->precision(), x_node->layout());
    auto reshaped_x_op = reshaped_x_node->data<ge::op::Reshape>();
    reshaped_x_op->set_input_tensor(*x_node->data());
    reshaped_x_op->set_attr_shape(padded_x_shape);
    x_node = reshaped_x_node;
  }

  // Bias node
  auto scale_bias_dims =
      DDim({1, padded_x_shape[1], padded_x_shape[2], padded_x_shape[3]});
  std::shared_ptr<Node> bias_node = nullptr;
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_name = op_info->Input("Bias").front();
    auto bias = scope->FindMutableTensor(bias_name);
    auto bias_dims = bias->dims();
    CHECK_EQ(bias_dims.size(), 1);
    CHECK_EQ(bias_dims.production(), right);
    if (!bias->persistable()) {
      LOG(WARNING) << "[NPU] Only supporting persistable bias tensor.";
      return FAILED;
    }
    bias_node = graph->Add(bias_name, *bias, scale_bias_dims);
  } else {
    bias_node = graph->Add(y_name + "/bias", 0.0f, scale_bias_dims);
  }

  // Scale node
  std::shared_ptr<Node> scale_node = nullptr;
  if (HasInputArg(op_info, scope, "Scale")) {
    auto scale_name = op_info->Input("Scale").front();
    auto scale = scope->FindMutableTensor(scale_name);
    auto scale_dims = scale->dims();
    CHECK_EQ(scale_dims.size(), 1);
    CHECK_EQ(scale_dims.production(), right);
    if (!scale->persistable()) {
      LOG(WARNING) << "[NPU] Only supporting persistable scale tensor.";
      return FAILED;
    }
    scale_node = graph->Add(scale_name, *scale, scale_bias_dims);
  } else {
    scale_node = graph->Add(y_name + "/scale", 1.0f, scale_bias_dims);
  }

  // LayerNorm node
  auto layer_norm_node = graph->Add<hiai::op::LayerNorm>(y_name);
  auto layer_norm_op = layer_norm_node->data<hiai::op::LayerNorm>();
  layer_norm_op->set_input_x(*x_node->data());
  layer_norm_op->set_input_gamma(*scale_node->data());
  layer_norm_op->set_input_beta(*bias_node->data());
  layer_norm_op->set_attr_epsilon(epsilon);

  // Reshaped Y node if needs
  if (reshape) {
    auto reshaped_y_node = graph->Add<ge::op::Reshape>(
        y_name, layer_norm_node->precision(), layer_norm_node->layout());
    auto reshaped_y_op = reshaped_y_node->data<ge::op::Reshape>();
    reshaped_y_op->set_input_tensor(*layer_norm_node->data());
    reshaped_y_op->set_attr_shape(padded_y_shape);
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(layer_norm,
                         kNPU,
                         paddle::lite::subgraph::npu::LayerNormConverter);
