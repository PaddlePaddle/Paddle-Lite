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

#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/huawei_ascend_npu/bridges/graph.h"
#include "lite/kernels/huawei_ascend_npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace huawei_ascend_npu {

int LayerNormConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  // Get input and output vars
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto x_rank = static_cast<int>(x_dims.size());
  CHECK(x_rank >= 2 && x_rank <= 4);

  bool has_bias = op_info->HasInput("Bias");
  bool has_scale = op_info->HasInput("Scale");

  auto y_name = op_info->Output("Y").front();
  auto y = scope->FindMutableTensor(y_name);
  auto y_dims = y->dims();

  auto mean_name = op_info->Output("Mean").front();
  auto mean = scope->FindMutableTensor(mean_name);
  auto mean_dims = mean->dims();
  CHECK_EQ(mean_dims.size(), 1);

  auto var_name = op_info->Output("Variance").front();
  auto var = scope->FindMutableTensor(var_name);
  auto var_dims = var->dims();
  CHECK_EQ(var_dims.size(), 1);

  // Get op attributes
  auto epsilon = op_info->GetAttr<float>("epsilon");
  auto begin_norm_axis = op_info->GetAttr<int>("begin_norm_axis");
  if (begin_norm_axis < 0) {
    begin_norm_axis += x_rank;
  }
  CHECK_GT(begin_norm_axis, 0);
  CHECK_LT(begin_norm_axis, x_rank);
  CHECK(begin_norm_axis >= 1 && begin_norm_axis < x_rank);
  auto matrix_dim = x_dims.Flatten2D(begin_norm_axis);
  int batch_size = matrix_dim[0];
  int feature_size = matrix_dim[1];
  CHECK_EQ(mean_dims.production(), batch_size);
  CHECK_EQ(var_dims.production(), batch_size);

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // Get shape of bias and scale
  DDim scale_bias_dims = x_dims.Slice(begin_norm_axis, x_dims.size());
  CHECK_EQ(scale_bias_dims.production(), feature_size);
  // auto scale_bias_dims = DDim({x_dims[x_dims.size()-1]});
  // Bias node
  std::shared_ptr<Node> bias_node = nullptr;
  if (has_bias) {
    auto bias_name = op_info->Input("Bias").front();
    auto bias = scope->FindMutableTensor(bias_name);
    auto bias_dims = bias->dims();
    CHECK_EQ(bias_dims.size(), 1);
    CHECK_EQ(bias_dims.production(), feature_size);
    bias_node = graph->Add(bias_name, *bias, scale_bias_dims);
  } else {
    bias_node = graph->Add<float>(y_name + "/bias", 0.f, scale_bias_dims);
  }

  // Scale node
  std::shared_ptr<Node> scale_node = nullptr;
  if (has_scale) {
    auto scale_name = op_info->Input("Scale").front();
    auto scale = scope->FindMutableTensor(scale_name);
    auto scale_dims = scale->dims();
    CHECK_EQ(scale_dims.size(), 1);
    CHECK_EQ(scale_dims.production(), feature_size);
    scale_node = graph->Add(scale_name, *scale, scale_bias_dims);
  } else {
    scale_node = graph->Add<float>(y_name + "/scale", 1.f, scale_bias_dims);
  }

  // LayerNorm node
  auto layer_norm_node = graph->Add<ge::op::LayerNorm>(y_name + "/layer_norm");
  auto layer_norm_op = layer_norm_node->data<ge::op::LayerNorm>();
  layer_norm_op->set_input_x(*x_node->data());
  layer_norm_op->set_input_gamma(*scale_node->data());
  layer_norm_op->set_input_beta(*bias_node->data());
  layer_norm_op->set_attr_begin_norm_axis(begin_norm_axis);
  layer_norm_op->set_attr_begin_params_axis(begin_norm_axis);
  layer_norm_op->set_attr_epsilon(epsilon);
  INPUT_UPDATE(layer_norm_op, x, x_node);
  INPUT_UPDATE(layer_norm_op, gamma, scale_node);
  INPUT_UPDATE(layer_norm_op, beta, bias_node);
  OUTPUT_UPDATE(layer_norm_op, y, layer_norm_node);
  OUTPUT_UPDATE(layer_norm_op, mean, layer_norm_node);
  OUTPUT_UPDATE(layer_norm_op, variance, layer_norm_node);

  // Get output of Y
  auto out_y_node = graph->Add<ge::op::Identity>(y_name);
  auto out_y_op = out_y_node->data<ge::op::Identity>();
  out_y_op->set_input_x(*layer_norm_node->data(), "y");
  INPUT_UPDATE(out_y_op, x, layer_norm_node);
  OUTPUT_UPDATE(out_y_op, y, out_y_node);

  // Get output of Mean
  auto out_mean_node = graph->Add<ge::op::Identity>(mean_name);
  auto out_mean_op = out_mean_node->data<ge::op::Identity>();
  out_mean_op->set_input_x(*layer_norm_node->data(), "mean");
  INPUT_UPDATE(out_mean_op, x, layer_norm_node);
  OUTPUT_UPDATE(out_mean_op, y, out_mean_node);

  // Get output of Variance
  auto out_var_node = graph->Add<ge::op::Identity>(var_name);
  auto out_var_op = out_var_node->data<ge::op::Identity>();
  out_var_op->set_input_x(*layer_norm_node->data(), "variance");
  INPUT_UPDATE(out_var_op, x, layer_norm_node);
  OUTPUT_UPDATE(out_var_op, y, out_var_node);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    layer_norm,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::LayerNormConverter);
