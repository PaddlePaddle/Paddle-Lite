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
#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int BatchNormConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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
  auto scale_name = op_info->Input("Scale").front();
  auto scale = scope->FindMutableTensor(scale_name);
  auto bias_name = op_info->Input("Bias").front();
  auto bias = scope->FindMutableTensor(bias_name);
  auto mean_name = op_info->Input("Mean").front();
  auto mean = scope->FindMutableTensor(mean_name);
  auto variance_name = op_info->Input("Variance").front();
  auto variance = scope->FindMutableTensor(variance_name);
  auto y_name = op_info->Output("Y").front();
  float momentum = op_info->GetAttr<float>("momentum");
  float epsilon = op_info->GetAttr<float>("epsilon");
  int mode = 1;  // bnScale, bnBias tensor dims are 1xCx1x1
  bool use_global_stats = !op_info->HasAttr("use_global_stats") ||
                          op_info->GetAttr<bool>("use_global_stats");
  if (!use_global_stats) {
    LOG(WARNING) << "[NPU] Only use_global_stats=true is supported by HiAI DDK";
  }

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // Scale, Bias, Mean, Variance node
  auto scale_node = graph->Add(scale_name, *scale);
  auto bias_node = graph->Add(bias_name, *bias);
  auto mean_node = graph->Add(mean_name, *mean);
  auto variance_node = graph->Add(variance_name, *variance);

  // Batch Norm node
  auto batch_norm_node = graph->Add<ge::op::BatchNormExt2>(y_name);
  auto batch_norm_op = batch_norm_node->data<ge::op::BatchNormExt2>();
  batch_norm_op->set_input_x(*x_node->data());
  batch_norm_op->set_input_scale(*scale_node->data());
  batch_norm_op->set_input_offset(*bias_node->data());
  batch_norm_op->set_input_mean(*mean_node->data());
  batch_norm_op->set_input_variance(*variance_node->data());
  batch_norm_op->set_attr_momentum(momentum);
  batch_norm_op->set_attr_epsilon(epsilon);
  batch_norm_op->set_attr_mode(mode);
  batch_norm_op->set_attr_use_global_stats(use_global_stats);
  return SUCCESS;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(batch_norm,
                         kNPU,
                         paddle::lite::subgraph::npu::BatchNormConverter);
