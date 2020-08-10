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

int BatchNormConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  // Get input data nodes
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
  // Get output var nodes
  auto y_name = op_info->Output("Y").front();
  // Get attributes
  float epsilon = op_info->GetAttr<float>("epsilon");
  // Check is_test
  auto is_test_type = op_info->GetAttrType("is_test");
  if (is_test_type == OpDescAPI::AttrType::INT) {
    CHECK_EQ(op_info->GetAttr<int>("is_test"), 1)
        << "[HUAWEI_ASCEND_NPU] Only is_test=1 or is_test=true is supported in "
           "inference mode.";
  } else if (is_test_type == OpDescAPI::AttrType::BOOLEAN) {
    CHECK_EQ(op_info->GetAttr<bool>("is_test"), true)
        << "[HUAWEI_ASCEND_NPU] Only is_test=1 or is_test=true is supported in "
           "inference mode.";
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

  // Batch Norm node - output nodes
  auto batch_norm_node = graph->Add<ge::op::BatchNorm>(y_name + "/batch_norm");
  auto batch_norm_op = batch_norm_node->data<ge::op::BatchNorm>();
  batch_norm_op->set_input_x(*x_node->data());
  batch_norm_op->set_input_scale(*scale_node->data());
  batch_norm_op->set_input_offset(*bias_node->data());
  batch_norm_op->set_input_mean(*mean_node->data());
  batch_norm_op->set_input_variance(*variance_node->data());
  batch_norm_op->set_attr_epsilon(epsilon);
  batch_norm_op->set_attr_data_format("NCHW");
  batch_norm_op->set_attr_is_training(false);
  INPUT_UPDATE(batch_norm_op, x, x_node);
  INPUT_UPDATE(batch_norm_op, scale, scale_node);
  INPUT_UPDATE(batch_norm_op, offset, bias_node);
  INPUT_UPDATE(batch_norm_op, mean, mean_node);
  INPUT_UPDATE(batch_norm_op, variance, variance_node);
  OUTPUT_UPDATE(batch_norm_op, y, batch_norm_node);

  // Create Variable node for batch norm output y
  auto out_y_node = graph->Add<ge::op::Identity>(y_name);
  auto out_y_op = out_y_node->data<ge::op::Identity>();
  out_y_op->set_input_x(*batch_norm_node->data(), "y");
  INPUT_UPDATE(out_y_op, x, batch_norm_node);
  OUTPUT_UPDATE(out_y_op, y, out_y_node);

  return SUCCESS;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    batch_norm,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::BatchNormConverter);
