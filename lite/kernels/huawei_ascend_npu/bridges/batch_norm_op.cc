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


#include "lite/operators/batch_norm_op.h"
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

  // Get input vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);

  auto mean_name = op_info->Input("Mean").front();
  auto mean = scope->FindMutableTensor(mean_name);

  auto variance_name = op_info->Input("Variance").front();
  auto variance = scope->FindMutableTensor(variance_name);

  auto bias_name = op_info->Input("Bias").front();
  auto bias = scope->FindMutableTensor(bias_name);

  auto scale_name = op_info->Input("Scale").front();
  auto scale = scope->FindMutableTensor(scale_name);

  // Get output vars
  auto y_name = op_info->Output("Y").front();

  // Get other attr
  float epsilon = op_info->GetAttr<float>("epsilon");
  int mode = 1;  // bnScale, bnBias tensor dims are 1xCx1x1
  bool is_test = !op_info->HasAttr("is_test") ||
                  op_info->GetAttr<bool>("is_test");
  bool use_global_stats = !op_info->HasAttr("use_global_stats") ||
                           op_info->GetAttr<bool>("use_global_stats");
  use_global_stats = is_test || use_global_stats;
  if (!use_global_stats) {
     LOG(WARNING) << "[HUAWEI_ASCEND_NPU] Only use_global_stats=true is supported";
  }

  // Input node
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

  // new momentum_node, size=1
  auto* momentum_tensor = scope->NewTensor(x_name + "/momentum");
  momentum_tensor->Resize({1});
  momentum_tensor->set_persistable(true);
  momentum_tensor->set_precision(PrecisionType::kFloat);
  auto momentum_tensor_data = momentum_tensor->mutable_data<float>();
  momentum_tensor_data[0] = 1.;
  auto momentum_node = graph->Add(x_name + "/momentum", *momentum_tensor);

  // Batch Norm node
  auto batch_norm_node = graph->Add<ge::op::BNInference>(y_name);
  auto batch_norm_op = batch_norm_node->data<ge::op::BNInference>();
  batch_norm_op->set_input_x(*x_node->data());
  batch_norm_op->set_input_mean(*mean_node->data());
  batch_norm_op->set_input_variance(*variance_node->data());
  batch_norm_op->set_input_scale(*scale_node->data());
  batch_norm_op->set_input_offset(*bias_node->data());
  batch_norm_op->set_input_momentum(*momentum_node->data());
  batch_norm_op->set_attr_epsilon(epsilon);
  batch_norm_op->set_attr_use_global_stats(use_global_stats);
  batch_norm_op->set_attr_mode(mode);

  TENSOR_UPDATE_INPUT(batch_norm_op,
                      x,
                      ge::FORMAT_NCHW,
                      CvtPrecisionType(x_node->precision()));
  TENSOR_UPDATE_INPUT(batch_norm_op,
                      mean,
                      ge::FORMAT_NCHW,
                      CvtPrecisionType(mean_node->precision()));
  TENSOR_UPDATE_INPUT(batch_norm_op,
                      variance,
                      ge::FORMAT_NCHW,
                      CvtPrecisionType(variance_node->precision()));
  TENSOR_UPDATE_INPUT(batch_norm_op,
                      scale,
                      ge::FORMAT_NCHW,
                      CvtPrecisionType(scale_node->precision()));
  TENSOR_UPDATE_INPUT(batch_norm_op,
                      offset,
                      ge::FORMAT_NCHW,
                      CvtPrecisionType(bias_node->precision()));
  TENSOR_UPDATE_INPUT(batch_norm_op,
                      momentum,
                      ge::FORMAT_NCHW,
                      CvtPrecisionType(momentum_node->precision()));
  TENSOR_UPDATE_OUTPUT(batch_norm_op,
                       y,
                       ge::FORMAT_NCHW,
                       CvtPrecisionType(batch_norm_node->precision()));

  return SUCCESS;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(batch_norm,
                         kHuaweiAscendNPU,
                         paddle::lite::subgraph::huawei_ascend_npu::BatchNormConverter);
