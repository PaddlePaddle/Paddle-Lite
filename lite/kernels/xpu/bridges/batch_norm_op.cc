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

#include "lite/backends/xpu/builder.h"
#include "lite/kernels/xpu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {
namespace bridges {

node_map_type BatchNormConverter(const std::shared_ptr<lite::OpLite> op,
                                 graph_ctx_type* graph_ctx,
                                 const node_map_type& input_nodes) {
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = lite::xpu::UniqueName(op_type);
  LOG(INFO) << "[XPU] Converting " + op_type + "...";

  // check context
  CHECK(graph_ctx != nullptr);
  CHECK(graph_ctx->builder != nullptr);
  CHECK(graph_ctx->params != nullptr);

  // get input, and attributes
  auto x_var_name = op_info->Input("X").front();
  auto scale_var_name = op_info->Input("Scale").front();
  auto* scale = scope->FindMutableTensor(scale_var_name);
  auto bias_var_name = op_info->Input("Bias").front();
  auto* bias = scope->FindMutableTensor(bias_var_name);
  auto mean_var_name = op_info->Input("Mean").front();
  auto* mean = scope->FindMutableTensor(mean_var_name);
  auto variance_var_name = op_info->Input("Variance").front();
  auto* variance = scope->FindMutableTensor(variance_var_name);
  auto epsilon = op_info->GetAttr<float>("epsilon");

  // create scale node
  CHECK(!input_nodes.count(scale_var_name));
  auto scale_const_node = std::make_shared<xtcl::xExpr>(
      graph_ctx->builder->CreateTensor(scale_var_name,
                                       lite::xpu::CvtShape(scale->dims()),
                                       ::xtcl::Float(32)));
  auto scale_const_tensor = lite::xpu::CvtTensor(scale);
  graph_ctx->params->emplace(
      std::make_pair(scale_var_name, *scale_const_tensor));

  // create bias node
  CHECK(!input_nodes.count(bias_var_name));
  auto bias_const_node =
      std::make_shared<xtcl::xExpr>(graph_ctx->builder->CreateTensor(
          bias_var_name, lite::xpu::CvtShape(bias->dims()), ::xtcl::Float(32)));
  auto bias_const_tensor = lite::xpu::CvtTensor(bias);
  graph_ctx->params->emplace(std::make_pair(bias_var_name, *bias_const_tensor));

  // create mean node
  CHECK(!input_nodes.count(mean_var_name));
  auto mean_const_node =
      std::make_shared<xtcl::xExpr>(graph_ctx->builder->CreateTensor(
          mean_var_name, lite::xpu::CvtShape(mean->dims()), ::xtcl::Float(32)));
  auto mean_const_tensor = lite::xpu::CvtTensor(mean);
  graph_ctx->params->emplace(std::make_pair(mean_var_name, *mean_const_tensor));

  // create variance node
  CHECK(!input_nodes.count(variance_var_name));
  auto variance_const_node = std::make_shared<xtcl::xExpr>(
      graph_ctx->builder->CreateTensor(variance_var_name,
                                       lite::xpu::CvtShape(variance->dims()),
                                       ::xtcl::Float(32)));
  auto variance_const_tensor = lite::xpu::CvtTensor(variance);
  graph_ctx->params->emplace(
      std::make_pair(variance_var_name, *variance_const_tensor));

  // create batch_norm node and set params from op
  CHECK(input_nodes.count(x_var_name));
  auto batch_norm_node = std::make_shared<xtcl::xExpr>(
      graph_ctx->builder->CreateBatchNorm(*input_nodes.at(x_var_name),
                                          *scale_const_node,
                                          *bias_const_node,
                                          *mean_const_node,
                                          *variance_const_node,
                                          1,
                                          epsilon));
  batch_norm_node = std::make_shared<xtcl::xExpr>(
      graph_ctx->builder->GetField(*batch_norm_node, 0));
  graph_ctx->builder->SetLayer(unique_op_type);

  // output converted nodes
  node_map_type output_nodes;
  output_nodes[op_info->Output("Y").front()] = batch_norm_node;
  return output_nodes;
}

}  // namespace bridges
}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_XPU_BRIDGE(batch_norm,
                    paddle::lite::kernels::xpu::bridges::BatchNormConverter);
