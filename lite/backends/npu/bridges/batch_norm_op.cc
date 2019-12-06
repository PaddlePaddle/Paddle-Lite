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

#include "lite/backends/npu/bridges/registry.h"
#include "lite/backends/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridges {

int BatchNormConverter(cvt_ctx_type* ctx, lite::OpLite* op) {
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  auto x_var_name = op_info->Input("X").front();
  auto y_var_name = op_info->Output("Y").front();
  auto batch_norm_node = ctx->AddNode<ge::op::BatchNormExt2>(y_var_name);
  CHECK(ctx->HasNode(x_var_name));
  batch_norm_node->set_input_x(*ctx->GetNode(x_var_name));

  auto scale_var_name = op_info->Input("Scale").front();
  auto scale = scope->FindVar(scale_var_name)->GetMutable<Tensor>();
  auto scale_const_node = ctx->AddNode<ge::op::Const>(scale_var_name);
  scale_const_node->set_attr_value(CvtTensor(scale));

  auto bias_var_name = op_info->Input("Bias").front();
  auto bias = scope->FindVar(bias_var_name)->GetMutable<Tensor>();
  auto bias_const_node = ctx->AddNode<ge::op::Const>(bias_var_name);
  bias_const_node->set_attr_value(CvtTensor(bias));

  auto mean_var_name = op_info->Input("Mean").front();
  auto mean = scope->FindVar(mean_var_name)->GetMutable<Tensor>();
  auto mean_const_node = ctx->AddNode<ge::op::Const>(mean_var_name);
  mean_const_node->set_attr_value(CvtTensor(mean));

  auto variance_var_name = op_info->Input("Variance").front();
  auto variance = scope->FindVar(variance_var_name)->GetMutable<Tensor>();
  auto variance_const_node = ctx->AddNode<ge::op::Const>(variance_var_name);
  variance_const_node->set_attr_value(CvtTensor(variance));

  float momentum = op_info->GetAttr<float>("momentum");
  float epsilon = op_info->GetAttr<float>("epsilon");
  int mode = 1;  // bnScale, bnBias tensor dims are 1xCx1x1
  bool use_global_stats = op_info->GetAttr<bool>("use_global_stats");

  batch_norm_node->set_input_scale(*scale_const_node);
  batch_norm_node->set_input_offset(*bias_const_node);
  batch_norm_node->set_input_mean(*mean_const_node);
  batch_norm_node->set_input_variance(*variance_const_node);
  batch_norm_node->set_attr_momentum(momentum);
  batch_norm_node->set_attr_epsilon(epsilon);
  batch_norm_node->set_attr_mode(mode);
  batch_norm_node->set_attr_use_global_stats(use_global_stats);
  return SUCCESS;
}

}  // namespace bridges
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(batch_norm, paddle::lite::npu::bridges::BatchNormConverter);
