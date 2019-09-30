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

#include "lite/operators/batch_norm_op.h"
#include "ai_ddk_lib/include/graph/buffer.h"
#include "ai_ddk_lib/include/graph/graph.h"
#include "ai_ddk_lib/include/graph/model.h"
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "ai_ddk_lib/include/graph/operator.h"
#include "ai_ddk_lib/include/graph/operator_reg.h"
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/utils.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

node_map_type BatchNormConverter(
    const std::shared_ptr<lite::OpLite> batch_norm_op,
    const node_map_type& inputs_map) {
  auto scope = batch_norm_op->scope();
  auto op_info = batch_norm_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = UniqueName(op_type);
  LOG(INFO) << "Converting " + op_type + "...";

  std::shared_ptr<ge::op::BatchNorm> batch_norm_node =
      std::make_shared<ge::op::BatchNorm>(unique_op_type);
  auto x_var_name = op_info->Input("X").front();

  auto scale_var_name = op_info->Input("Scale").front();
  lite::Tensor* scale = scope->FindVar(scale_var_name)->GetMutable<Tensor>();
  auto npu_scale = std::make_shared<ge::op::Const>(scale_var_name);
  npu_scale->set_attr_value(CvtFromLiteTensor(scale));
  OpList::Global().add(npu_scale);

  auto bias_var_name = op_info->Input("Bias").front();
  lite::Tensor* bias = scope->FindVar(bias_var_name)->GetMutable<Tensor>();
  auto npu_bias = std::make_shared<ge::op::Const>(bias_var_name);
  npu_bias->set_attr_value(CvtFromLiteTensor(bias));
  OpList::Global().add(npu_bias);

  auto mean_var_name = op_info->Input("Mean").front();
  lite::Tensor* mean = scope->FindVar(mean_var_name)->GetMutable<Tensor>();
  auto npu_mean = std::make_shared<ge::op::Const>(mean_var_name);
  npu_mean->set_attr_value(CvtFromLiteTensor(mean));
  OpList::Global().add(npu_mean);

  auto variance_var_name = op_info->Input("Variance").front();
  lite::Tensor* variance =
      scope->FindVar(variance_var_name)->GetMutable<Tensor>();
  auto npu_variance = std::make_shared<ge::op::Const>(variance_var_name);
  npu_variance->set_attr_value(CvtFromLiteTensor(variance));
  OpList::Global().add(npu_variance);

  float npu_momentum = op_info->GetAttr<float>("momentum");
  float npu_epsilon = op_info->GetAttr<float>("epsilon");
  int npu_mode = 1;  // bnScale, bnBias tensor dims are 1xCx1x1
  bool npu_use_global_stats = op_info->GetAttr<bool>("use_global_stats");

  batch_norm_node->set_input_x(*inputs_map.at(x_var_name));
  batch_norm_node->set_input_scale(*npu_scale);
  batch_norm_node->set_input_b(*npu_bias);
  batch_norm_node->set_input_mean(*npu_mean);
  batch_norm_node->set_input_variance(*npu_variance);
  batch_norm_node->set_attr_momentum(npu_momentum);
  batch_norm_node->set_attr_epsilon(npu_epsilon);
  batch_norm_node->set_attr_mode(npu_mode);
  batch_norm_node->set_attr_use_global_stats(npu_use_global_stats);

  OpList::Global().add(inputs_map.at(x_var_name));
  OpList::Global().add(batch_norm_node);

  node_map_type outputs_map;
  outputs_map[op_info->Output("Y").front()] = batch_norm_node;
  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(batch_norm, paddle::lite::npu::bridge::BatchNormConverter);
