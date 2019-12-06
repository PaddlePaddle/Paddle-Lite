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

int ActConverter(cvt_ctx_type* ctx, lite::OpLite* op) {
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Create act node and set input node which is obtained from the node map
  auto x_var_name = op_info->Input("X").front();
  auto out_var_name = op_info->Output("Out").front();
  auto act_node = ctx->AddNode<ge::op::Activation>(out_var_name);
  CHECK(ctx->HasNode(x_var_name));
  act_node->set_input_x(*ctx->GetNode(x_var_name));

  // TODO(hong19860320) set the coef value for act Ops, such as leaky_relu,
  // clipped_relu etc.
  act_node->set_attr_mode(CvtActMode(op_type));

  if (op_type == "relu_clipped") {
    auto Relu_clipped_coef = op_info->GetAttr<float>("Relu_clipped_coef");
    act_node->set_attr_coef(Relu_clipped_coef);
  } else if (op_type == "leaky_relu") {
    auto alpha = op_info->GetAttr<float>("alpha");
    act_node->set_attr_negative_slope(alpha);
  } else if (op_type == "hard_sigmoid") {
    auto slope = op_info->GetAttr<float>("slope");
    auto offset = op_info->GetAttr<float>("offset");
    act_node->set_attr_negative_slope(slope);
    act_node->set_attr_coef(offset);
  }
  return SUCCESS;
}

}  // namespace bridges
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(sigmoid, paddle::lite::npu::bridges::ActConverter);
REGISTER_NPU_BRIDGE(relu, paddle::lite::npu::bridges::ActConverter);
REGISTER_NPU_BRIDGE(tanh, paddle::lite::npu::bridges::ActConverter);
REGISTER_NPU_BRIDGE(relu_clipped, paddle::lite::npu::bridges::ActConverter);
// REGISTER_NPU_BRIDGE(elu, paddle::lite::npu::bridges::ActConverter);
REGISTER_NPU_BRIDGE(leaky_relu, paddle::lite::npu::bridges::ActConverter);
REGISTER_NPU_BRIDGE(abs, paddle::lite::npu::bridges::ActConverter);
REGISTER_NPU_BRIDGE(softsign, paddle::lite::npu::bridges::ActConverter);
REGISTER_NPU_BRIDGE(softplus, paddle::lite::npu::bridges::ActConverter);
REGISTER_NPU_BRIDGE(hard_sigmoid, paddle::lite::npu::bridges::ActConverter);
