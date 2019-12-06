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

int ElementwiseConverter(cvt_ctx_type* ctx, OpLite* op) {
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto op_name = ctx->UniqueName(op_type);
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  auto x_var_name = op_info->Input("X").front();
  auto y_var_name = op_info->Input("Y").front();
  auto out_var_name = op_info->Output("Out").front();

  CHECK(ctx->HasNode(x_var_name));
  std::shared_ptr<ge::Operator> x_node = ctx->GetNode(x_var_name);
  std::shared_ptr<ge::Operator> y_node = nullptr;
  if (ctx->HasNode(y_var_name)) {
    y_node = ctx->GetNode(y_var_name);
  } else {
    auto y_const_node = ctx->AddNode<ge::op::Const>(y_var_name);
    auto* y = scope->FindMutableTensor(y_var_name);
    y_const_node->set_attr_value(CvtTensor(y));
    y_node = y_const_node;
  }

  std::shared_ptr<ge::Operator> elementwise_node = nullptr;
  if (op_type == "elementwise_add" ||
      op_type == "fusion_elementwise_add_activation") {
    auto elt_node = ctx->AddNode<ge::op::Add>(op_name);
    elt_node->set_input_x1(*x_node);
    elt_node->set_input_x2(*y_node);
    elementwise_node = elt_node;
  } else if (op_type == "elementwise_sub") {
    auto elt_node = ctx->AddNode<ge::op::Sub>(op_name);
    elt_node->set_input_x1(*x_node);
    elt_node->set_input_x2(*y_node);
    elementwise_node = elt_node;
  } else if (op_type == "elementwise_mul") {
    auto elt_node = ctx->AddNode<ge::op::Mul>(op_name);
    elt_node->set_input_x(*x_node);
    elt_node->set_input_y(*y_node);
    elementwise_node = elt_node;
  } else if (op_type == "elementwise_div") {
    auto elt_node = ctx->AddNode<ge::op::RealDiv>(op_name);
    elt_node->set_input_x1(*x_node);
    elt_node->set_input_x2(*y_node);
    elementwise_node = elt_node;
  } else {
    LOG(WARNING) << "[NPU] Unsupported op type: " << op_type;
    return FAILED;
  }

  if (op_type == "fusion_elementwise_add_activation") {
    auto act_type = op_info->GetAttr<std::string>("act_type");
    auto act_node = ctx->AddNode<ge::op::Activation>(op_name + "/act");
    act_node->set_input_x(*elementwise_node);
    // TODO(hong19860320) set the coef value for act Ops, such as leaky_relu,
    // clipped_relu etc.
    act_node->set_attr_mode(CvtActMode(act_type));
    ctx->SetNode(out_var_name, act_node);
  } else {
    ctx->SetNode(out_var_name, elementwise_node);
  }
  return SUCCESS;
}

}  // namespace bridges
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(elementwise_add,
                    paddle::lite::npu::bridges::ElementwiseConverter);
REGISTER_NPU_BRIDGE(fusion_elementwise_add_activation,
                    paddle::lite::npu::bridges::ElementwiseConverter);
REGISTER_NPU_BRIDGE(elementwise_sub,
                    paddle::lite::npu::bridges::ElementwiseConverter);
REGISTER_NPU_BRIDGE(elementwise_mul,
                    paddle::lite::npu::bridges::ElementwiseConverter);
REGISTER_NPU_BRIDGE(elementwise_div,
                    paddle::lite::npu::bridges::ElementwiseConverter);
