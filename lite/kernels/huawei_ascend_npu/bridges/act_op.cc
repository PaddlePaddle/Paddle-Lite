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

template <typename ActType>
int ActConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto out_name = op_info->Output("Out").front();

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // Act node
  auto act_node = graph->template Add<ActType>(out_name);
  auto act_op = act_node->template data<ActType>();
  act_op->set_input_x(*x_node->data());
  INPUT_UPDATE(act_op, x, x_node);
  OUTPUT_UPDATE(act_op, y, act_node);

  return SUCCESS;
}

template <>
int ActConverter<ge::op::LeakyRelu>(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto out_name = op_info->Output("Out").front();

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // Act node
  auto act_node = graph->template Add<ge::op::LeakyRelu>(out_name);
  auto act_op = act_node->template data<ge::op::LeakyRelu>();
  act_op->set_input_x(*x_node->data());
  // only for leaky_relu
  auto alpha = op_info->GetAttr<float>("alpha");
  act_op->set_attr_negative_slope(alpha);
  INPUT_UPDATE(act_op, x, x_node);
  OUTPUT_UPDATE(act_op, y, act_node);

  return SUCCESS;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    sigmoid,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::ActConverter<ge::op::Sigmoid>);
REGISTER_SUBGRAPH_BRIDGE(
    relu,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::ActConverter<ge::op::Relu>);
REGISTER_SUBGRAPH_BRIDGE(
    tanh,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::ActConverter<ge::op::Tanh>);
REGISTER_SUBGRAPH_BRIDGE(
    relu6,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::ActConverter<ge::op::Relu6>);
REGISTER_SUBGRAPH_BRIDGE(
    leaky_relu,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::ActConverter<ge::op::LeakyRelu>);
REGISTER_SUBGRAPH_BRIDGE(
    softsign,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::ActConverter<ge::op::Softsign>);
REGISTER_SUBGRAPH_BRIDGE(
    softplus,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::ActConverter<ge::op::Softplus>);
