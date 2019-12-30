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

#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int ActConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_type = kernel->GetInputDeclType("X");
  CHECK(x_type->precision() == PRECISION(kFloat));
  CHECK(x_type->layout() == DATALAYOUT(kNCHW));
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto out_name = op_info->Output("Out").front();
  auto out_type = kernel->GetOutputDeclType("Out");
  CHECK(out_type->precision() == PRECISION(kFloat));
  CHECK(out_type->layout() == DATALAYOUT(kNCHW));

  // X node
  std::shared_ptr<ge::Operator> x_node = nullptr;
  if (graph->HasNode(x_name)) {
    x_node = graph->GetNode(x_name);
  } else {
    x_node = graph->AddNode(x_name, x_dims);
  }

  // Act node
  auto act_node = graph->AddNode<ge::op::Activation>(out_name);
  act_node->set_input_x(*x_node);
  // TODO(hong19860320) set the coef value for act Ops, such as leaky_relu,
  // clipped_relu etc.
  act_node->set_attr_mode(CvtActMode(op_type));
  if (op_type == "relu_clipped") {
    auto Relu_clipped_coef = op_info->GetAttr<float>("Relu_clipped_coef");
    act_node->set_attr_coef(Relu_clipped_coef);
  } else if (op_type == "relu6") {
    float Relu_clipped_coef = 6.f;
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

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(NPU,
                         sigmoid,
                         paddle::lite::subgraph::npu::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(NPU, relu, paddle::lite::subgraph::npu::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(NPU, tanh, paddle::lite::subgraph::npu::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(NPU,
                         relu_clipped,
                         paddle::lite::subgraph::npu::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(NPU, relu6, paddle::lite::subgraph::npu::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(NPU,
                         leaky_relu,
                         paddle::lite::subgraph::npu::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(NPU, abs, paddle::lite::subgraph::npu::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(NPU,
                         softsign,
                         paddle::lite::subgraph::npu::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(NPU,
                         softplus,
                         paddle::lite::subgraph::npu::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(NPU,
                         hard_sigmoid,
                         paddle::lite::subgraph::npu::ActConverter);
