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

void CvtYShape(std::vector<int64_t>* x_shape,
               std::vector<int64_t>* y_shape,
               int axis) {
  CHECK_GE(x_shape->size(), y_shape->size());

  if (axis < 0) {
    axis = x_shape->size() - y_shape->size();
  }

  if (x_shape->size() < 4) {
    x_shape->insert(x_shape->begin(), 1);
    axis++;
  }
  while (x_shape->size() < 4) {
    x_shape->push_back(1);
  }

  y_shape->insert(y_shape->begin(), axis, 1);
  while (y_shape->size() < 4) {
    y_shape->push_back(1);
  }
  CHECK_EQ(x_shape->size(), 4UL);
  CHECK_EQ(y_shape->size(), 4UL);
}

int ElementwiseConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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

  auto y_name = op_info->Input("Y").front();
  auto y_type = kernel->GetInputDeclType("Y");
  CHECK(y_type->precision() == PRECISION(kFloat));
  CHECK(y_type->layout() == DATALAYOUT(kNCHW));
  auto y = scope->FindMutableTensor(y_name);
  auto y_dims = y->dims();

  auto out_name = op_info->Output("Out").front();
  auto out_type = kernel->GetOutputDeclType("Out");
  CHECK(out_type->precision() == PRECISION(kFloat));
  CHECK(out_type->layout() == DATALAYOUT(kNCHW));

  auto axis = op_info->GetAttr<int>("axis");

  auto x_new_shape = x_dims.Vectorize();
  auto y_new_shape = y_dims.Vectorize();
  CvtYShape(&x_new_shape, &y_new_shape, axis);

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x, x_new_shape);
  }

  // Y node
  std::shared_ptr<Node> y_node = nullptr;
  if (graph->Has(y_name)) {
    y_node = graph->Get(y_name);
  } else {
    y_node = graph->Add(y_name, *y, y_new_shape);
  }

  // Elementwise node
  std::shared_ptr<Node> elt_node = nullptr;
  if (op_type == "elementwise_add" ||
      op_type == "fusion_elementwise_add_activation") {
    elt_node = graph->Add<ge::op::Add>(out_name);
    auto elt_op = elt_node->data<ge::op::Add>();
    elt_op->set_input_x1(*x_node->data());
    elt_op->set_input_x2(*y_node->data());
  } else if (op_type == "elementwise_sub" ||
             op_type == "fusion_elementwise_sub_activation") {
    elt_node = graph->Add<ge::op::Sub>(out_name);
    auto elt_op = elt_node->data<ge::op::Sub>();
    elt_op->set_input_x1(*x_node->data());
    elt_op->set_input_x2(*y_node->data());
  } else if (op_type == "elementwise_mul" ||
             op_type == "fusion_elementwise_mul_activation") {
    elt_node = graph->Add<ge::op::Mul>(out_name);
    auto elt_op = elt_node->data<ge::op::Mul>();
    elt_op->set_input_x(*x_node->data());
    elt_op->set_input_y(*y_node->data());
  } else if (op_type == "elementwise_div" ||
             op_type == "fusion_elementwise_div_activation") {
    elt_node = graph->Add<ge::op::RealDiv>(out_name);
    auto elt_op = elt_node->data<ge::op::RealDiv>();
    elt_op->set_input_x1(*x_node->data());
    elt_op->set_input_x2(*y_node->data());
  } else {
    LOG(WARNING) << "[NPU] Unsupported op type: " << op_type;
    return FAILED;
  }

  // Act node
  if (op_type == "fusion_elementwise_add_activation" ||
      op_type == "fusion_elementwise_sub_activation" ||
      op_type == "fusion_elementwise_mul_activation" ||
      op_type == "fusion_elementwise_div_activation") {
    auto act_type = op_info->GetAttr<std::string>("act_type");
    auto act_node = graph->Add<ge::op::Activation>(out_name);
    auto act_op = act_node->data<ge::op::Activation>();
    act_op->set_input_x(*elt_node->data());
    // TODO(hong19860320) set the coef value for act Ops, such as leaky_relu,
    // clipped_relu etc.
    act_op->set_attr_mode(CvtActMode(act_type));
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(elementwise_add,
                         kNPU,
                         paddle::lite::subgraph::npu::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(elementwise_sub,
                         kNPU,
                         paddle::lite::subgraph::npu::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(elementwise_mul,
                         kNPU,
                         paddle::lite::subgraph::npu::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(elementwise_div,
                         kNPU,
                         paddle::lite::subgraph::npu::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(fusion_elementwise_add_activation,
                         kNPU,
                         paddle::lite::subgraph::npu::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(fusion_elementwise_sub_activation,
                         kNPU,
                         paddle::lite::subgraph::npu::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(fusion_elementwise_mul_activation,
                         kNPU,
                         paddle::lite::subgraph::npu::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(fusion_elementwise_div_activation,
                         kNPU,
                         paddle::lite::subgraph::npu::ElementwiseConverter);
