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

  // only support:
  // (n,c,h,w) * (n,c,h,w)
  // (n,c,h,w) * (1,c,1,1)
  // (n,c,h,w) * (1,c,h,1)
  // (n,c,h,w) * (1,c,h,w)
  int y_shape_size = y_shape->size();
  if (y_shape_size == 1) {
    y_shape->insert(y_shape->begin(), 1);
    y_shape->insert(y_shape->end(), 2, 1);
  } else if (y_shape_size == 2) {
    y_shape->insert(y_shape->begin(), 1);
    y_shape->insert(y_shape->end(), 1);
  } else if (y_shape_size == 3) {
    y_shape->insert(y_shape->begin(), 1);
  }
  if (y_shape_size < 4) {
    int n = 1;
    for (int i = 0; i < axis; i++) {
      n *= x_shape->at(i);
    }
    x_shape->erase(x_shape->begin(), x_shape->begin() + axis);
    x_shape->insert(x_shape->begin(), n);
    x_shape->insert(x_shape->end(), 4 - x_shape->size(), 1);
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
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();

  auto y_name = op_info->Input("Y").front();
  auto y = scope->FindMutableTensor(y_name);
  auto y_dims = y->dims();

  auto out_name = op_info->Output("Out").front();
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();

  auto axis = op_info->GetAttr<int>("axis");

  auto x_new_shape = x_dims.Vectorize();
  auto y_new_shape = y_dims.Vectorize();
  CvtYShape(&x_new_shape, &y_new_shape, axis);

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
    if (x_dims.Vectorize() != x_new_shape) {
      auto reshaped_x_node = graph->Add<ge::op::Reshape>(x_name + "/reshape");
      auto reshaped_x_op = reshaped_x_node->data<ge::op::Reshape>();
      reshaped_x_op->set_input_tensor(*x_node->data());
      reshaped_x_op->set_attr_shape(
          ge::AttrValue::LIST_INT(x_new_shape.begin(), x_new_shape.end()));
      reshaped_x_op->set_attr_axis(0);
      x_node = reshaped_x_node;
    }
  } else {
    x_node = graph->Add(x_name, *x, x_new_shape);
  }

  // Y node
  std::shared_ptr<Node> y_node = nullptr;
  if (graph->Has(y_name)) {
    y_node = graph->Get(y_name);
    if (y_dims.Vectorize() != y_new_shape) {
      auto reshaped_y_node = graph->Add<ge::op::Reshape>(y_name + "/reshape");
      auto reshaped_y_op = reshaped_y_node->data<ge::op::Reshape>();
      reshaped_y_op->set_input_tensor(*y_node->data());
      reshaped_y_op->set_attr_shape(
          ge::AttrValue::LIST_INT(y_new_shape.begin(), y_new_shape.end()));
      reshaped_y_op->set_attr_axis(0);
      y_node = reshaped_y_node;
    }
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

  if (out_dims.Vectorize() != x_new_shape) {
    auto reshaped_elt_node = graph->Add<ge::op::Reshape>(out_name);
    auto reshaped_elt_op = reshaped_elt_node->data<ge::op::Reshape>();
    reshaped_elt_op->set_input_tensor(*elt_node->data());
    auto out_shape = out_dims.Vectorize();
    reshaped_elt_op->set_attr_shape(
        ge::AttrValue::LIST_INT(out_shape.begin(), out_shape.end()));
    reshaped_elt_op->set_attr_axis(0);
    elt_node = reshaped_elt_node;
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
