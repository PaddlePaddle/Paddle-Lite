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

#include "lite/core/optimizer/mir/fusion/fc_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void FcFuser::BuildPattern() {
  auto inputs_teller0 = [](const Node* node) -> bool {
    return true;
    auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
    auto input_w_name = op_desc.Input("Y").front();
    auto* scope = const_cast<Node*>(node)->AsStmt().op()->scope();
    auto w_shape = scope->FindVar(input_w_name)->Get<lite::Tensor>().dims();
    size_t w_rank = w_shape.size();

    return w_rank == 2;
  };

  auto inputs_teller1 = [](const Node* node) -> bool {
    auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
    auto input_b_name = op_desc.Input("Y").front();
    auto* scope = const_cast<Node*>(node)->AsStmt().op()->scope();
    auto b_shape = scope->FindVar(input_b_name)->Get<lite::Tensor>().dims();
    size_t b_rank = b_shape.size();
    return b_rank == 2 || b_rank == 1;
  };

  // create nodes.
  auto* x = VarNode("x")->assert_is_op_input("mul", "X");
  auto* W = VarNode("W")->assert_is_op_input("mul", "Y");
  auto* b = VarNode("b")->assert_is_persistable_var();
  auto* mul = OpNode("mul", "mul")->assert_node_satisfied(inputs_teller0);
  auto* mul_out = VarNode("mul_out");
  auto* add =
      OpNode("add", "elementwise_add")->assert_node_satisfied(inputs_teller1);
  auto* Out = VarNode("Out");

  // create topology.
  std::vector<PMNode*> mul_inputs{W, x};
  std::vector<PMNode*> add_inputs{mul_out, b};
  mul_inputs >> *mul >> *mul_out;

  // Some op specialities.
  mul_out->AsIntermediate();
  mul->AsIntermediate();
  add->AsIntermediate();

  if (with_relu_) {
    auto* add_out = VarNode("add_out");
    auto* relu = OpNode("relu", "relu");
    std::vector<PMNode*> relu_inputs{add_out};
    add_inputs >> *add >> *add_out;
    relu_inputs >> *relu >> *Out;
    add_out->AsIntermediate();
    relu->AsIntermediate();
  } else {
    add_inputs >> *add >> *Out;
  }
}

void FcFuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {
  auto mul = matched.at("mul")->stmt()->op();
  auto* scope = mul->scope();
  auto mul_weight = scope->FindVar(matched.at("W")->arg()->name);
  auto mul_weight_dims = mul_weight->Get<lite::Tensor>().dims();
  auto bias = scope->FindVar(matched.at("b")->arg()->name);
  auto bias_dims = bias->Get<lite::Tensor>().dims();
  if (bias_dims.size() == 2 && bias_dims[0] != 1) {
    nodes_.erase(nodes_.begin(), nodes_.end());
    LOG(WARNING) << "elementwiseadd bias size equal to 2, but bias[0] not "
                    "equal to 1 , eleminate failed";
    return;
  }
  if (bias_dims[bias_dims.size() - 1] != mul_weight_dims[1]) {
    nodes_.erase(nodes_.begin(), nodes_.end());
    LOG(WARNING) << "elementwise_add bias last shape not equal to weight "
                    "shape1, eleminate failed";
    return;
  }
  auto op_desc = GenOpDesc(matched);
  auto fc_op = LiteOpRegistry::Global().Create("fc");
  auto& valid_places = mul->valid_places();
  fc_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(fc_op, valid_places);

  IR_NODE_LINK_TO(matched.at("W"), new_op_node);
  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(matched.at("b"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("Out"));
}

cpp::OpDesc FcFuser::GenOpDesc(const key2nodes_t& matched) {
  auto op_desc = *matched.at("mul")->stmt()->op_info();

  // Get the input scale from mul
  std::vector<float> x_scale_vct;
  std::vector<float> y_scale_vct;
  auto input_x_name = op_desc.Input("X").front();
  auto input_y_name = op_desc.Input("Y").front();
  bool is_quantized_op = op_desc.HasInputScale(input_x_name) &&
                         op_desc.HasInputScale(input_y_name);
  if (is_quantized_op) {
    x_scale_vct = op_desc.GetInputScale(input_x_name);
    y_scale_vct = op_desc.GetInputScale(op_desc.Input("Y").front());
  }

  op_desc.mutable_inputs()->clear();
  op_desc.mutable_outputs()->clear();
  op_desc.SetType("fc");
  op_desc.SetInput("Input", {matched.at("x")->arg()->name});
  op_desc.SetInput("W", {matched.at("W")->arg()->name});
  op_desc.SetInput("Bias", {matched.at("b")->arg()->name});
  op_desc.SetOutput("Out", {matched.at("Out")->arg()->name});
  op_desc.SetAttr(
      "in_num_col_dims",
      matched.at("mul")->stmt()->op_info()->GetAttr<int>("x_num_col_dims"));
  if (with_relu_) {
    op_desc.SetAttr("activation_type", std::string{"relu"});
  }

  // Set the input scale into fc
  if (is_quantized_op) {
    op_desc.SetInputScale(matched.at("x")->arg()->name, x_scale_vct);
    op_desc.SetInputScale(matched.at("W")->arg()->name, y_scale_vct);
  }

  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
