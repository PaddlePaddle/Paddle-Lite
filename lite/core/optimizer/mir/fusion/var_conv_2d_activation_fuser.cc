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

#include "lite/core/mir/fusion/var_conv_2d_activation_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void VarConvActivationFuser::BuildPattern() {
  // create nodes.
  auto* input = VarNode("X")->assert_is_op_input(conv_type_, "X")->AsInput();
  auto* filter = VarNode("W")->assert_is_op_input(conv_type_, "W")->AsInput();
  auto* column =
      VarNode("COLUMN")->assert_is_op_input(conv_type_, "COLUMN")->AsInput();
  auto* row = VarNode("ROW")->assert_is_op_input(conv_type_, "ROW")->AsInput();

  auto* conv2d = OpNode("var_conv_2d", conv_type_)->AsIntermediate();

  auto* act = OpNode("act", act_type_)->AsIntermediate();

  auto* conv2d_out = VarNode("conv2d_out")
                         ->assert_is_op_output(conv_type_, "Out")
                         ->assert_is_op_input(act_type_, "X")
                         ->AsIntermediate();
  auto* conv2d_out_1 = VarNode("conv2d_out_1")
                           ->assert_is_op_output(conv_type_, "Col")
                           ->AsIntermediate();

  auto* out =
      VarNode("output")->assert_is_op_output(act_type_, "Out")->AsOutput();

  // create topology.
  std::vector<PMNode*> conv2d_inputs{filter, input, column, row};
  conv2d_inputs >> *conv2d >> *conv2d_out >> *act >> *out;
  *conv2d >> *conv2d_out_1;
}

void VarConvActivationFuser::InsertNewNode(SSAGraph* graph,
                                           const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto conv_op = LiteOpRegistry::Global().Create(conv_type_);
  auto conv_old = matched.at("var_conv_2d")->stmt()->op();
  auto* scope = conv_old->scope();
  auto& valid_places = conv_old->valid_places();
  conv_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(conv_op, valid_places);

  IR_NODE_LINK_TO(matched.at("X"), new_op_node);
  IR_NODE_LINK_TO(matched.at("W"), new_op_node);
  IR_NODE_LINK_TO(matched.at("COLUMN"), new_op_node);
  IR_NODE_LINK_TO(matched.at("ROW"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("output"));
}

cpp::OpDesc VarConvActivationFuser::GenOpDesc(const key2nodes_t& matched) {
  cpp::OpDesc op_desc = *matched.at("var_conv_2d")->stmt()->op_info();
  op_desc.SetOutput("Out", {matched.at("output")->arg()->name});
  cpp::OpDesc act_op_desc = *matched.at("act")->stmt()->op_info();

  if (act_type_ == "relu") {
    op_desc.SetAttr("fuse_relu", true);
  }
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
