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

#include "lite/core/optimizer/mir/fusion/greater_than_cast_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void GreaterThanCastFuser::BuildPattern() {
  // create nodes
  // greater_than
  PMNode* input_x =
      VarNode("input_x")->assert_is_op_input("greater_than", "X")->AsInput();
  PMNode* input_y =
      VarNode("input_y")->assert_is_op_input("greater_than", "Y")->AsInput();
  PMNode* greater_than =
      OpNode("greater_than", "greater_than")->AsIntermediate();
  PMNode* greater_than_out = VarNode("greater_than_out")
                                 ->assert_is_op_output("greater_than", "Out")
                                 ->assert_is_op_input("cast", "X")
                                 ->AsIntermediate();

  // cast
  PMNode* cast = OpNode("cast", "cast")->AsIntermediate();
  PMNode* out = VarNode("out")->assert_is_op_output("cast", "Out")->AsOutput();

  // create topology.
  std::vector<PMNode*> greater_than_inputs{input_x, input_y};
  greater_than_inputs >> *greater_than >> *greater_than_out >> *cast >> *out;
}

void GreaterThanCastFuser::InsertNewNode(SSAGraph* graph,
                                         const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto greater_than = LiteOpRegistry::Global().Create("greater_than");
  auto greater_than_old = matched.at("greater_than")->stmt()->op();
  auto* scope = greater_than_old->scope();
  auto& valid_places = greater_than_old->valid_places();
  greater_than->Attach(op_desc, scope);

  auto* new_op_node =
      graph->GraphCreateInstructNode(greater_than, valid_places);

  IR_NODE_LINK_TO(matched.at("input_x"), new_op_node);
  IR_NODE_LINK_TO(matched.at("input_y"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("out"));
}

cpp::OpDesc GreaterThanCastFuser::GenOpDesc(const key2nodes_t& matched) {
  cpp::OpDesc op_desc = *matched.at("greater_than")->stmt()->op_info();
  op_desc.mutable_inputs()->clear();
  op_desc.mutable_outputs()->clear();

  op_desc.SetInput("X", {matched.at("input_x")->arg()->name});
  op_desc.SetInput("Y", {matched.at("input_y")->arg()->name});
  op_desc.SetOutput("Out", {matched.at("out")->arg()->name});
  op_desc.SetAttr("fuse_greater_than", true);
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
