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

#include "lite/core/mir/fusion/fc_prelu_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void FcPreluFuser::BuildPattern() {
  // create nodes
  // fc
  PMNode* input =
      VarNode("input")->assert_is_op_input("fc", "Input")->AsInput();
  PMNode* weights =
      VarNode("weights")->assert_is_op_input("fc", "W")->AsInput();
  PMNode* bias = VarNode("bias")->assert_is_op_input("fc", "Bias")->AsInput();
  PMNode* fc = OpNode("fc", "fc")->AsIntermediate();
  PMNode* fc_out = VarNode("fc_out")
                       ->assert_is_op_output("fc", "Out")
                       ->assert_is_op_input("prelu", "X")
                       ->AsIntermediate();

  // prelu
  PMNode* alpha =
      VarNode("alpha")->assert_is_op_input("prelu", "Alpha")->AsInput();
  PMNode* prelu = OpNode("prelu", "prelu")->AsIntermediate();
  PMNode* out =
      VarNode("output")->assert_is_op_output("prelu", "Out")->AsOutput();

  // create topology.
  std::vector<PMNode*> fc_inputs{bias, weights, input};
  fc_inputs >> *fc >> *fc_out >> *prelu >> *out;
  *alpha >> *prelu;
}

void FcPreluFuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto fc_op = LiteOpRegistry::Global().Create("fc");
  auto fc_old = matched.at("fc")->stmt()->op();
  auto* scope = fc_old->scope();
  auto& valid_places = fc_old->valid_places();
  fc_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(fc_op, valid_places);

  IR_NODE_LINK_TO(matched.at("input"), new_op_node);
  IR_NODE_LINK_TO(matched.at("weights"), new_op_node);
  IR_NODE_LINK_TO(matched.at("bias"), new_op_node);
  IR_NODE_LINK_TO(matched.at("alpha"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("output"));
}

cpp::OpDesc FcPreluFuser::GenOpDesc(const key2nodes_t& matched) {
  cpp::OpDesc op_desc = *matched.at("fc")->stmt()->op_info();
  op_desc.SetInput("Alpha", {matched.at("alpha")->arg()->name});
  op_desc.SetOutput("Out", {matched.at("output")->arg()->name});

  cpp::OpDesc prelu_op_desc = *matched.at("prelu")->stmt()->op_info();
  auto prelu_mode = prelu_op_desc.GetAttr<std::string>("mode");
  op_desc.SetAttr("prelu_mode", prelu_mode);
  op_desc.SetAttr("activation_type", std::string{"prelu"});
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
