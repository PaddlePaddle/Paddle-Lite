// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/mir/fusion/scaleacts_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ScaleactsFuser::BuildPattern() {
  // create input nodes.
  auto* x = VarNode("x")->assert_is_op_input("scale", "X")->AsInput();

  auto scales_teller1 = [](const Node* node) -> bool {
    bool bias_after_scale =
        const_cast<Node*>(node)->AsStmt().op_info()->GetAttr<bool>(
            "bias_after_scale");
    bool has_act =
        const_cast<Node*>(node)->AsStmt().op_info()->HasAttr("activation_type");
    return bias_after_scale && has_act;
  };
  auto scales_teller2 = [](const Node* node) -> bool {
    bool bias_after_scale =
        const_cast<Node*>(node)->AsStmt().op_info()->GetAttr<bool>(
            "bias_after_scale");
    bool has_act =
        const_cast<Node*>(node)->AsStmt().op_info()->HasAttr("activation_type");
    return bias_after_scale && (!has_act);
  };

  // create op nodes
  auto* scale1 = OpNode("scale1", "scale")
                     ->assert_is_op("scale")
                     ->assert_node_satisfied(scales_teller1)
                     ->AsIntermediate();
  auto* scale2 = OpNode("scale2", "scale")
                     ->assert_is_op("scale")
                     ->assert_node_satisfied(scales_teller2)
                     ->AsIntermediate();

  // create intermediate nodes
  auto* scale1_out = VarNode("scale1_out")
                         ->assert_is_op_output("scale", "Out")
                         ->assert_is_op_input("scale", "X")
                         ->AsIntermediate();

  // create output node
  auto* out = VarNode("out")->assert_is_op_output("scale", "Out")->AsOutput();

  // create topology.
  *x >> *scale1 >> *scale1_out >> *scale2 >> *out;
}

void ScaleactsFuser::InsertNewNode(SSAGraph* graph,
                                   const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto scale_op = LiteOpRegistry::Global().Create("scale");
  auto scale = matched.at("scale1")->stmt()->op();
  auto* scope = scale->scope();
  auto& valid_places = scale->valid_places();
  scale_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(scale_op, valid_places);

  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("out"));
}

cpp::OpDesc ScaleactsFuser::GenOpDesc(const key2nodes_t& matched) {
  auto* op_desc_tmp = matched.at("scale2")->stmt()->op_info();
  float scale1 = op_desc_tmp->GetAttr<float>("scale");
  float bias1 = op_desc_tmp->GetAttr<float>("bias");

  auto op_desc = *matched.at("scale1")->stmt()->op_info();
  op_desc.SetAttr("fuse_scaleact", true);
  op_desc.SetAttr("scale1", scale1);
  op_desc.SetAttr("bias1", bias1);

  auto& out_name = matched.at("out")->arg()->name;
  op_desc.SetOutput("Out", {out_name});

  // should we consider int8 case?

  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
