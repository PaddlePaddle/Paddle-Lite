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

#include "lite/core/optimizer/mir/fusion/p_norm_fill_constant_max_div_fuser.h"
#include <cmath>
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void PNormFillConstantMaxDivFuser::BuildPattern() {
  // create nodes.
  // p_norm
  auto* x = VarNode("x")->assert_is_op_input("p_norm", "X")->AsInput();
  auto* p_norm =
      OpNode("p_norm", "p_norm")
          ->assert_op_attr<bool>("keepdim", true)
          ->assert_op_attr<bool>("asvector", false)
          ->assert_op_attr_satisfied<float>(
              "porder",
              [](float attr) { return (std::fabs(attr - 2.0) < 1e-5); })
          ->AsIntermediate();
  auto* p_norm_out = VarNode("p_norm_out")->AsIntermediate();
  // fill_constant
  auto* fill_constant =
      OpNode("fill_constant", "fill_constant")->AsIntermediate();
  auto* fill_constant_out = VarNode("fill_constant_out")->AsIntermediate();
  // elementwise_max
  auto* elementwise_max =
      OpNode("elementwise_max", "elementwise_max")
          ->assert_op_attr_satisfied<int>(
              "axis", [](int attr) { return attr == -1 || attr == 0; })
          ->AsIntermediate();

  auto* elementwise_max_out = VarNode("elementwise_max_out")->AsIntermediate();
  // elementwise_div
  auto* elementwise_div =
      OpNode("elementwise_div", "elementwise_div")
          ->assert_op_attr_satisfied<int>(
              "axis", [](int attr) { return attr == -1 || attr == 0; })
          ->AsIntermediate();
  auto* elementwise_div_out = VarNode("elementwise_div_out")->AsOutput();

  // create topology.
  std::vector<PMNode*> max_inputs{fill_constant_out, p_norm_out};
  std::vector<PMNode*> div_inputs{x, elementwise_max_out};
  *x >> *p_norm >> *p_norm_out;
  *fill_constant >> *fill_constant_out;
  max_inputs >> *elementwise_max >> *elementwise_max_out;
  div_inputs >> *elementwise_div >> *elementwise_div_out;
}

void PNormFillConstantMaxDivFuser::InsertNewNode(SSAGraph* graph,
                                                 const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto norm_op = LiteOpRegistry::Global().Create("norm");
  auto p_norm = matched.at("p_norm")->stmt()->op();
  auto* scope = p_norm->scope();
  auto& valid_places = p_norm->valid_places();
  norm_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(norm_op, valid_places);

  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("elementwise_div_out"));
}

cpp::OpDesc PNormFillConstantMaxDivFuser::GenOpDesc(
    const key2nodes_t& matched) {
  cpp::OpDesc op_desc;
  op_desc.SetType("norm");
  op_desc.SetInput("X", {matched.at("x")->arg()->name});
  op_desc.SetOutput("Out", {matched.at("elementwise_div_out")->arg()->name});
  op_desc.SetAttr(
      "axis", matched.at("p_norm")->stmt()->op_info()->GetAttr<int>("axis"));
  op_desc.SetAttr(
      "epsilon",
      matched.at("fill_constant")->stmt()->op_info()->GetAttr<float>("value"));
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
