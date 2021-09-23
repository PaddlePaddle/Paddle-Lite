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

#include "lite/core/optimizer/mir/fusion/elementwise_add_activation_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ElementwiseActivationFuser::BuildPattern() {
  // create input nodes.
  auto* x = VarNode("x")->assert_is_op_input(eltwise_type_, "X")->AsInput();
  auto* y = VarNode("y")->assert_is_op_input(eltwise_type_, "Y")->AsInput();

  // create op nodes
  auto* elt = OpNode("elt", eltwise_type_)
                  ->assert_is_op(eltwise_type_)
                  ->AsIntermediate();
  auto* act =
      OpNode("act", act_type_)->assert_is_op(act_type_)->AsIntermediate();

  // create intermediate nodes
  auto* elt_out = VarNode("add_out")
                      ->assert_is_op_output(eltwise_type_, "Out")
                      ->assert_is_op_input(act_type_, "X")
                      ->AsIntermediate();

  // create output node
  auto* out =
      VarNode("output")->assert_is_op_output(act_type_, "Out")->AsOutput();

  // create topology.
  std::vector<PMNode*> elt_inputs{x, y};
  elt_inputs >> *elt >> *elt_out;
  *elt_out >> *act >> *out;
}

void ElementwiseActivationFuser::InsertNewNode(SSAGraph* graph,
                                               const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  std::shared_ptr<lite::OpLite> op;
  if (eltwise_type_ == "elementwise_add") {
    op = LiteOpRegistry::Global().Create("fusion_elementwise_add_activation");
  } else if (eltwise_type_ == "elementwise_sub") {
    op = LiteOpRegistry::Global().Create("fusion_elementwise_sub_activation");
  } else if (eltwise_type_ == "elementwise_mul") {
    op = LiteOpRegistry::Global().Create("fusion_elementwise_mul_activation");
  } else {
    LOG(FATAL) << "not supported elementwise_type: " << eltwise_type_;
  }

  auto old_op = matched.at("elt")->stmt()->op();
  auto* scope = old_op->scope();
  auto& valid_places = old_op->valid_places();
  op->Attach(op_desc, scope);
  auto* new_op_node = graph->GraphCreateInstructNode(op, valid_places);

  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(matched.at("y"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("output"));
}

cpp::OpDesc ElementwiseActivationFuser::GenOpDesc(const key2nodes_t& matched) {
  auto op_desc = *matched.at("elt")->stmt()->op_info();
  auto* act_op_desc = matched.at("act")->stmt()->op_info();
  if (eltwise_type_ == "elementwise_add") {
    op_desc.SetType("fusion_elementwise_add_activation");
  } else if (eltwise_type_ == "elementwise_sub") {
    op_desc.SetType("fusion_elementwise_sub_activation");
  } else if (eltwise_type_ == "elementwise_mul") {
    op_desc.SetType("fusion_elementwise_mul_activation");
  } else {
    LOG(FATAL) << "not supported elementwise_type: " << eltwise_type_;
  }
  op_desc.SetAttr("act_type", act_type_);
  auto& out_name = matched.at("output")->arg()->name;
  op_desc.SetOutput("Out", {out_name});
  if (act_op_desc->HasOutputScale(out_name)) {
    op_desc.SetOutputScale(out_name, act_op_desc->GetOutputScale(out_name));
  }
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
