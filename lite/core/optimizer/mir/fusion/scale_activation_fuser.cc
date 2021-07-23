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

#include "lite/core/mir/fusion/scale_activation_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ScaleActivationFuser::BuildPattern() {
  // create input nodes.
  auto* x = VarNode("x")->assert_is_op_input("scale", "X")->AsInput();

  // create op nodes
  auto* scale =
      OpNode("scale", "scale")->assert_is_op("scale")->AsIntermediate();
  auto* act =
      OpNode("act", act_type_)->assert_is_op(act_type_)->AsIntermediate();

  // create intermediate nodes
  auto* scale_out = VarNode("scale_out")
                        ->assert_is_op_output("scale", "Out")
                        ->assert_is_op_input(act_type_, "X")
                        ->AsIntermediate();

  // create output node
  auto* out =
      VarNode("output")->assert_is_op_output(act_type_, "Out")->AsOutput();
  // create topology.
  *x >> *scale >> *scale_out;
  *scale_out >> *act >> *out;
}

void ScaleActivationFuser::InsertNewNode(SSAGraph* graph,
                                         const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto scale_op = LiteOpRegistry::Global().Create("scale");
  auto scale = matched.at("scale")->stmt()->op();
  auto* scope = scale->scope();
  auto& valid_places = scale->valid_places();
  scale_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(scale_op, valid_places);

  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("output"));
}

cpp::OpDesc ScaleActivationFuser::GenOpDesc(const key2nodes_t& matched) {
  auto op_desc = *matched.at("scale")->stmt()->op_info();
  auto* act_op_desc = matched.at("act")->stmt()->op_info();
  op_desc.SetAttr("activation_type", act_type_);
  if (act_type_ == "relu") {
    op_desc.SetAttr("fuse_relu", true);
  } else if (act_type_ == "relu6") {
    float alpha = act_op_desc->GetAttr<float>("threshold");
    op_desc.SetAttr("alpha", alpha);
  } else if (act_type_ == "leaky_relu") {
    float alpha = act_op_desc->GetAttr<float>("alpha");
    op_desc.SetAttr("alpha", alpha);
  }
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
