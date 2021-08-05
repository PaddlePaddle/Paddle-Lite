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

#include "lite/core/optimizer/mir/fusion/instance_norm_activation_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void InstanceNormActivationFuser::BuildPattern() {
  // create input nodes.
  auto* x = VarNode("x")->assert_is_op_input("instance_norm", "X")->AsInput();
  auto* bias =
      VarNode("bias")->assert_is_op_input("instance_norm", "Bias")->AsInput();
  auto* scale =
      VarNode("scale")->assert_is_op_input("instance_norm", "Scale")->AsInput();
  // create op nodes
  auto* instance_norm = OpNode("instance_norm", "instance_norm")
                            ->assert_is_op("instance_norm")
                            ->AsIntermediate();
  auto* act =
      OpNode("act", act_type_)->assert_is_op(act_type_)->AsIntermediate();

  // create intermediate nodes
  auto* instance_norm_out = VarNode("instance_norm_out")
                                ->assert_is_op_output("instance_norm", "Y")
                                ->assert_is_op_input(act_type_, "X")
                                ->AsIntermediate();
  auto* save_mean = VarNode("save_mean")
                        ->assert_is_op_output("instance_norm", "SavedMean")
                        ->AsIntermediate();
  auto* save_variance =
      VarNode("save_variance")
          ->assert_is_op_output("instance_norm", "SavedVariance")
          ->AsIntermediate();

  // create output node
  auto* out =
      VarNode("output")->assert_is_op_output(act_type_, "Out")->AsOutput();
  // create topology.
  std::vector<PMNode*> instance_inputs{x, scale, bias};
  std::vector<PMNode*> instance_outputs{
      instance_norm_out, save_mean, save_variance};

  instance_inputs >> *instance_norm >> instance_outputs;
  *instance_norm_out >> *act >> *out;
}

void InstanceNormActivationFuser::InsertNewNode(SSAGraph* graph,
                                                const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto instance_norm_op = LiteOpRegistry::Global().Create("instance_norm");
  auto instance_norm = matched.at("instance_norm")->stmt()->op();
  auto* scope = instance_norm->scope();
  auto& valid_places = instance_norm->valid_places();
  instance_norm_op->Attach(op_desc, scope);

  auto* new_op_node =
      graph->GraphCreateInstructNode(instance_norm_op, valid_places);

  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(matched.at("bias"), new_op_node);
  IR_NODE_LINK_TO(matched.at("scale"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("output"));
}

cpp::OpDesc InstanceNormActivationFuser::GenOpDesc(const key2nodes_t& matched) {
  auto op_desc = *matched.at("instance_norm")->stmt()->op_info();
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
  op_desc.SetOutput("Y", {out_name});
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
