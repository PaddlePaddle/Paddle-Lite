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

#include "lite/core/mir/fusion/conv_activation_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ConvActivationFuser::BuildPattern() {
  // create nodes.
  auto* input =
      VarNode("input")->assert_is_op_input(conv_type_, "Input")->AsInput();
  auto* filter =
      VarNode("filter")->assert_is_op_input(conv_type_, "Filter")->AsInput();
  PMNode* bias = nullptr;
  if (has_bias_) {
    bias = VarNode("bias")->assert_is_op_input(conv_type_, "Bias")->AsInput();
  }
  auto* conv2d = OpNode("conv2d", conv_type_)->AsIntermediate();

  auto* act = OpNode("act", act_type_)->AsIntermediate();

  auto* conv2d_out = VarNode("conv2d_out")
                         ->assert_is_op_output(conv_type_, "Output")
                         ->assert_is_op_input(act_type_, "X")
                         ->AsIntermediate();

  auto* out =
      VarNode("output")->assert_is_op_output(act_type_, "Out")->AsOutput();

  // create topology.
  std::vector<PMNode*> conv2d_inputs{filter, input};
  conv2d_inputs >> *conv2d >> *conv2d_out >> *act >> *out;
  if (has_bias_) {
    *bias >> *conv2d;
  }
}

void ConvActivationFuser::InsertNewNode(SSAGraph* graph,
                                        const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto conv_op = LiteOpRegistry::Global().Create(conv_type_);
  auto conv_old = matched.at("conv2d")->stmt()->op();
  auto* scope = conv_old->scope();
  auto& valid_places = conv_old->valid_places();
  conv_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(conv_op, valid_places);

  IR_NODE_LINK_TO(matched.at("input"), new_op_node);
  IR_NODE_LINK_TO(matched.at("filter"), new_op_node);
  if (has_bias_) {
    IR_NODE_LINK_TO(matched.at("bias"), new_op_node);
  }
  IR_NODE_LINK_TO(new_op_node, matched.at("output"));
}

cpp::OpDesc ConvActivationFuser::GenOpDesc(const key2nodes_t& matched) {
  cpp::OpDesc op_desc = *matched.at("conv2d")->stmt()->op_info();
  op_desc.SetOutput("Output", {matched.at("output")->arg()->name});
  cpp::OpDesc act_op_desc = *matched.at("act")->stmt()->op_info();

  op_desc.SetAttr("with_act", true);
  op_desc.SetAttr("act_type", act_type_);
  if (act_type_ == "relu") {
    op_desc.SetAttr("fuse_relu", true);
  } else if (act_type_ == "relu6") {
    float alpha = act_op_desc.GetAttr<float>("threshold");
    op_desc.SetAttr("fuse_brelu_threshold", alpha);
  } else if (act_type_ == "leaky_relu") {
    float alpha = act_op_desc.GetAttr<float>("alpha");
    op_desc.SetAttr("leaky_relu_alpha", alpha);
  }
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
