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

#include "lite/core/mir/fusion/conv_elementwise_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ConvElementwiseFuser::BuildPattern() {
  // create input nodes.
  auto* input =
      VarNode("input")->assert_is_op_input(conv_type_, "Input")->AsInput();
  auto* filter =
      VarNode("filter")->assert_is_op_input(conv_type_, "Filter")->AsInput();
  auto* bias = VarNode("bias")
                   ->assert_is_op_input("elementwise_add", "Y")
                   ->AsInput()
                   ->assert_is_persistable_var();

  // create op nodes
  auto* conv2d = OpNode("conv2d", conv_type_)->assert_is_op(conv_type_);
  auto* add = OpNode("add", "elementwise_add")
                  ->assert_is_op("elementwise_add")
                  ->AsIntermediate();

  // create intermediate nodes
  auto* conv2d_out = VarNode("conv2d_out")
                         ->assert_is_op_output(conv_type_, "Output")
                         ->assert_is_op_input("elementwise_add", "X")
                         ->AsIntermediate();
  // create output node
  auto* add_out = VarNode("output")
                      ->assert_is_op_output("elementwise_add", "Out")
                      ->AsOutput();

  // create topology.
  if (has_bias_) {
    auto* conv_bias = VarNode("conv_bias")
                          ->assert_is_op_input(conv_type_, "Bias")
                          ->AsInput()
                          ->AsIntermediate();
    conv2d->LinksFrom({input, filter, conv_bias}).LinksTo({conv2d_out});
  } else {
    conv2d->LinksFrom({input, filter}).LinksTo({conv2d_out});
  }
  add->LinksFrom({conv2d_out, bias}).LinksTo({add_out});
}

void ConvElementwiseFuser::InsertNewNode(SSAGraph* graph,
                                         const key2nodes_t& matched) {
  auto conv_instruct = matched.at("conv2d")->stmt();
  auto conv_op_desc = conv_instruct->mutable_op_info();
  auto* scope = conv_instruct->op()->scope();

  if (has_bias_) {
    auto bias_t = scope->FindVar(matched.at("bias")->arg()->name)
                      ->GetMutable<lite::Tensor>();
    auto bias_d = bias_t->mutable_data<float>();

    auto conv_bias_var_t = scope->FindVar(matched.at("conv_bias")->arg()->name)
                               ->GetMutable<lite::Tensor>();
    auto conv_bias_var_d = conv_bias_var_t->mutable_data<float>();
    for (int i = 0; i < bias_t->numel(); i++) {
      bias_d[i] += conv_bias_var_d[i];
    }
  }
  conv_op_desc->SetType(conv_type_);
  conv_op_desc->SetInput("Input", {matched.at("input")->arg()->name});
  conv_op_desc->SetInput("Filter", {matched.at("filter")->arg()->name});
  conv_op_desc->SetOutput("Output", {matched.at("output")->arg()->name});
  conv_op_desc->SetInput("Bias", {matched.at("bias")->arg()->name});
  auto update_conv_desc = *conv_instruct->mutable_op_info();
  conv_instruct->ResetOp(update_conv_desc, graph->valid_places());

  IR_NODE_LINK_TO(matched.at("bias"), matched.at("conv2d"));
  IR_OP_VAR_LINK(matched.at("conv2d"), matched.at("output"));
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
