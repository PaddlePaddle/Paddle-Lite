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
  auto* conv2d =
      OpNode("conv2d", conv_type_)->assert_is_op(conv_type_)->AsIntermediate();
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
  std::vector<PMNode*> conv2d_inputs{filter, input};
  // consider a special case: conv with bias
  if (conv_has_bias_) {
    PMNode* conv_bias = VarNode("conv_bias")
                            ->assert_is_op_input(conv_type_, "Bias")
                            ->AsIntermediate();
    conv2d_inputs.emplace_back(conv_bias);
  }
  std::vector<PMNode*> add_inputs{conv2d_out, bias};
  conv2d_inputs >> *conv2d >> *conv2d_out;
  add_inputs >> *add >> *add_out;
}

void ConvElementwiseFuser::InsertNewNode(SSAGraph* graph,
                                         const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto conv_op = LiteOpRegistry::Global().Create(conv_type_);
  auto old_conv_instruct = matched.at("conv2d")->stmt();
  auto old_conv_op = old_conv_instruct->op();
  auto* scope = old_conv_op->scope();
  auto& valid_places = old_conv_op->valid_places();

  // elementwise_add bias
  auto elementwise_add_bias_t = scope->FindVar(matched.at("bias")->arg()->name)
                                    ->GetMutable<lite::Tensor>();
  auto elementwise_add_bias_d = elementwise_add_bias_t->mutable_data<float>();

  // conv weight
  auto conv_weight_t = scope->FindVar(matched.at("filter")->arg()->name)
                           ->GetMutable<lite::Tensor>();

  /////////////////////////////////////////////////////////////////////////////////////
  // ConvElementwiseFuser
  //   if `conv_bias` existed, store previous old `conv_bias` to
  //   `new_conv_bias`,
  //     add `elementwise_add_bias` to `new_conv_bias`.
  //   if `conv_bias` not existed, initalize `new_conv_bias` with zero value,
  //   with {conv_weight_t.dims()[0], 1, 1, 1} dimension,
  //     accumulate `elementwise_add_bias` to `new_conv_bias`.
  /////////////////////////////////////////////////////////////////////////////////////
  Tensor new_conv_bias_t;
  new_conv_bias_t.Resize({conv_weight_t->dims()[0], 1, 1, 1});
  auto new_conv_bias_d = new_conv_bias_t.mutable_data<float>();

  if (conv_has_bias_ == true && op_desc.HasInput("Bias") &&
      op_desc.Input("Bias").size() > 0) {
    auto conv_bias_var = scope->FindVar(op_desc.Input("Bias").front());
    if (conv_bias_var != nullptr) {
      auto old_conv_bias_t = &(conv_bias_var->Get<lite::Tensor>());
      new_conv_bias_t.CopyDataFrom(*old_conv_bias_t);
    }
  } else {  // conv_has_bias_ == false
    for (unsigned int i = 0; i < new_conv_bias_t.data_size(); ++i) {
      new_conv_bias_d[i] = 0;
    }
  }
  // add `elementwise_add_bias` to `new_conv_bias`
  CHECK(elementwise_add_bias_t->data_size() == new_conv_bias_t.data_size())
      << "elementwise_add_bias_t.data_size() != new_conv_bias_t.data_size()";
  for (unsigned int i = 0; i < new_conv_bias_t.data_size(); ++i) {
    new_conv_bias_d[i] += elementwise_add_bias_d[i];
  }

  /// store `new_conv_bias` in `elementwise_add_bias`
  elementwise_add_bias_t->CopyDataFrom(new_conv_bias_t);

  conv_op->Attach(op_desc, scope);
  auto* new_op_node = graph->GraphCreateInstructNode(conv_op, valid_places);

  IR_NODE_LINK_TO(matched.at("input"), new_op_node);
  IR_NODE_LINK_TO(matched.at("filter"), new_op_node);
  IR_NODE_LINK_TO(matched.at("bias"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("output"));
}

cpp::OpDesc ConvElementwiseFuser::GenOpDesc(const key2nodes_t& matched) {
  auto* desc = matched.at("conv2d")->stmt()->op_info();
  cpp::OpDesc op_desc = *desc;

  op_desc.SetType(conv_type_);
  op_desc.SetInput("Input", {matched.at("input")->arg()->name});
  op_desc.SetInput("Filter", {matched.at("filter")->arg()->name});
  op_desc.SetInput("Bias", {matched.at("bias")->arg()->name});
  op_desc.SetOutput("Output", {matched.at("output")->arg()->name});

  // Other inputs. See operators/conv_op.h
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
