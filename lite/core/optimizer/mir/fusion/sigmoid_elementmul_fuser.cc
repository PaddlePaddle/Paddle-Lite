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

#include "lite/core/optimizer/mir/fusion/sigmoid_elementmul_fuser.h"
#include <cmath>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void SigmoidElementmulFuser::BuildPattern() {
  // create nodes.
  auto* x = VarNode("x")->assert_is_op_input("sigmoid", "X");
  // auto *x1 = VarNode("X")->assert_is_op_input("elementwise_mul", "X");
  // auto *y = VarNode("Y")->assert_is_op_input("elementwise_mul", "Y");

  auto* sigmoid_op = OpNode("sigmoid", "sigmoid");
  auto* sigmoid_out = VarNode("sigmoid_out");

  auto* elemul_op = OpNode("elementwise_mul", "elementwise_mul");
  auto* Out = VarNode("Out");
  // create topology.
  std::vector<PMNode*> sigmoid_inputs{x};
  std::vector<PMNode*> elemul_inputs{x, sigmoid_out};
  sigmoid_inputs >> *sigmoid_op >> *sigmoid_out;
  // some op specialities.
  sigmoid_out->AsIntermediate();
  sigmoid_op->AsIntermediate();
  elemul_op->AsIntermediate();

  elemul_inputs >> *elemul_op >> *Out;
}

cpp::OpDesc SigmoidElementmulFuser::GenOpDesc(const key2nodes_t& matched) {
  auto op_desc = *matched.at("sigmoid")->stmt()->op_info();

  op_desc.mutable_inputs()->clear();
  op_desc.mutable_outputs()->clear();
  // set input && output && attrbute of OpDesc
  op_desc.SetType("swish");
  op_desc.SetInput("X", {matched.at("x")->arg()->name});
  op_desc.SetAttr<float>("beta", 1);
  op_desc.SetOutput("Out", {matched.at("Out")->arg()->name});

  return op_desc;
}

void SigmoidElementmulFuser::InsertNewNode(SSAGraph* graph,
                                           const key2nodes_t& matched) {
  // create opdesc for fused_op
  auto op_desc = GenOpDesc(matched);
  // create fused_op
  auto swish_op = LiteOpRegistry::Global().Create("swish");
  // get scope and valid_places of old topo
  auto sigmoid = matched.at("sigmoid")->stmt()->op();
  auto* scope = sigmoid->scope();
  auto& valid_places = sigmoid->valid_places();
  // set scope and valid_places for fused_op
  swish_op->Attach(op_desc, scope);
  auto* new_op_node = graph->GraphCreateInstructNode(swish_op, valid_places);

  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("Out"));
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
