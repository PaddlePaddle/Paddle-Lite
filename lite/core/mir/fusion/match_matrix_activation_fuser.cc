// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/mir/fusion/match_matrix_activation_fuser.h"

#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void MatchMatrixActFuser::BuildPattern() {
  // create nodes.
  auto* x = VarNode("x")->assert_is_op_input("match_matrix_tensor", "X");
  auto* W = VarNode("W")->assert_is_op_input("match_matrix_tensor", "W");
  auto* y = VarNode("y")->assert_is_op_input("match_matrix_tensor", "Y");
  auto* mm = OpNode("match_matrix_tensor", "match_matrix_tensor");
  auto* mm_out =
      VarNode("mm_out")->assert_is_op_output("match_matrix_tensor", "Out");
  auto* mm_tmp =
      VarNode("mm_tmp")->assert_is_op_output("match_matrix_tensor", "Tmp");
  auto* act = OpNode("act", activation_);
  auto* out = VarNode("Out")->assert_is_op_output(activation_, "Out");

  // create topology.
  std::vector<PMNode*> mm_inputs{x, W, y};
  std::vector<PMNode*> mm_ouputs{mm_out, mm_tmp};
  mm_inputs >> *mm >> mm_ouputs;

  // Some op specialities.
  mm_out->AsIntermediate();
  mm->AsIntermediate();
  act->AsIntermediate();

  *mm_out >> *act >> *out;
}

void MatchMatrixActFuser::InsertNewNode(SSAGraph* graph,
                                        const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto mm_op = LiteOpRegistry::Global().Create("match_matrix_tensor");
  auto mm = matched.at("match_matrix_tensor")->stmt()->op();
  auto* scope = mm->scope();
  auto& valid_places = mm->valid_places();
  mm_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(mm_op, valid_places);

  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(matched.at("W"), new_op_node);
  IR_NODE_LINK_TO(matched.at("y"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("Out"));
}

cpp::OpDesc MatchMatrixActFuser::GenOpDesc(const key2nodes_t& matched) {
  auto op_desc = *matched.at("match_matrix_tensor")->stmt()->op_info();
  int dim_t = matched.at("match_matrix_tensor")
                  ->stmt()
                  ->op_info()
                  ->GetAttr<int>("dim_t");
  op_desc.mutable_inputs()->clear();
  op_desc.mutable_outputs()->clear();
  op_desc.SetType("match_matrix_tensor");
  op_desc.SetInput("X", {matched.at("x")->arg()->name});
  op_desc.SetInput("W", {matched.at("W")->arg()->name});
  op_desc.SetInput("Y", {matched.at("y")->arg()->name});
  op_desc.SetOutput("Out", {matched.at("Out")->arg()->name});
  op_desc.SetOutput("Tmp", {matched.at("mm_tmp")->arg()->name});
  op_desc.SetAttr("dim_t", dim_t);
  op_desc.SetAttr("fuse_relu", true);

  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
