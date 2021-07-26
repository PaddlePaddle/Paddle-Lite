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

#include "lite/core/optimizer/mir/fusion/sequence_reverse_embedding_fuser.h"

#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void SequenceReverseEmbeddingFuser::BuildPattern() {
  // create input nodes.
  auto* x =
      VarNode("x")->assert_is_op_input("sequence_reverse", "X")->AsInput();
  auto* w = VarNode("w")->assert_is_op_input("lookup_table", "W")->AsInput();

  // create op nodes
  auto* sequence_reverse = OpNode("sequence_reverse", "sequence_reverse")
                               ->assert_is_op("sequence_reverse")
                               ->AsIntermediate();
  auto* lookup_table = OpNode("lookup_table", "lookup_table")
                           ->assert_is_op("lookup_table")
                           ->AsIntermediate();

  // create intermediate nodes
  auto* sequence_reverse_out =
      VarNode("sequence_reverse_out")
          ->assert_is_op_output("sequence_reverse", "Y")
          ->assert_is_op_input("lookup_table", "Ids")
          ->AsIntermediate();

  // create output node
  auto* out =
      VarNode("out")->assert_is_op_output("lookup_table", "Out")->AsOutput();

  // create topology.
  *x >> *sequence_reverse >> *sequence_reverse_out >> *lookup_table >> *out;
  *w >> *lookup_table;
}

void SequenceReverseEmbeddingFuser::InsertNewNode(SSAGraph* graph,
                                                  const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto fuse_op = LiteOpRegistry::Global().Create("sequence_reverse_embedding");
  auto lookup_table = matched.at("lookup_table")->stmt()->op();
  auto* scope = lookup_table->scope();
  auto& valid_places = lookup_table->valid_places();
  fuse_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(fuse_op, valid_places);

  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(matched.at("w"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("out"));
}

cpp::OpDesc SequenceReverseEmbeddingFuser::GenOpDesc(
    const key2nodes_t& matched) {
  auto op_desc = *matched.at("lookup_table")->stmt()->op_info();
  op_desc.SetType("sequence_reverse_embedding");
  auto& in_name = matched.at("x")->arg()->name;
  auto& w_name = matched.at("w")->arg()->name;
  auto& out_name = matched.at("out")->arg()->name;
  op_desc.SetInput("Ids", {in_name});
  op_desc.SetInput("W", {w_name});
  op_desc.SetOutput("Out", {out_name});
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
