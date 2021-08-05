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

#include "lite/core/mir/fusion/sequence_pool_concat_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

#define STR1(R) #R
#define STR2(R) STR1(R)

#define POOL_CONCAT_PATTERN(num)                                            \
  auto* x_##num = VarNode(STR2(sequence_pool_x_##num))                      \
                      ->assert_is_op_input("sequence_pool", "X")            \
                      ->AsInput();                                          \
  auto* sequence_pool_##num =                                               \
      OpNode(STR2(sequence_pool_##num), "sequence_pool")->AsIntermediate(); \
  auto* sequence_pool_##num##_out =                                         \
      VarNode(STR2(sequence_pool_##num##_out))                              \
          ->assert_is_op_output("sequence_pool", "Out")                     \
          ->assert_is_op_nth_input("concat", "X", num - 1)                  \
          ->AsIntermediate();                                               \
  auto* sequence_pool_##num##_idx =                                         \
      VarNode(STR2(sequence_pool_##num##_idx))                              \
          ->assert_is_op_output("sequence_pool", "MaxIndex")                \
          ->AsIntermediate();                                               \
  *sequence_pool_##num >> *sequence_pool_##num##_idx;                       \
  *x_##num >> *sequence_pool_##num >> *sequence_pool_##num##_out >> *concat;

// """
// merge {sequence_pool x 7, concat} => merge_sequence_pool_and_concat
//   src1              src2               src7            src1    src2      src7
//     |                |                                  |       |         |
//     v                v                                  |       |   ...   |
// sequence_pool  sequence_pool  ...(sequence_pool)        |       |         |
//     |                |              |              =>   -------------------
//     ---------------------------------                          |
//             |                                                  |
//             v                                                  v
//           concat                                     sequence_pool_concat
// """
void SequencePool7ConcatFuser::BuildPattern() {
  // create nodes.
  auto* concat = OpNode("concat", "concat")->AsIntermediate();

  auto* concat_out =
      VarNode("concat_out")->assert_is_op_output("concat", "Out");
  *concat >> *concat_out;

  POOL_CONCAT_PATTERN(1);
  POOL_CONCAT_PATTERN(2);
  POOL_CONCAT_PATTERN(3);
  POOL_CONCAT_PATTERN(4);
  POOL_CONCAT_PATTERN(5);
  POOL_CONCAT_PATTERN(6);
  POOL_CONCAT_PATTERN(7);
}

void SequencePool7ConcatFuser::InsertNewNode(SSAGraph* graph,
                                             const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto sequence_pool_concat_op =
      LiteOpRegistry::Global().Create("sequence_pool_concat");

  auto concat = matched.at("concat")->stmt()->op();
  auto* scope = concat->scope();
  auto& valid_places = concat->valid_places();
  sequence_pool_concat_op->Attach(op_desc, scope);

  auto* new_op_node =
      graph->GraphCreateInstructNode(sequence_pool_concat_op, valid_places);

  IR_NODE_LINK_TO(matched.at("sequence_pool_x_1"), new_op_node);
  IR_NODE_LINK_TO(matched.at("sequence_pool_x_2"), new_op_node);
  IR_NODE_LINK_TO(matched.at("sequence_pool_x_3"), new_op_node);
  IR_NODE_LINK_TO(matched.at("sequence_pool_x_4"), new_op_node);
  IR_NODE_LINK_TO(matched.at("sequence_pool_x_5"), new_op_node);
  IR_NODE_LINK_TO(matched.at("sequence_pool_x_6"), new_op_node);
  IR_NODE_LINK_TO(matched.at("sequence_pool_x_7"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("concat_out"));
}

cpp::OpDesc SequencePool7ConcatFuser::GenOpDesc(const key2nodes_t& matched) {
  cpp::OpDesc op_desc = *matched.at("concat")->stmt()->op_info();
  op_desc.SetType("sequence_pool_concat");
  op_desc.SetInput("X",
                   {matched.at("sequence_pool_x_1")->arg()->name,
                    matched.at("sequence_pool_x_2")->arg()->name,
                    matched.at("sequence_pool_x_3")->arg()->name,
                    matched.at("sequence_pool_x_4")->arg()->name,
                    matched.at("sequence_pool_x_5")->arg()->name,
                    matched.at("sequence_pool_x_6")->arg()->name,
                    matched.at("sequence_pool_x_7")->arg()->name});

  std::vector<std::string> pooltypes;
  pooltypes.push_back(matched.at("sequence_pool_1")
                          ->stmt()
                          ->op_info()
                          ->GetAttr<std::string>("pooltype"));
  pooltypes.push_back(matched.at("sequence_pool_2")
                          ->stmt()
                          ->op_info()
                          ->GetAttr<std::string>("pooltype"));
  pooltypes.push_back(matched.at("sequence_pool_3")
                          ->stmt()
                          ->op_info()
                          ->GetAttr<std::string>("pooltype"));
  pooltypes.push_back(matched.at("sequence_pool_4")
                          ->stmt()
                          ->op_info()
                          ->GetAttr<std::string>("pooltype"));
  pooltypes.push_back(matched.at("sequence_pool_5")
                          ->stmt()
                          ->op_info()
                          ->GetAttr<std::string>("pooltype"));
  pooltypes.push_back(matched.at("sequence_pool_6")
                          ->stmt()
                          ->op_info()
                          ->GetAttr<std::string>("pooltype"));
  pooltypes.push_back(matched.at("sequence_pool_7")
                          ->stmt()
                          ->op_info()
                          ->GetAttr<std::string>("pooltype"));
  op_desc.SetAttr("pooltype", pooltypes);

  op_desc.SetOutput("Out", {matched.at("concat_out")->arg()->name});

  return op_desc;
}

void SequencePool2ConcatFuser::BuildPattern() {
  // create nodes.
  auto* concat = OpNode("concat", "concat")->AsIntermediate();

  auto* concat_out =
      VarNode("concat_out")->assert_is_op_output("concat", "Out");
  *concat >> *concat_out;

  POOL_CONCAT_PATTERN(1);
  POOL_CONCAT_PATTERN(2);
}

void SequencePool2ConcatFuser::InsertNewNode(SSAGraph* graph,
                                             const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto sequence_pool_concat_op =
      LiteOpRegistry::Global().Create("sequence_pool_concat");

  auto concat = matched.at("concat")->stmt()->op();
  auto* scope = concat->scope();
  auto& valid_places = concat->valid_places();
  sequence_pool_concat_op->Attach(op_desc, scope);

  auto* new_op_node =
      graph->GraphCreateInstructNode(sequence_pool_concat_op, valid_places);

  IR_NODE_LINK_TO(matched.at("sequence_pool_x_1"), new_op_node);
  IR_NODE_LINK_TO(matched.at("sequence_pool_x_2"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("concat_out"));
}

cpp::OpDesc SequencePool2ConcatFuser::GenOpDesc(const key2nodes_t& matched) {
  cpp::OpDesc op_desc = *matched.at("concat")->stmt()->op_info();
  op_desc.SetType("sequence_pool_concat");
  op_desc.SetInput("X",
                   {matched.at("sequence_pool_x_1")->arg()->name,
                    matched.at("sequence_pool_x_2")->arg()->name});

  std::vector<std::string> pooltypes;
  pooltypes.push_back(matched.at("sequence_pool_1")
                          ->stmt()
                          ->op_info()
                          ->GetAttr<std::string>("pooltype"));
  pooltypes.push_back(matched.at("sequence_pool_2")
                          ->stmt()
                          ->op_info()
                          ->GetAttr<std::string>("pooltype"));

  op_desc.SetAttr("pooltype", pooltypes);
  op_desc.SetOutput("Out", {matched.at("concat_out")->arg()->name});

  return op_desc;
}

#undef POOL_CONCAT_PATTERN
#undef STR1
#undef STR2

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
