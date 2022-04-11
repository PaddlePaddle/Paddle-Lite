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

#include "optimizer/test.h"
#include <algorithm>
#include <map>
#include <vector>
#include "core/types.h"
#include "utility/debug.h"
#include "utility/graph.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

void TestFuser::BuildPattern() {
  NNADAPTER_LOG(INFO) << "aaaaaaaaaaaaa";
  // Fill op
  auto* matmul = OpNode("matmul", NNADAPTER_MAT_MUL);
  auto* add = OpNode("add", NNADAPTER_ADD);

  // Range op
  auto* malmul_input_x = VarNode("matmul_x")
                             ->assert_is_op_nth_input(NNADAPTER_MAT_MUL, 0)
                             ->assert_var_not_persistable()
                             ->AsInput();
  auto* malmul_input_y = VarNode("matmul_y")
                             ->assert_is_op_nth_input(NNADAPTER_MAT_MUL, 1)
                             ->assert_is_persistable_var()
                             ->AsInput();
  // auto* malmul_input_x =
  // VarNode("matmul_x")->assert_var_not_persistable()->AsInput();
  // auto* malmul_input_y =
  // VarNode("matmul_y")->assert_is_persistable_var()->AsInput();
  auto* malmul_out = VarNode("matmul_out")->AsOutput();
  auto* add_input_y =
      VarNode("add_input_y")->assert_is_persistable_var()->AsInput();
  // auto* add_input_y =
  // VarNode("add_input_y")->assert_is_op_nth_input(NNADAPTER_ADD,
  // 1)->assert_is_persistable_var()->AsInput();
  auto* add_out = VarNode("add_out")->AsOutput();

  // create topology.
  std::vector<PMNode*> mul_inputs{malmul_input_x, malmul_input_y};
  std::vector<PMNode*> add_inputs{malmul_out, add_input_y};
  mul_inputs >> *matmul >> *malmul_out;
  add_inputs >> *add >> *add_out;

  // // Some op specialities.
  // conv2d->AsIntermediate();
  // add->AsIntermediate();
  // relu->AsIntermediate();
}

void TestFuser::InsertNewNode(Graph* graph, const key2nodes_t& matched) {
  //   auto op_desc = GenOpDesc(matched);
  //   auto range_op = LiteOpRegistry::Global().Create("range");
  //   auto range = matched.at("range")->stmt()->op();
  //   auto* scope = range->scope();
  //   auto& valid_places = range->valid_places();
  //   range_op->Attach(op_desc, scope);

  //   // Create new range op node
  //   auto* new_op_node = graph->GraphCreateInstructNode(range_op,
  //   valid_places);
  //   auto new_op = new_op_node->stmt()->op();

  //   IR_NODE_LINK_TO(matched.at("start"), new_op_node);
  //   IR_NODE_LINK_TO(matched.at("end"), new_op_node);
  //   IR_NODE_LINK_TO(matched.at("step"), new_op_node);
  //   IR_NODE_LINK_TO(new_op_node, matched.at("range_out"));
}

// cpp::OpDesc FillRangeFuser::GenOpDesc(const key2nodes_t& matched) {
// }

}  // namespace nnadapter
