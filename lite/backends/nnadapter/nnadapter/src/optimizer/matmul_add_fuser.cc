// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "optimizer/matmul_add_fuser.h"
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

void MatMulAddFuser::BuildPattern() {
  // Teller function about matmul's inputs
  auto matmul_inputs_teller = [](const Node* node) -> bool {
    auto operation = *const_cast<Node*>(node)->operation();
    auto op_inlinks = operation.input_operands;
    return op_inlinks.size() == 4 &&
           op_inlinks[0]->type.dimensions.count == 2 &&
           op_inlinks[1]->type.dimensions.count == 2;
  };
  // Op node
  auto* matmul = OpNode("matmul", NNADAPTER_MAT_MUL)
                     ->assert_node_satisfied(matmul_inputs_teller);
  auto* add = OpNode("add", NNADAPTER_ADD);
  // Var node
  auto* matmul_input_x = VarNode("matmul_input_x")
                             ->assert_is_op_nth_input(NNADAPTER_MAT_MUL, 0)
                             ->assert_var_not_persistable()
                             ->AsInput();
  auto* matmul_input_y = VarNode("matmul_input_y")
                             ->assert_is_op_nth_input(NNADAPTER_MAT_MUL, 1)
                             ->assert_is_persistable_var()
                             ->AsInput();
  auto* matmul_transpose_x = VarNode("matmul_transpose_x")
                                 ->assert_is_op_nth_input(NNADAPTER_MAT_MUL, 2)
                                 ->assert_is_persistable_var()
                                 ->AsInput();
  auto* matmul_transpose_y = VarNode("matmul_transpose_y")
                                 ->assert_is_op_nth_input(NNADAPTER_MAT_MUL, 3)
                                 ->assert_is_persistable_var()
                                 ->AsInput();
  auto* malmul_out = VarNode("matmul_out")->AsOutput();
  auto* add_input_y = VarNode("add_input_y")
                          ->assert_is_op_nth_input(NNADAPTER_ADD, 1)
                          ->assert_is_persistable_var()
                          ->AsInput();
  auto* add_input_fuse_code = VarNode("add_input_fuse_code")
                                  ->assert_is_op_nth_input(NNADAPTER_ADD, 2)
                                  ->assert_is_persistable_var()
                                  ->AsInput();
  auto* add_out = VarNode("add_out")->AsOutput();
  // create topology.
  std::vector<PMNode*> mul_inputs{
      matmul_input_x, matmul_input_y, matmul_transpose_x, matmul_transpose_y};
  std::vector<PMNode*> add_inputs{malmul_out, add_input_y, add_input_fuse_code};
  mul_inputs >> *matmul >> *malmul_out;
  add_inputs >> *add >> *add_out;
  // Some op specialities.
  matmul->AsIntermediate();
  add->AsIntermediate();
  matmul_transpose_x->AsIntermediate();
  matmul_transpose_y->AsIntermediate();
  malmul_out->AsIntermediate();
}

void MatMulAddFuser::InsertNewNode(Graph* graph,
                                   core::Model* model,
                                   const key2nodes_t& matched) {
  auto* fc_operation = AddOperation(model);
  auto matmul_op = matched.at("matmul")->operation();
  auto matmul_op_inlinks = matmul_op->input_operands;
  auto matmul_op_outlinks = matmul_op->output_operands;
  auto add_op = matched.at("add")->operation();
  auto add_op_inlinks = add_op->input_operands;
  auto add_op_outlinks = add_op->output_operands;

  auto x_operand = matmul_op_inlinks[0];
  auto y_operand = matmul_op_inlinks[1];
  auto transpose_x_operand = matmul_op_inlinks[2];
  auto transpose_y_operand = matmul_op_inlinks[3];
  auto bias_operand = add_op_inlinks[1];
  auto fuse_code_operand = add_op_inlinks[2];

  if (*reinterpret_cast<bool*>(transpose_x_operand->buffer)) {
    TransposeOperand(x_operand, std::vector<int32_t>({1, 0}));
  }
  if (!*reinterpret_cast<bool*>(transpose_y_operand->buffer)) {
    TransposeOperand(y_operand, std::vector<int32_t>({1, 0}));
  }

  fc_operation->type = NNADAPTER_FULLY_CONNECTED;
  fc_operation->input_operands = {
      x_operand, y_operand, bias_operand, fuse_code_operand};
  fc_operation->output_operands = add_op_outlinks;

  // Clean operand or operation for model
  RemoveOperand(model, transpose_x_operand);
  RemoveOperand(model, transpose_y_operand);
  RemoveOperand(model, matmul_op_outlinks[0]);
  RemoveOperation(model, matmul_op);

  // Create new fc op node for graph
  auto* new_op_node = graph->GraphCreateInstructNode(*fc_operation);
  IR_NODE_LINK_TO(matched.at("matmul_input_x"), new_op_node);
  IR_NODE_LINK_TO(matched.at("matmul_input_y"), new_op_node);
  IR_NODE_LINK_TO(matched.at("add_input_y"), new_op_node);
  IR_NODE_LINK_TO(matched.at("add_input_fuse_code"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("add_out"));
}

}  // namespace nnadapter
