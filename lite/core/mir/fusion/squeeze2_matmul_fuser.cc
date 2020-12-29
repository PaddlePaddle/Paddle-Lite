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

#include "lite/core/mir/fusion/squeeze2_matmul_fuser.h"
#include <cmath>
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void Squeeze2MatmulFuser::BuildPattern() {
  // create nodes.
  auto* squeeze2_in_x = VarNode("x")->assert_is_op_input("squeeze2", "X");
  auto* squeeze2_op = OpNode("squeeze2", "squeeze2");
  auto* squeeze2_out = VarNode("squeeze2_out");
  auto* squeeze2_xshape = VarNode("squeeze2_xshape");

  auto* matmul_y = VarNode("y")->assert_is_op_input("matmul", "Y");
  auto* matmul_op = OpNode("matmul", "matmul");
  auto* matmul_out = VarNode("Out");

  // create topology.
  std::vector<PMNode*> squeeze2_inputs{squeeze2_in_x};
  std::vector<PMNode*> squeeze2_outputs{squeeze2_out, squeeze2_xshape};
  std::vector<PMNode*> matmul_inputs{squeeze2_out, matmul_y};
  squeeze2_inputs >> *squeeze2_op >> squeeze2_outputs;

  // Some op specialities.
  squeeze2_op->AsIntermediate();
  squeeze2_out->AsIntermediate();
  matmul_op->AsIntermediate();

  matmul_inputs >> *matmul_op >> *matmul_out;
}

void Squeeze2MatmulFuser::InsertNewNode(SSAGraph* graph,
                                        const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto mul_op = LiteOpRegistry::Global().Create("mul");
  auto matmul = matched.at("matmul")->stmt()->op();
  auto* scope = matmul->scope();
  auto& valid_places = matmul->valid_places();
  mul_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(mul_op, valid_places);

  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(matched.at("y"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("Out"));
}

cpp::OpDesc Squeeze2MatmulFuser::GenOpDesc(const key2nodes_t& matched) {
  bool trigger_flag = true;

  auto squeeze2_op_desc = *matched.at("squeeze2")->stmt()->op_info();
  auto squeeze2_input_x_name = squeeze2_op_desc.Input("X").front();
  auto* scope = matched.at("squeeze2")->stmt()->op()->scope();
  size_t squeeze2_in_x_rank =
      scope->FindVar(squeeze2_input_x_name)->Get<lite::Tensor>().dims().size();
  std::vector<int> squeeze2_op_axes =
      squeeze2_op_desc.GetAttr<std::vector<int>>("axes");

  trigger_flag = trigger_flag && squeeze2_in_x_rank == 4 &&
                 squeeze2_op_axes == std::vector<int>{2, 3};

  // Get the input scale from matmul
  auto op_desc = *matched.at("matmul")->stmt()->op_info();
  auto input_x_name = op_desc.Input("X").front();
  auto input_y_name = op_desc.Input("Y").front();
  bool transpose_X = op_desc.GetAttr<bool>("transpose_X");
  bool transpose_Y = op_desc.GetAttr<bool>("transpose_Y");
  float alpha = op_desc.GetAttr<float>("alpha");

  auto x_shape = scope->FindVar(input_x_name)->Get<lite::Tensor>().dims();
  auto y_shape = scope->FindVar(input_y_name)->Get<lite::Tensor>().dims();
  size_t matmul_in_x_rank = x_shape.size();
  size_t matmul_in_y_rank = y_shape.size();

  trigger_flag = trigger_flag && !transpose_X && !transpose_Y &&
                 std::fabs(alpha - 1.0) < 1e-5 && matmul_in_x_rank == 2 &&
                 matmul_in_y_rank == 2;

  if (trigger_flag) {
    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();
    op_desc.SetType("mul");
    op_desc.SetInput("X", {matched.at("x")->arg()->name});
    op_desc.SetInput("Y", {matched.at("y")->arg()->name});
    op_desc.SetAttr<int>("x_num_col_dims", 1);
    op_desc.SetAttr<int>("y_num_col_dims", 1);
    op_desc.SetOutput("Out", {matched.at("Out")->arg()->name});
  }

  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
