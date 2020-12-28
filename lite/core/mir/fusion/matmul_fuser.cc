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

#include "lite/core/mir/fusion/matmul_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void MatmulFuser::BuildPattern() {
  // create nodes.
  auto* x = VarNode("x")->assert_is_op_input("matmul", "X");
  auto* y = VarNode("y")->assert_is_op_input("matmul", "Y");
  auto* matmul = OpNode("matmul", "matmul");
  auto* matmul_out = VarNode("Out");

  // create topology.
  std::vector<PMNode*> matmul_inputs{x, y};
  matmul_inputs >> *matmul >> *matmul_out;

  // Some op specialities.
  matmul->AsIntermediate();
}

void MatmulFuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {
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

cpp::OpDesc MatmulFuser::GenOpDesc(const key2nodes_t& matched) {
  auto op_desc = *matched.at("matmul")->stmt()->op_info();
  auto* scope = matched.at("matmul")->stmt()->op()->scope();

  // Get the input scale from matmul

  auto input_x_name = op_desc.Input("X").front();
  auto input_y_name = op_desc.Input("Y").front();
  bool transpose_X = op_desc.GetAttr<bool>("transpose_X");
  bool transpose_Y = op_desc.GetAttr<bool>("transpose_Y");
  float alpha = op_desc.GetAttr<float>("alpha");

  auto x_shape = scope->FindVar(input_x_name)->Get<lite::Tensor>().dims();
  auto y_shape = scope->FindVar(input_y_name)->Get<lite::Tensor>().dims();
  size_t x_rank = x_shape.size();
  size_t y_rank = y_shape.size();

  if (!transpose_X && !transpose_Y && std::abs(alpha - 1.0) < 1e-5 &&
      x_rank == 2 && y_rank == 2) {
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
