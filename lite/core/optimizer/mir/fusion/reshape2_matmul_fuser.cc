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

#include "lite/core/optimizer/mir/fusion/reshape2_matmul_fuser.h"
#include <cmath>
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void Reshape2MatmulFuser::BuildPattern() {
  // Teller function about reshape2's inputs:
  //    The reshape2 op must satisfy the following conditions:
  //    1. reshape2 has one input node, which means it don't
  //       have Shape or ShapeTensor input
  //    2. the rank of input X is 4 and the last two dims of input X is 1
  auto reshape2_inputs_teller = [](const Node* node) -> bool {
    auto reshape2_op_desc = *const_cast<Node*>(node)->stmt()->op_info();
    auto reshape2_input_x_name = reshape2_op_desc.Input("X").front();
    auto* scope = const_cast<Node*>(node)->stmt()->op()->scope();
    auto reshape2_in_x_shape =
        scope->FindVar(reshape2_input_x_name)->Get<lite::Tensor>().dims();
    size_t reshape2_in_x_rank = reshape2_in_x_shape.size();

    return (reshape2_in_x_rank == 4 && reshape2_in_x_shape[2] == 1 &&
            reshape2_in_x_shape[3] == 1);
  };

  // create nodes.
  auto* reshape2_in_x = VarNode("x")->assert_is_op_input("reshape2", "X");
  auto* reshape2_op =
      OpNode("reshape2", "reshape2")
          ->assert_op_attr_satisfied<std::vector<int>>(
              "shape",
              [](const std::vector<int>& attr) { return attr.size() == 2; })
          ->assert_node_satisfied(reshape2_inputs_teller);
  auto* reshape2_out = VarNode("reshape2_out");
  auto* reshape2_xshape = VarNode("reshape2_xshape");

  // Teller function about matmul's inputs:
  //          the rank of input X and Y should 2
  auto matmul_inputs_teller = [](const Node* node) -> bool {
    auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
    auto input_x_name = op_desc.Input("X").front();
    auto input_y_name = op_desc.Input("Y").front();
    auto* scope = const_cast<Node*>(node)->AsStmt().op()->scope();

    auto x_shape = scope->FindVar(input_x_name)->Get<lite::Tensor>().dims();
    auto y_shape = scope->FindVar(input_y_name)->Get<lite::Tensor>().dims();
    size_t x_rank = x_shape.size();
    size_t y_rank = y_shape.size();

    return (x_rank == 2 && y_rank == 2);
  };

  // create nodes.
  auto* matmul_y = VarNode("y")->assert_is_op_input("matmul", "Y");
  /*
   * The mul op must satisfy the following conditions:
   * 1. the transpose_X and transpose_Y attrs are false
   * 2. the alpha attr is 1.0
   * 3. the rank of input X and Y is 2
   */
  auto* matmul_op =
      OpNode("matmul", "matmul")
          ->assert_op_attr<bool>("transpose_X", false)
          ->assert_op_attr<bool>("transpose_Y", false)
          ->assert_op_attr_satisfied<float>(
              "alpha",
              [](float attr) { return (std::fabs(attr - 1.0) < 1e-5); })
          ->assert_node_satisfied(matmul_inputs_teller);
  auto* matmul_out = VarNode("Out");
  std::vector<PMNode*> reshape2_inputs{reshape2_in_x};
  std::vector<PMNode*> reshape2_outputs{reshape2_out, reshape2_xshape};
  std::vector<PMNode*> matmul_inputs{reshape2_out, matmul_y};

  // Create topology: reshape2 --> matmul
  reshape2_inputs >> *reshape2_op >> reshape2_outputs;
  matmul_inputs >> *matmul_op >> *matmul_out;

  // reshape2_op, matmul_op and reshape2_out will be removed after this fusion.
  reshape2_out->AsIntermediate();
  reshape2_op->AsIntermediate();
  matmul_op->AsIntermediate();
}

void Reshape2MatmulFuser::InsertNewNode(SSAGraph* graph,
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

cpp::OpDesc Reshape2MatmulFuser::GenOpDesc(const key2nodes_t& matched) {
  auto op_desc = *matched.at("matmul")->stmt()->op_info();
  op_desc.mutable_inputs()->clear();
  op_desc.mutable_outputs()->clear();
  op_desc.SetType("mul");
  op_desc.SetInput("X", {matched.at("x")->arg()->name});
  op_desc.SetInput("Y", {matched.at("y")->arg()->name});
  op_desc.SetAttr<int>("x_num_col_dims", 1);
  op_desc.SetAttr<int>("y_num_col_dims", 1);
  op_desc.SetOutput("Out", {matched.at("Out")->arg()->name});

  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
