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

#include "lite/core/optimizer/mir/fusion/transpose_softmax_transpose_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void TransposeSoftmaxTransposeFuser::BuildPattern() {
  // create nodes.
  auto* x1 = VarNode("x1")->assert_is_op_input(transpose_type_, "X");
  auto* y1 = VarNode("y1")->assert_is_op_output(transpose_type_, "Out");
  auto* y2 = VarNode("y2")->assert_is_op_output(softmax_type_, "Out");
  auto* out = VarNode("out")->assert_is_op_output(transpose_type_, "Out");

  PMNode* xshape1 = nullptr;
  PMNode* xshape2 = nullptr;
  if (transpose_type_ == "transpose2") {
    xshape1 =
        VarNode("xshape1")->assert_is_op_output(transpose_type_, "XShape");
    xshape2 =
        VarNode("xshape2")->assert_is_op_output(transpose_type_, "XShape");
  }

  auto* transpose1 =
      OpNode("transpose1", transpose_type_)->assert_is_op(transpose_type_);

  auto* softmax = OpNode("softmax", softmax_type_)
                      ->assert_op_attr_satisfied<int>(
                          "axis", [](int attr) { return attr == -1; });

  auto* transpose2 =
      OpNode("transpose2", transpose_type_)->assert_is_op(transpose_type_);

  // create topology.
  *x1 >> *transpose1 >> *y1 >> *softmax >> *y2 >> *transpose2 >> *out;
  if (xshape1) *transpose1 >> *xshape1;
  if (xshape2) *transpose2 >> *xshape2;

  // nodes to remove
  y1->AsIntermediate();
  y2->AsIntermediate();
  if (xshape1) xshape1->AsIntermediate();
  if (xshape2) xshape2->AsIntermediate();
  transpose1->AsIntermediate();
  softmax->AsIntermediate();
  transpose2->AsIntermediate();
}

void TransposeSoftmaxTransposeFuser::InsertNewNode(SSAGraph* graph,
                                                   const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto softmax_op = LiteOpRegistry::Global().Create(softmax_type_);
  auto softmax_old = matched.at("softmax")->stmt()->op();
  auto* scope = softmax_old->scope();
  auto& valid_places = softmax_old->valid_places();
  softmax_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(softmax_op, valid_places);

  IR_NODE_LINK_TO(matched.at("x1"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("out"));
}

cpp::OpDesc TransposeSoftmaxTransposeFuser::GenOpDesc(
    const key2nodes_t& matched) {
  cpp::OpDesc op_desc;
  op_desc.SetType("softmax");
  op_desc.SetInput("X", {matched.at("x1")->arg()->name});
  op_desc.SetOutput("Out", {matched.at("out")->arg()->name});
  op_desc.SetAttr("axis",
                  *(matched.at("transpose1")
                        ->stmt()
                        ->op_info()
                        ->GetAttr<std::vector<int>>("axis")
                        .end() -
                    1));

  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
