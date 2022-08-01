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

#include "lite/core/optimizer/mir/fusion/shuffle_channel_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ShuffleChannelFuser::BuildPattern() {
  // create nodes.
  auto* x1 = VarNode("x1")->assert_is_op_input(reshape_type_, "X");
  auto* y1 = VarNode("y1")->assert_is_op_output(reshape_type_, "Out");
  auto* y2 = VarNode("y2")->assert_is_op_output(transpose_type_, "Out");
  auto* out = VarNode("out")->assert_is_op_output(reshape_type_, "Out");

  PMNode* xshape1 = nullptr;
  PMNode* xshape2 = nullptr;
  PMNode* xshape3 = nullptr;
  if (reshape_type_ == "reshape2") {
    xshape1 = VarNode("xshape1")->assert_is_op_output(reshape_type_, "XShape");
    xshape3 = VarNode("xshape3")->assert_is_op_output(reshape_type_, "XShape");
  }
  if (transpose_type_ == "transpose2") {
    xshape2 =
        VarNode("xshape2")->assert_is_op_output(transpose_type_, "XShape");
  }

  auto* reshape1 = OpNode("reshape1", reshape_type_)
                       ->assert_op_attr_satisfied<std::vector<int>>(
                           "shape", [](const std::vector<int>& attr) {
                             return attr.size() >= 5 && attr[1] > 0;
                           });
  auto* transpose =
      OpNode("transpose_op", transpose_type_)
          ->assert_op_attr_satisfied<std::vector<int>>(
              "axis", [](const std::vector<int>& attr) {
                return attr.size() >= 5 && attr[1] == 2 && attr[2] == 1;
              });
  auto* reshape2 = OpNode("reshape2", reshape_type_)
                       ->assert_op_attr_satisfied<std::vector<int>>(
                           "shape", [](const std::vector<int>& attr) {
                             return attr.size() >= 4;
                           });

  // create topology.
  *x1 >> *reshape1 >> *y1 >> *transpose >> *y2 >> *reshape2 >> *out;
  if (xshape1) *reshape1 >> *xshape1;
  if (xshape2) *transpose >> *xshape2;
  if (xshape3) *reshape2 >> *xshape3;

  // Some op specialities.
  y1->AsIntermediate();
  y2->AsIntermediate();
  if (xshape1) xshape1->AsIntermediate();
  if (xshape2) xshape2->AsIntermediate();
  if (xshape3) xshape3->AsIntermediate();
  reshape1->AsIntermediate();
  transpose->AsIntermediate();
  reshape2->AsIntermediate();
}

void ShuffleChannelFuser::InsertNewNode(SSAGraph* graph,
                                        const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto shuffle_channel_op = LiteOpRegistry::Global().Create("shuffle_channel");
  auto transpose = matched.at("transpose_op")->stmt()->op();
  auto* scope = transpose->scope();
  auto& valid_places = transpose->valid_places();
  shuffle_channel_op->Attach(op_desc, scope);

  auto* new_op_node =
      graph->GraphCreateInstructNode(shuffle_channel_op, valid_places);

  IR_NODE_LINK_TO(matched.at("x1"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("out"));
}

cpp::OpDesc ShuffleChannelFuser::GenOpDesc(const key2nodes_t& matched) {
  cpp::OpDesc op_desc;
  op_desc.SetType("shuffle_channel");
  op_desc.SetInput("X", {matched.at("x1")->arg()->name});
  op_desc.SetOutput("Out", {matched.at("out")->arg()->name});
  op_desc.SetAttr("group",
                  matched.at("reshape1")
                      ->stmt()
                      ->op_info()
                      ->GetAttr<std::vector<int>>("shape")[1]);
  cpp::OpDesc reshape = *matched.at("reshape2")->stmt()->op_info();
  if (reshape.HasAttr("out_threshold")) {
    float out_threshold = reshape.GetAttr<float>("out_threshold");
    op_desc.SetAttr("out_threshold", out_threshold);
    VLOG(4) << "shuffle_channel fusion,out_threshold:" << out_threshold;
  }
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
