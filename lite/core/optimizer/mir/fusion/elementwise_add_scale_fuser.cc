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

#include "lite/core/optimizer/mir/fusion/elementwise_add_scale_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ElementwiseScaleFuser::BuildPattern() {
  auto* x = VarNode("x")->assert_is_op_input(eltwise_type_, "X")->AsInput();
  auto* y = VarNode("y")->assert_is_op_input(eltwise_type_, "Y")->AsInput();

  // create op nodes
  auto* elt = OpNode("elt", eltwise_type_)
                  ->assert_is_op(eltwise_type_)
                  ->AsIntermediate();
  auto* scale =
      OpNode("scale", "scale")->assert_is_op("scale")->AsIntermediate();

  // create intermediate nodes
  auto* elt_out = VarNode("add_out")
                      ->assert_is_op_output(eltwise_type_, "Out")
                      ->assert_is_op_input("scale", "X")
                      ->AsIntermediate();

  // create output node
  auto* out =
      VarNode("output")->assert_is_op_output("scale", "Out")->AsOutput();

  // create topology.
  std::vector<PMNode*> elt_inputs{x, y};
  elt_inputs >> *elt >> *elt_out;
  *elt_out >> *scale >> *out;
}

void ElementwiseScaleFuser::InsertNewNode(SSAGraph* graph,
                                          const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  std::shared_ptr<lite::OpLite> op;
  if (eltwise_type_ == "elementwise_mul") {
    op = LiteOpRegistry::Global().Create("elementwise_mul");
  } else {
    LOG(FATAL) << "not supported elementwise_type: " << eltwise_type_;
  }

  auto old_op = matched.at("elt")->stmt()->op();
  auto* scope = old_op->scope();
  auto& valid_places = old_op->valid_places();
  op->Attach(op_desc, scope);
  auto* new_op_node = graph->GraphCreateInstructNode(op, valid_places);

  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(matched.at("y"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("output"));
}

cpp::OpDesc ElementwiseScaleFuser::GenOpDesc(const key2nodes_t& matched) {
  auto op_desc = *matched.at("elt")->stmt()->op_info();
  op_desc.SetOutput("Out", {matched.at("output")->arg()->name});
  auto* scale_op_desc = matched.at("scale")->stmt()->op_info();
  op_desc.SetAttr("fuse_scale", true);
  op_desc.SetAttr<std::string>("activation_type", "");
  if (scale_op_desc->HasAttr("activation_type")) {
    op_desc.SetAttr("activation_type",
                    scale_op_desc->GetAttr<std::string>("activation_type"));
  }
  float scale = scale_op_desc->GetAttr<float>("scale");
  op_desc.SetAttr("scale", scale);
  float bias = scale_op_desc->GetAttr<float>("bias");
  op_desc.SetAttr("bias", bias);
  op_desc.SetAttr("alpha",
                  6.f);  // default value for placeholder  of element+scale pass
  if (scale_op_desc->HasAttr("alpha")) {
    float alpha = scale_op_desc->GetAttr<float>("alpha");
    op_desc.SetAttr("alpha", alpha);
  }
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
