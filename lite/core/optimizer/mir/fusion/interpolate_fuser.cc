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

#include "lite/core/optimizer/mir/fusion/interpolate_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void InterpolateFuser::BuildPattern() {
  // type1             fill_constant -->
  // x --> shape --> slice --> cast --> elementwise_mul --> interpolate
  //   `-------------------------------------------------->
  auto* x = VarNode("x");
  auto* shape = OpNode("shape", "shape")->AsIntermediate();
  auto* shape_out = VarNode("shape_out")->AsIntermediate();
  auto* slice = OpNode("slice", "slice")
                    ->assert_op_attr_satisfied<std::vector<int>>(
                        "axes",
                        [](const std::vector<int>& attr) {
                          return attr.size() == 1 && attr[0] == 0;
                        })
                    ->assert_op_attr_satisfied<std::vector<int>>(
                        "starts",
                        [](const std::vector<int>& attr) {
                          return attr.size() == 1 && attr[0] == 2;
                        })
                    ->assert_op_attr_satisfied<std::vector<int>>(
                        "ends",
                        [](const std::vector<int>& attr) {
                          return attr.size() == 1 && attr[0] == 4;
                        })
                    ->AsIntermediate();
  auto* slice_out = VarNode("slice_out")->AsIntermediate();
  auto* cast = OpNode("cast", "cast")->AsIntermediate();
  auto* cast_out = VarNode("cast_out")->AsIntermediate();
  auto* fill_constant =
      OpNode("fill_constant", "fill_constant")->AsIntermediate();
  auto* fill_constant_out = VarNode("fill_constant_out")->AsIntermediate();
  auto* elementwise_mul =
      OpNode("elementwise_mul", "elementwise_mul")
          ->assert_op_attr_satisfied<int>(
              "axis", [](int attr) { return attr == -1 || attr == 0; })
          ->AsIntermediate();
  auto* elementwise_mul_out = VarNode("elementwise_mul_out")->AsIntermediate();
  auto* interpolate = OpNode("interpolate", interp_type_)->AsIntermediate();
  auto* interpolate_out = VarNode("interpolate_out");

  // create topology.
  *x >> *shape >> *shape_out >> *slice >> *slice_out >> *cast >> *cast_out >>
      *elementwise_mul >> *elementwise_mul_out >> *interpolate >>
      *interpolate_out;
  *fill_constant >> *fill_constant_out >> *elementwise_mul;
  *x >> *interpolate;
}

void InterpolateFuser::InsertNewNode(SSAGraph* graph,
                                     const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto interp_op = LiteOpRegistry::Global().Create(interp_type_);
  auto interp_old = matched.at("interpolate")->stmt()->op();
  auto* scope = interp_old->scope();
  auto& valid_places = interp_old->valid_places();
  interp_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(interp_op, valid_places);

  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("interpolate_out"));
}

cpp::OpDesc InterpolateFuser::GenOpDesc(const key2nodes_t& matched) {
  auto op_desc = *matched.at("interpolate")->stmt()->op_info();
  op_desc.SetInput("OutSize", {});
  op_desc.SetAttr(
      "scale",
      matched.at("fill_constant")->stmt()->op_info()->GetAttr<float>("value"));
  return op_desc;
}

void InterpolateFuser2::BuildPattern() {
  // type2 x --> shape --> slice --> cast --> scale --> interpolate
  //        `---------------------------------------->
  auto* x = VarNode("x");
  auto* shape = OpNode("shape", "shape")->AsIntermediate();
  auto* shape_out = VarNode("shape_out")->AsIntermediate();
  auto* slice = OpNode("slice", "slice")
                    ->assert_op_attr_satisfied<std::vector<int>>(
                        "axes",
                        [](const std::vector<int>& attr) {
                          return attr.size() == 1 && attr[0] == 0;
                        })
                    ->assert_op_attr_satisfied<std::vector<int>>(
                        "starts",
                        [](const std::vector<int>& attr) {
                          return attr.size() == 1 && attr[0] == 2;
                        })
                    ->assert_op_attr_satisfied<std::vector<int>>(
                        "ends",
                        [](const std::vector<int>& attr) {
                          return attr.size() == 1 && attr[0] == 4;
                        })
                    ->AsIntermediate();
  auto* slice_out = VarNode("slice_out")->AsIntermediate();
  auto* cast = OpNode("cast", "cast")->AsIntermediate();
  auto* cast_out = VarNode("cast_out")->AsIntermediate();
  auto* scale = OpNode("scale", "scale")->AsIntermediate();
  auto* scale_out = VarNode("scale_out")->AsIntermediate();
  auto* interpolate = OpNode("interpolate", interp_type_)->AsIntermediate();
  auto* interpolate_out = VarNode("interpolate_out");

  // create topology.
  *x >> *shape >> *shape_out >> *slice >> *slice_out >> *cast >> *cast_out >>
      *scale >> *scale_out >> *interpolate >> *interpolate_out;
  *x >> *interpolate;
}

void InterpolateFuser2::InsertNewNode(SSAGraph* graph,
                                      const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto interp_op = LiteOpRegistry::Global().Create(interp_type_);
  auto interp_old = matched.at("interpolate")->stmt()->op();
  auto* scope = interp_old->scope();
  auto& valid_places = interp_old->valid_places();
  interp_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(interp_op, valid_places);

  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("interpolate_out"));
}

cpp::OpDesc InterpolateFuser2::GenOpDesc(const key2nodes_t& matched) {
  auto op_desc = *matched.at("interpolate")->stmt()->op_info();
  op_desc.SetInput("OutSize", {});
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
