// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/optimizer/mir/fusion/elementwise_add_reshape_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

/* fuse elementwise_add block in resnet50-like model            */
/* For example:                                                 */
/* sub block                                                    */
/*                     in_Input                                 */
/*                       |                                      */
/*                       |                                      */
/*                     conv2d                                   */
/*                       |    reshape2                          */
/*                       |    /                                 */
/*                       |   /                                  */
/*                  elementwise_add                             */
/*                       |                                      */
/*                       |                                      */
/*                     out_Out                                  */
/*                                                              */
/* After the pass is applied:                                   */
/*                     in_Input                                 */
/*                       |                                      */
/*                       |                                      */
/*                     conv2d                                   */
/*                       |                                      */
/*                       |                                      */
/*                       |                                      */
/*                  elementwise_add                             */
/*                       |                                      */
/*                       |                                      */
/*                     out_Out                                  */
/*                                                              */

void ElementwiseReshapeFuser::BuildPattern() {
  auto* reshape_in =
      VarNode("reshape_in")->assert_is_op_input(reshape_type_, "X")->AsInput();

  auto* x = VarNode("x")
                ->assert_is_op_input(eltwise_type_, "X")
                ->assert_is_op_output("conv2d", "Output")
                ->AsInput();

  // create intermediate nodes
  auto* y = VarNode("y")
                ->assert_is_op_output(reshape_type_, "Out")
                ->assert_is_op_input(eltwise_type_, "Y")
                ->AsIntermediate();
  auto* reshape_xshape = VarNode("reshape_xshape");

  // create op nodes
  auto* reshape = OpNode("reshape", reshape_type_)
                      ->assert_is_op(reshape_type_)
                      ->AsIntermediate();
  auto* elt = OpNode("elt", eltwise_type_)
                  ->assert_is_op(eltwise_type_)
                  ->AsIntermediate();  // todo

  // create output node
  auto* out =
      VarNode("output")->assert_is_op_output(eltwise_type_, "Out")->AsOutput();

  // create topology.
  std::vector<PMNode*> reshape_outputs{y, reshape_xshape};
  std::vector<PMNode*> elt_inputs{x, y};
  *reshape_in >> *reshape >> reshape_outputs;
  elt_inputs >> *elt >> *out;
}

void ElementwiseReshapeFuser::InsertNewNode(SSAGraph* graph,
                                            const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  std::shared_ptr<lite::OpLite> op;
  if (eltwise_type_ == "elementwise_add") {
    op = LiteOpRegistry::Global().Create("elementwise_add");  // todo
  } else {
    LOG(FATAL) << "not supported elementwise_type: " << eltwise_type_;
  }

  auto old_op = matched.at("elt")->stmt()->op();
  auto* scope = old_op->scope();

  auto filter_name = matched.at("reshape_in")->arg()->name;

  op_desc.SetInput("Y", {filter_name});

  auto& valid_places = old_op->valid_places();
  op->Attach(op_desc, scope);
  auto* new_op_node = graph->GraphCreateInstructNode(op, valid_places);

  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(matched.at("reshape_in"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("output"));
}

cpp::OpDesc ElementwiseReshapeFuser::GenOpDesc(const key2nodes_t& matched) {
  auto op_desc = *matched.at("elt")->stmt()->op_info();
  if (eltwise_type_ == "elementwise_add") {
    op_desc.SetType("elementwise_add");
  } else {
    LOG(FATAL) << "not supported elementwise_type: " << eltwise_type_;
  }
  op_desc.SetOutput("Out", {matched.at("output")->arg()->name});
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
