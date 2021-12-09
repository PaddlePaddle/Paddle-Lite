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

#include "lite/core/optimizer/mir/fusion/fill_range_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void FillRangeFuser::BuildPattern() {
  // Fill op
  auto* fill_range_start = OpNode("fill_range_start", "fill_constant");
  auto* fill_range_end = OpNode("fill_range_end", "fill_constant");
  auto* fill_range_step = OpNode("fill_range_step", "fill_constant");

  // Range op
  auto* range = OpNode("range", "range");
  auto* start =
      VarNode("start")->assert_is_op_input("range", "Start")->AsInput();
  auto* end = VarNode("end")->assert_is_op_input("range", "End")->AsInput();
  auto* step = VarNode("step")->assert_is_op_input("range", "Step")->AsInput();
  auto* range_out =
      VarNode("range_out")->assert_is_op_output("range", "Out")->AsOutput();

  *fill_range_start >> *start >> *range;
  *fill_range_end >> *end >> *range;
  *fill_range_step >> *step >> *range;
  *range >> *range_out;

  // Some op specialities.
  fill_range_start->AsIntermediate();
  fill_range_end->AsIntermediate();
  fill_range_step->AsIntermediate();
  range->AsIntermediate();
}

void FillRangeFuser::InsertNewNode(SSAGraph* graph,
                                   const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto range_op = LiteOpRegistry::Global().Create("range");
  auto range = matched.at("range")->stmt()->op();
  auto* scope = range->scope();
  auto& valid_places = range->valid_places();
  range_op->Attach(op_desc, scope);

  // Create new range op node
  auto* new_op_node = graph->GraphCreateInstructNode(range_op, valid_places);
  auto new_op = new_op_node->stmt()->op();

  IR_NODE_LINK_TO(matched.at("start"), new_op_node);
  IR_NODE_LINK_TO(matched.at("end"), new_op_node);
  IR_NODE_LINK_TO(matched.at("step"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("range_out"));
}

cpp::OpDesc FillRangeFuser::GenOpDesc(const key2nodes_t& matched) {
  cpp::OpDesc fill_range_start_op_desc =
      *matched.at("fill_range_start")->stmt()->op_info();
  cpp::OpDesc fill_range_end_op_desc =
      *matched.at("fill_range_end")->stmt()->op_info();
  cpp::OpDesc fill_range_step_op_desc =
      *matched.at("fill_range_step")->stmt()->op_info();

  // TODO(shentanyue) supported later
  if ((fill_range_start_op_desc.HasInput("ValueTensor") &&
       fill_range_start_op_desc.Input("ValueTensor").size() > 0) ||
      (fill_range_start_op_desc.HasInput("str_value") &&
       !fill_range_start_op_desc.GetAttr<std::string>("str_value").empty())) {
    LOG(FATAL) << "Unsupported for ValueTensor input or str_value input for "
                  "fill_contant op.";
  }
  if ((fill_range_end_op_desc.HasInput("ValueTensor") &&
       fill_range_end_op_desc.Input("ValueTensor").size() > 0) ||
      (fill_range_end_op_desc.HasInput("str_value") &&
       !fill_range_end_op_desc.GetAttr<std::string>("str_value").empty())) {
    LOG(FATAL) << "Unsupported for ValueTensor input or str_value input for "
                  "fill_contant op.";
  }
  if ((fill_range_step_op_desc.HasInput("ValueTensor") &&
       fill_range_step_op_desc.Input("ValueTensor").size() > 0) ||
      (fill_range_step_op_desc.HasInput("str_value") &&
       !fill_range_step_op_desc.GetAttr<std::string>("str_value").empty())) {
    LOG(FATAL) << "Unsupported for ValueTensor input or str_value input for "
                  "fill_contant op.";
  }

  auto start = fill_range_start_op_desc.GetAttr<float>("value");
  auto end = fill_range_end_op_desc.GetAttr<float>("value");
  auto step = fill_range_step_op_desc.GetAttr<float>("value");

  auto range_instruct = matched.at("range")->stmt();
  auto range_op_desc = range_instruct->mutable_op_info();
  auto range = range_instruct->op();
  auto* range_scope = range->scope();

  auto range_start_var_name = matched.at("start")->arg()->name;
  auto range_start_tensor =
      range_scope->FindVar(range_start_var_name)->GetMutable<lite::Tensor>();
  auto* range_start_data = range_start_tensor->mutable_data<float>();

  auto range_end_var_name = matched.at("end")->arg()->name;
  auto range_end_tensor =
      range_scope->FindVar(range_end_var_name)->GetMutable<lite::Tensor>();
  auto* range_end_data = range_end_tensor->mutable_data<float>();

  auto range_step_var_name = matched.at("step")->arg()->name;
  auto range_step_tensor =
      range_scope->FindVar(range_step_var_name)->GetMutable<lite::Tensor>();
  auto* range_step_data = range_step_tensor->mutable_data<float>();

  if (range_start_tensor->data_size() != 1 ||
      range_end_tensor->data_size() != 1 ||
      range_step_tensor->data_size() != 1) {
    LOG(FATAL) << "Unsupported for tensor var";
  }
  range_start_data[0] = start;
  range_end_data[0] = end;
  range_step_data[0] = step;

  range_start_tensor->set_persistable(true);
  range_end_tensor->set_persistable(true);
  range_step_tensor->set_persistable(true);
  matched.at("start")->arg()->is_weight = true;
  matched.at("end")->arg()->is_weight = true;
  matched.at("step")->arg()->is_weight = true;

  range_op_desc->SetType("range");
  range_op_desc->SetInput("Start", {range_start_var_name});
  range_op_desc->SetInput("End", {range_end_var_name});
  range_op_desc->SetInput("Step", {range_step_var_name});
  range_op_desc->SetOutput("Out", {matched.at("range_out")->arg()->name});
  return *range_op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
