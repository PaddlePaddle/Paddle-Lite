// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/optimizer/mir/fusion/flatten_fc_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void FlattenFcFuser::BuildPattern() {
  // flatten_contiguous_range
  PMNode* x = VarNode("x")
                  ->assert_is_op_input("flatten_contiguous_range", "X")
                  ->AsInput();
  PMNode* flatten_contiguous_range =
      OpNode("flatten_contiguous_range", "flatten_contiguous_range")
          ->AsIntermediate();
  PMNode* out = VarNode("output")
                    ->assert_is_op_output("flatten_contiguous_range", "Out")
                    ->AsIntermediate();
  PMNode* xshape;
  if (has_xshape_) {
    xshape = VarNode("xshape")
                 ->assert_is_op_output("flatten_contiguous_range", "XShape")
                 ->AsIntermediate();
  }

  // fc
  // PMNode* input   = VarNode("input")->assert_is_op_input("fc",
  // "Input")->AsIntermediate();
  PMNode* weights =
      VarNode("weights")->assert_is_op_input("fc", "W")->AsInput();
  PMNode* bias = VarNode("bias")->assert_is_op_input("fc", "Bias")->AsInput();
  PMNode* fc = OpNode("fc", "fc")->AsIntermediate();
  PMNode* fc_out =
      VarNode("fc_out")->assert_is_op_output("fc", "Out")->AsOutput();

  // create topology.
  std::vector<PMNode*> fc_inputs{bias, weights, out};
  *x >> *flatten_contiguous_range >> *out;
  if (has_xshape_) {
    *flatten_contiguous_range >> *xshape;
  }
  fc_inputs >> *fc >> *fc_out;
}

void FlattenFcFuser::InsertNewNode(SSAGraph* graph,
                                   const key2nodes_t& matched) {
  auto flatten = matched.at("flatten_contiguous_range")->stmt()->op();
  auto* scope = flatten->scope();
  auto flatten_input = scope->FindVar(matched.at("x")->arg()->name);
  auto flatten_input_dims = flatten_input->Get<lite::Tensor>().dims();
  auto flatten_desc = matched.at("flatten_contiguous_range")->stmt()->op_info();
  auto start_axis = flatten_desc->GetAttr<int>("start_axis");
  auto fc_desc = matched.at("fc")->stmt()->op_info();
  auto in_mum_col_dims = fc_desc->GetAttr<int>("in_num_col_dims");
  int real_start_axis = start_axis;
  if (start_axis < 0) {
    real_start_axis = start_axis + flatten_input_dims.size();
  }
  if (in_mum_col_dims >= real_start_axis + 1) {
    nodes_.erase(nodes_.begin(), nodes_.end());
    LOG(WARNING) << "in_mum_col_dims_old >= real_start_axis + 1, fuse failed";
    return;
  }
  auto op_desc = GenOpDesc(matched);
  auto fc_op = LiteOpRegistry::Global().Create("fc");
  auto fc_old = matched.at("fc")->stmt()->op();
  auto& valid_places = fc_old->valid_places();
  fc_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(fc_op, valid_places);

  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(matched.at("weights"), new_op_node);
  IR_NODE_LINK_TO(matched.at("bias"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("fc_out"));
}

cpp::OpDesc FlattenFcFuser::GenOpDesc(const key2nodes_t& matched) {
  cpp::OpDesc op_desc = *matched.at("fc")->stmt()->op_info();
  op_desc.SetInput("Input", {matched.at("x")->arg()->name});
  op_desc.SetOutput("Out", {matched.at("fc_out")->arg()->name});
  auto in_num_col_dim = op_desc.GetAttr<int>("in_num_col_dims");
  op_desc.SetAttr("in_num_col_dims", in_num_col_dim);
  std::string op_type = "mul";
  op_desc.SetAttr("op_type", op_type);
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
