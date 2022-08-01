// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/optimizer/mir/fusion/keepdims_convert_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void KeepdimsConvertFuser::BuildPattern() {
  // create input nodes
  auto* input = VarNode("input")->assert_is_op_input(op_type_, "X")->AsInput();

  // create intermediate nodes

  // create op nodes
  auto attr_names = attr_names_;
  auto op_teller = [&](const Node* node) -> bool {
    const std::vector<std::string> attr_names{"keep_dim", "keepdims"};
    // Convert false to true when the above attribute exists and it's false.
    // Note the attribute is false by default when the attribute doesn't exist.
    auto* op_desc = const_cast<Node*>(node)->AsStmt().op_info();
    for (auto attr_name : attr_names) {
      if (op_desc->HasAttr(attr_name)) {
        if (op_desc->GetAttr<bool>(attr_name)) {
          return false;
        }
      }
    }
    return true;
  };

  auto* op = OpNode("op", op_type_)
                 ->assert_is_op(op_type_)
                 ->assert_node_satisfied(op_teller);

  // create output node
  auto* output =
      VarNode("output")->assert_is_op_output(op_type_, "Out")->AsOutput();

  // create topology
  *input >> *op >> *output;
}

void KeepdimsConvertFuser::InsertNewNode(SSAGraph* graph,
                                         const key2nodes_t& matched) {
  auto* inst = matched.at("op")->stmt();
  auto new_op_desc = GenOpDesc(matched);
  auto new_op = LiteOpRegistry::Global().Create("reshape");
  auto op = inst->op();
  auto* scope = op->scope();
  auto& valid_places = op->valid_places();
  new_op->Attach(new_op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(new_op, valid_places);

  auto outlinks_num = matched.at("op")->outlinks.size();
  CHECK_EQ(outlinks_num, 1L) << "outlinks num should be 1, but got "
                             << outlinks_num;
  auto* in = matched.at("op")->outlinks.front();
  auto* new_input_arg = graph->NewArgumentNode(new_input_name_);
  const Type& from = *in->AsArg().type;
  new_input_arg->AsArg().type =
      LiteType::GetTensorTy(from.target(), from.precision(), from.layout());

  // Set keepdims/keep_dim attribute to true
  // Update Out arg name
  auto op_desc = inst->mutable_op_info();
  for (auto attr_name : attr_names_) {
    op_desc->SetAttr(attr_name, true);
    op_desc->SetOutput("Out", {new_input_name_});
  }

  IR_NODE_LINK_TO(matched.at("op"), new_input_arg);
  IR_NODE_LINK_TO(new_input_arg, new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("output"));
  // Remove the old link
  RemoveDirectedLink(matched.at("op"), matched.at("output"));
}

cpp::OpDesc KeepdimsConvertFuser::GenOpDesc(const key2nodes_t& matched) {
  auto* inst = matched.at("op")->stmt();

  // Create the new var manually
  auto* in = matched.at("op")->outlinks.front();
  new_input_name_ = string_format("%s/trans", in->AsArg().name.c_str());
  inst->op()->scope()->Var(new_input_name_);

  cpp::OpDesc op_desc;
  op_desc.SetType("reshape");
  op_desc.SetInput("X", {new_input_name_});
  op_desc.SetOutput("Out", {matched.at("output")->arg()->name});
  op_desc.SetAttr("shape", GetTensorDims(inst));
  return op_desc;
}

std::vector<int> KeepdimsConvertFuser::GetTensorDims(const Node::Stmt* inst) {
  const auto op = inst->op();
  const auto* op_info = inst->op_info();
  auto var_names = op_info->output_names();
  CHECK_EQ(var_names.size(), 1);
  std::string var_name = var_names[0];

  auto* scope = op->scope();
  auto* var = scope->FindVar(var_name);
  if (var == nullptr) {
    LOG(FATAL) << "var is nullptr! var_name: " << var_name;
  }

  const auto& tensor = var->Get<Tensor>();
  VLOG(4) << "tensor dims: " << tensor.dims();
  std::vector<int> dims;
  // Out dims may be empty. For example, argmax's in dims{3}, keepdims=false,
  // axis=0.
  // Set out dims manually.
  if (tensor.dims().empty()) {
    dims.push_back(1);
  } else {
    for (auto iter : tensor.dims().Vectorize()) {
      dims.push_back(iter);
    }
  }
  return dims;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
