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

#include "lite/core/mir/cycle_removal_pass.h"
#include <memory>
#include <string>
#include <vector>
#include "lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

SelfLoopDetector::SelfLoopDetector(const std::string& op_type) {
  op_type_ = op_type;
}

void SelfLoopDetector::BuildPattern() {
  mir::PMNode* io_var = VarNode("io_var")
                            ->assert_is_op_input(op_type_)
                            ->assert_is_op_output(op_type_)
                            ->assert_is_persistable_var();
  mir::PMNode* self_loop_op = OpNode("self_loop_op", op_type_);
  *io_var >> *self_loop_op >> *io_var;
}

void SelfLoopDetector::InsertNewNode(SSAGraph* graph,
                                     const key2nodes_t& matched) {
  mir::Node* arg_node{matched.at("io_var")};
  mir::Node* op_node{matched.at("self_loop_op")};
  mir::Node::Arg* arg{arg_node->arg()};
  mir::Node::Stmt* op{op_node->stmt()};

  const std::string& var_name{arg->name};
  const std::string new_output_name{var_name + "__OUT_VAR"};

  mir::Node* new_output_node{graph->NewArgumentNode(new_output_name)};
  new_output_node->AsArg().type = arg->type;

  op->mutable_op_info()->UpdateAllOutputs(var_name, new_output_name);
  lite::Variable* var = op->op()->scope()->FindVar(var_name);
  CHECK(var);
  lite::Tensor* in_tensor = var->GetMutable<lite::Tensor>();
  op->op()
      ->scope()
      ->Var(new_output_name)
      ->GetMutable<lite::Tensor>()
      ->ShareDataWith(*in_tensor);

  RemoveDirectedLink(op_node, arg_node);
  DirectedLink(op_node, new_output_node);
}
}  // namespace fusion

void CycleRemovalPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  const std::vector<std::string> target_ops{
      "batch_norm",
      "fake_quantize_range_abs_max",
      "fake_quantize_moving_average_abs_max",
      "fake_quantize_dequantize_moving_average_abs_max"};
  for (auto op_type : target_ops) {
    fusion::SelfLoopDetector fuser(op_type);
    fuser(graph.get());
  }
}
}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(cycle_removal_pass, paddle::lite::mir::CycleRemovalPass)
    .BindTargets({TARGET(kAny)});
