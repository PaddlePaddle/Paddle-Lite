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

#include "lite/core/optimizer/mir/control_flow_op_shared_inputs_and_outputs_place_sync_pass.h"
#include <algorithm>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void CheckAndSyncTypeOfVarNode(
    Node* sub_var_node,
    const std::unordered_map<std::string, const Type*>& ref_var_types) {
  CHECK(sub_var_node->IsArg());
  auto& sub_var_name = sub_var_node->AsArg().name;
  if (ref_var_types.count(sub_var_name)) {
    sub_var_node->AsArg().type = ref_var_types.at(sub_var_name);
  }
}

void ControlFlowOpSharedInputsAndOutputsPlaceSyncPass::SetAllGraphs(
    std::vector<std::unique_ptr<mir::SSAGraph>>* graphs) {
  CHECK(graphs && !graphs->empty());
  graphs_ = graphs;
}

void ControlFlowOpSharedInputsAndOutputsPlaceSyncPass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  const std::unordered_set<std::string> control_flow_op_types = {
      "while", "conditional_block"};
  auto block_size = graphs_->size();
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto op_info = op_node->AsStmt().mutable_op_info();
    auto op_type = op_info->Type();
    if (!control_flow_op_types.count(op_type)) continue;
    int sub_block_idx = op_info->GetAttr<int32_t>("sub_block");
    CHECK_GE(sub_block_idx, 0);
    CHECK_LT(sub_block_idx, block_size);
    std::unordered_map<std::string, const Type*> ref_var_types;
    for (auto* var_node : op_node->inlinks) {
      CHECK(var_node->IsArg());
      if (var_node->inlinks.empty()) continue;
      auto& var_name = var_node->AsArg().name;
      if (!ref_var_types.count(var_name)) {
        ref_var_types.insert(std::pair<std::string, const Type*>(
            var_name, var_node->AsArg().type));
      }
    }

    // sync input var
    for (auto& sub_op_node :
         (*graphs_)[sub_block_idx]->StmtTopologicalOrder()) {
      if (!sub_op_node->IsStmt()) continue;
      for (auto* sub_var_node : sub_op_node->inlinks) {
        CheckAndSyncTypeOfVarNode(sub_var_node, ref_var_types);
      }
      for (auto* sub_var_node : sub_op_node->outlinks) {
        auto& var_name = sub_var_node->AsArg().name;
        if (!ref_var_types.count(var_name)) {
          ref_var_types.insert(std::pair<std::string, const Type*>(
              var_name, sub_var_node->AsArg().type));
        }
      }
    }

    // sync output var
    for (auto* var_node : op_node->outlinks) {
      CHECK(var_node->IsArg());
      CheckAndSyncTypeOfVarNode(var_node, ref_var_types);
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(
    control_flow_op_shared_inputs_and_outputs_place_sync_pass,
    paddle::lite::mir::ControlFlowOpSharedInputsAndOutputsPlaceSyncPass)
    .BindTargets({TARGET(kXPU)});
