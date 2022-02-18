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

#include "lite/core/optimizer/mir/control_flow_op_shared_inputs_and_outputs_data_sync_pass.h"
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

void CheckAndSyncDataOfVarNode(
    Node* sub_var_node,
    const std::unordered_map<std::string, bool>& ref_var_is_weights,
    const std::unordered_map<std::string, const lite::Tensor*>& ref_var_datas,
    Scope* scope) {
  CHECK(sub_var_node->IsArg());
  auto& sub_var_name = sub_var_node->AsArg().name;
  // sync weights
  if (ref_var_is_weights.count(sub_var_name)) {
    sub_var_node->AsArg().is_weight = ref_var_is_weights.at(sub_var_name);
  }
  // sync var data
  if (ref_var_datas.count(sub_var_name)) {
    auto sub_var = scope->FindVar(sub_var_name);
    auto sub_var_tensor = sub_var->GetMutable<lite::Tensor>();
    sub_var_tensor->CopyDataFrom(*ref_var_datas.at(sub_var_name));
  }
}

void ControlFlowOpSharedInputsAndOutputsDataSyncPass::SetAllGraphs(
    std::vector<std::unique_ptr<mir::SSAGraph>>* graphs) {
  CHECK(graphs && !graphs->empty());
  graphs_ = graphs;
}

void ControlFlowOpSharedInputsAndOutputsDataSyncPass::Apply(
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
    std::unordered_map<std::string, bool> ref_var_is_weights;
    std::unordered_map<std::string, const lite::Tensor*> ref_var_datas;
    auto* scope = op_node->AsStmt().op()->scope();
    for (auto* var_node : op_node->inlinks) {
      CHECK(var_node->IsArg());
      auto& var_name = var_node->AsArg().name;
      if (!ref_var_is_weights.count(var_name)) {
        ref_var_is_weights.insert(std::pair<std::string, bool>(
            var_name, var_node->AsArg().is_weight));
        if (var_node->AsArg().is_weight) {
          auto var = scope->FindVar(var_name);
          auto var_tensor = var->GetMutable<lite::Tensor>();
          ref_var_datas.insert(
              std::pair<std::string, lite::Tensor*>(var_name, var_tensor));
        }
      }
    }

    // sync input var
    for (auto& sub_op_node :
         (*graphs_)[sub_block_idx]->StmtTopologicalOrder()) {
      auto* sub_scope = sub_op_node->AsStmt().op()->scope();
      if (!sub_op_node->IsStmt()) continue;
      for (auto* sub_var_node : sub_op_node->inlinks) {
        CheckAndSyncDataOfVarNode(
            sub_var_node, ref_var_is_weights, ref_var_datas, sub_scope);
      }
      for (auto* sub_var_node : sub_op_node->outlinks) {
        auto& sub_var_name = sub_var_node->AsArg().name;
        if (!ref_var_is_weights.count(sub_var_name)) {
          ref_var_is_weights.insert(std::pair<std::string, bool>(
              sub_var_name, sub_var_node->AsArg().is_weight));
          if (sub_var_node->AsArg().is_weight) {
            auto sub_var = sub_scope->FindVar(sub_var_name);
            auto sub_var_tensor = sub_var->GetMutable<lite::Tensor>();
            ref_var_datas.insert(std::pair<std::string, lite::Tensor*>(
                sub_var_name, sub_var_tensor));
          }
        }
      }
    }

    // sync output var
    for (auto* var_node : op_node->outlinks) {
      CHECK(var_node->IsArg());
      CheckAndSyncDataOfVarNode(
          var_node, ref_var_is_weights, ref_var_datas, scope);
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(
    control_flow_op_shared_inputs_and_outputs_data_sync_pass,
    paddle::lite::mir::ControlFlowOpSharedInputsAndOutputsDataSyncPass)
    .BindTargets({TARGET(kNNAdapter)});
